from typing import SupportsFloat, Any
import gymnasium
import numpy
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame
from cacherl.custom_env.MultiDiscreteUnique import MultiDiscreteUnique
from matplotlib import pyplot

class CacheEnv(gymnasium.Env[ObsType: dict[str,dict],ActType: int | dict[str, int]]):
    """
    Simulates a variable TTL cache. At each step, a single value
     in a mock data store will either be updated or queried, and the agent must decide whether
     to cache the value from the previous request(and with what TTL) or not.
     Cache misses and dirty rows being fetched from the cache will yield negative reward,
      while cache hits will yield positive rewards. The agent is able to observe the cache itself,
     including the current items in the cache, their current TTLs, and whether they are dirty or not.

     """
    metadata = {"render_modes": ["human"]}
    def _get_obs(self):
        return self.state

    def __init__(self, cache_size=32, store_size=128, max_ttl: int = 50, render_mode: str = "human"):
        """
        :param cache_size: size of cache
        :param store_size: size of data store holding all data
        :param max_ttl: maximum Time to Live in simulated cache
        """
        if max_ttl <= 0:
            raise ValueError("TTL must be greater than 0.")
        if store_size <= 0:
            raise ValueError("Size of simulated data store must be greater than 0.")
        if cache_size <= 0:
            raise ValueError("Size of simulated cache must be greater than 0.")
        if cache_size > store_size:
            raise ValueError("Cache size cannot be greater than store size.")

        super(CacheEnv, self).__init__()
        self.render_mode = render_mode
        self.hits = [0]
        self.misses = [0]
        self.cache_size = int(cache_size)
        self.store_size = int(store_size)
        self.max_ttl = int(max_ttl)
        cache_space = spaces.Dict({
         "store_rows":   MultiDiscreteUnique(numpy.array([self.store_size for _ in range(cache_size)]),start=[0 for _ in range(cache_size)]),
            "current_ttl": spaces.MultiDiscrete(numpy.array([max_ttl for _ in range(cache_size)]), start=[1 for _ in range(cache_size)]),
            "dirty_bits": spaces.MultiBinary(cache_size)
        })
        previous_request_space = spaces.Dict({
            "get_or_post": spaces.Discrete(2),
            "store_row": spaces.Discrete(store_size)
        })
        request_mask = numpy.random.exponential(scale=1.0, size=self.store_size)
        request_mask = numpy.abs(request_mask)
        request_mask = request_mask / numpy.sum(request_mask)
        self.request_mask = request_mask
        self.request_mask = {
            "store_row": request_mask,
            "get_or_post": numpy.array([0.5,0.5])
        }
        self.action_space = spaces.OneOf([
            spaces.Dict({
                "cache_row": spaces.Discrete(cache_size,start=0),
                "ttl": spaces.Discrete(max_ttl,start=1)
            }),
            spaces.Discrete(1,start=-1)
        ])

        self.observation_space: spaces.Dict = spaces.Dict({
            "previous_request": previous_request_space,
            "cache": cache_space,
        })
        self.state: ObsType = self.observation_space.sample()
        self.state["cache"]["dirty_bits"] = numpy.zeros(cache_size,dtype=int)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self.state = self.observation_space.sample()
        self.state["cache"]["dirty_bits"] = numpy.zeros(self.cache_size,dtype=int)
        return self.state, dict()

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if action[1] != -1 and("cache_row" not in action[1] or "ttl" not in action[1]):
            raise ValueError("Action should either be -1 for not caching, or a dictionary from cache_row to ttl.")
        self.state["previous_request"] = self.observation_space.spaces["previous_request"].sample(probability=self.request_mask)
        reward = 0.0
        cached_rows: numpy.ndarray = self.state["cache"]["store_rows"]
        requested_row = self.state["previous_request"]["store_row"]
        dirty_bits = self.state["cache"]["dirty_bits"]
        ttls = self.state["cache"]["current_ttl"]
        if action[1] is dict:
            if self.state["previous_request"]["get_or_post"] == 0:
                row_is_cached = False
                for i in range(len(cached_rows)):
                    if requested_row == cached_rows[i]:
                        row_is_cached = True
                        if  dirty_bits[i] == 1:
                            reward -= 1
                        else:
                            reward += 0.5
                            current_hits = self.misses[len(self.hits) - 1] + 1
                            current_misses = self.misses[len(self.misses) - 1]
                            self.misses.append(current_misses)
                            self.hits.append(current_hits)
                        break
                if not row_is_cached:
                    reward -= 0.5
                    current_misses = self.misses[len(self.misses) - 1] + 1
                    current_hits = self.misses[len(self.hits) - 1]
                    self.hits.append(current_hits)
                    self.misses.append(current_misses)
            else:
                for i in range(len(cached_rows)):
                    if requested_row == cached_rows[i]:
                        dirty_bits[i] = 1
                current_misses = self.misses[len(self.misses) - 1]
                current_hits = self.misses[len(self.hits) - 1]
                self.hits.append(current_hits)
                self.misses.append(current_misses)
        for i in range(len(cached_rows)):
            if cached_rows[i] != self.store_size:
                ttls[i] -= 1
            if ttls[i] == 0:
                cached_rows[i] = self.store_size
        return self.state, reward, False, False, {"hits": self.hits, "misses": self.misses}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == "human":
            pyplot.plot(self.hits,self.misses)

