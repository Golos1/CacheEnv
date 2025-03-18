from typing import SupportsFloat, Any
import gymnasium
import numpy
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame
from cacherl.utils.MultiDiscreteUnique import MultiDiscreteUnique
from matplotlib import pyplot

class CacheEnv(gymnasium.Env):
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
        return self._obs

    def _init_state(self):
        self._obs = self.observation_space.sample()
        self._state = list(self._obs)
        for i in range(len(self._state)):
            self._state[i] = list(self._state[i])
        self._state[1][2] = numpy.zeros(self.cache_size, dtype=int)

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
        cache_space = spaces.Tuple(
            (MultiDiscreteUnique(numpy.array([self.store_size for _ in range(cache_size)]),start=[0 for _ in range(cache_size)]),
            spaces.MultiDiscrete(numpy.array([max_ttl for _ in range(cache_size)]), start=[1 for _ in range(cache_size)]),
            spaces.MultiBinary(cache_size),
        ))
        previous_request_space = spaces.Tuple((
             spaces.Discrete(2),
             spaces.Discrete(store_size)
        ))
        request_mask = numpy.random.exponential(scale=1.0, size=self.store_size)
        request_mask = numpy.abs(request_mask)
        request_mask = request_mask / numpy.sum(request_mask)
        request_mask = tuple((numpy.array([0.5, 0.5]), request_mask))
        self._request_mask = request_mask
        self._request_space = previous_request_space
        self.action_space = spaces.OneOf([
            spaces.Tuple((
                spaces.Discrete(cache_size,start=0),
                spaces.Discrete(max_ttl,start=1)
        )),
            spaces.Discrete(1,start=-1)
        ])
        self.observation_space: spaces.Tuple = spaces.Tuple((
            previous_request_space,
             cache_space,
        ))
        self._init_state()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self._init_state()
        return self._obs, dict()

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self._state[0] = list(self._request_space.spaces[1].sample(probability=self._request_mask))
        reward = 0.0
        cached_rows: numpy.ndarray = self._state[1][0]
        requested_row = self._state[0][1]
        dirty_bits = self._state[1][2]
        ttls = self._state[1][1]
        if action[1] is dict:
            if self._state[0][0] == 0:
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
        self._obs = self._state
        for i in range(len(self._state)):
            self._obs[i] = tuple(self._state[i])
        self._obs = tuple(self._obs)
        return self._obs, reward, False, False, {"hits": self.hits, "misses": self.misses}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == "human":
            pyplot.plot(self.hits,self.misses)

