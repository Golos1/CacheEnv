from gymnasium.wrappers import FlattenObservation

from cacherl.custom_env.CacheEnv import CacheEnv

env = (CacheEnv())
print(tuple(env.observation_space.sample()))