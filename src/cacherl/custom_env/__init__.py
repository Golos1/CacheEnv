import gymnasium as gym
gym.register("CacheEnv-v1", entry_point="cacherl.custom_env.CacheEnv:CacheEnv")