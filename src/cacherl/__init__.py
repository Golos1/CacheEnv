import gymnasium as gym
gym.register("Golos1/CacheEnv-v1", entry_point="cacherl.custom_env.CacheEnv:CacheEnv")
env = gym.make("Golos1/CacheEnv-v1")
