import unittest
import gymnasium
import cacherl.custom_env.CacheEnv


def env_does_not_raise():
    try:
        result = gymnasium.make("CacheEnv-v1")
        return True
    except Exception:
        return False


class TransformsTestCase(unittest.TestCase):
    def test_flatness(self):
        self.assertTrue(env_does_not_raise())
