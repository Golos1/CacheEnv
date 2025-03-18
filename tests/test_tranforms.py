import unittest
import gymnasium
from gymnasium.wrappers import TransformObservation
import cacherl.custom_env.CacheEnv
import numpy
from cacherl.utils.transforms import flatten_collection


class TransformsTestCase(unittest.TestCase):
    def setUp(self):
        env = gymnasium.make("CacheEnv-v1")
        self.env = TransformObservation(env, flatten_collection,observation_space=env.observation_space)

    def test_flatness(self):
        sample,info = self.env.reset()
        arr = numpy.array(sample)
        self.assertEqual(arr.shape[0], len(sample))
