import unittest

import numpy

from cacherl.utils.MultiDiscreteUnique import MultiDiscreteUnique


class MultiDiscreteSpaceTestCase(unittest.TestCase):
    def setUp(self):
        self.space = MultiDiscreteUnique(numpy.array([[5,5],[4,4]]))
        self.sample = self.space.sample()
    def test_uniqueness(self):
        self.assertEqual(len(self.sample.flatten()), len(set(self.sample.flatten())),"values are not unique")
    def test_contains(self):
        self.assertTrue(self.space.contains(numpy.array([[1,2],[3,0]])),"false negative on contains")
    def test_contains_non_unique(self):
        self.assertFalse(self.space.contains(numpy.array([[1,2],[2,3]])),"false positive on containing non unique values.")
    def test_contains_out_of_range(self):
        self.assertFalse(self.space.contains(numpy.array([[1,2],[3,4]])),"false positive on containing values out of range")
if __name__ == '__main__':
    unittest.main()
