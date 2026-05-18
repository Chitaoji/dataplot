import unittest

import numpy as np
from src.dataplot.core import data


class TestDatasetResample(unittest.TestCase):
    def test_resample_with_last_first_mean_rules(self):
        ds = data([1.0, 2.0, 3.0, 4.0, 5.0])

        self.assertTrue(np.allclose(ds.resample(2, rule="last").data, [2.0, 4.0, 5.0]))
        self.assertTrue(np.allclose(ds.resample(2, rule="first").data, [1.0, 3.0, 5.0]))
        self.assertTrue(np.allclose(ds.resample(2, rule="mean").data, [1.5, 3.5, 5.0]))

    def test_resample_requires_positive_n(self):
        ds = data([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            ds.resample(0)

    def test_idx_selects_single_dataset_positions(self):
        ds = data(np.arange(10))

        indexed = ds.idx([3, 1, 3])

        self.assertTrue(np.array_equal(indexed.data, [3, 1, 3]))

    def test_joined_idx_uses_same_indices_for_each_dataset(self):
        joined = data(np.arange(10), np.arange(10) + 100)

        indexed = joined.idx([3, 1, 3])
        left, right = indexed.__multiobjects__

        self.assertTrue(np.array_equal(left.data, [3, 1, 3]))
        self.assertTrue(np.array_equal(right.data, [103, 101, 103]))

    def test_joined_random_sample_uses_same_indices_for_each_dataset(self):
        first = data(np.arange(10), np.arange(10) + 100)
        np.random.seed(123)

        sampled = first.sample(5, rule="random")
        left, right = sampled.__multiobjects__

        self.assertTrue(np.array_equal(right.data - left.data, np.full(5, 100)))

    def test_joined_random_sample_requires_equal_lengths(self):
        joined = data(np.arange(3), np.arange(4))

        with self.assertRaises(ValueError):
            joined.sample(2, rule="random")

