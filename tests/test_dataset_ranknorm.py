import unittest

import numpy as np
from scipy.stats import norm
from src.dataplot.core import data


class TestDatasetRankNormalize(unittest.TestCase):
    def test_rank_normalize_maps_ranks_to_normal_scores(self):
        ds = data([10.0, 30.0, 20.0, 20.0])
        transformed = ds.ranknorm()

        expected_ranks = np.array([1.0, 4.0, 2.5, 2.5])
        p = (expected_ranks - 0.5) / 4
        expected = norm.ppf(p)
        self.assertTrue(np.allclose(transformed.data, expected))

    def test_rank_normalize_keeps_non_finite_as_nan(self):
        ds = data([1.0, np.nan, np.inf, -np.inf, 2.0])
        transformed = ds.ranknorm()

        self.assertTrue(np.isfinite(transformed.data[0]))
        self.assertTrue(np.isfinite(transformed.data[-1]))
        self.assertTrue(np.isnan(transformed.data[1]))
        self.assertTrue(np.isnan(transformed.data[2]))
        self.assertTrue(np.isnan(transformed.data[3]))
