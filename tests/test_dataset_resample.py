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

