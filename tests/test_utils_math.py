import unittest

import numpy as np
from src.dataplot.utils.math import get_prob, get_quantile, linear_regression_1d


class TestMathUtils(unittest.TestCase):
    def test_linear_regression_1d_ignores_nan_and_inf_values(self):
        x = np.array([1.0, 2.0, 3.0, np.nan, np.inf, -np.inf])
        y = np.array([3.0, 5.0, 7.0, 11.0, np.nan, np.inf])

        a, b = linear_regression_1d(y=y, x=x)

        self.assertTrue(np.isclose(a, 1.0))
        self.assertTrue(np.isclose(b, 2.0))

    def test_get_quantile_handles_infinite_values_as_nan(self):
        data = np.array([1.0, 2.0, 3.0, np.inf, -np.inf])
        p = np.array([0.0, 0.5, 1.0])

        quantiles = get_quantile(data, p)

        self.assertTrue(np.allclose(quantiles, np.array([1.0, 2.0, 3.0])))

    def test_get_prob_uses_only_finite_value_count(self):
        data = np.array([1.0, 2.0, 3.0, np.inf, -np.inf])
        q = np.array([0.5, 2.0, 10.0])

        probs = get_prob(data, q)

        self.assertTrue(np.allclose(probs, np.array([0.0, 2 / 3, 1.0])))
