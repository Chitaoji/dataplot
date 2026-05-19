import matplotlib

matplotlib.use("Agg")
import unittest

import numpy as np

from src.dataplot.core import data, figure


class TestHexbin(unittest.TestCase):
    def test_hexbin_draws_polycollection(self):
        x = data(np.linspace(0, 1, 200), name="x")
        y = data(np.linspace(0, 1, 200) ** 2, name="y")

        artist = y.hexbin(x, gridsize=20, mincnt=1)

        with figure() as fig:
            artist.paint(fig.axes[0])
            ax = fig.axes[0].ax

        self.assertGreaterEqual(len(ax.collections), 1)

    def test_hexbin_raises_on_mismatched_lengths(self):
        x = data(np.array([1, 2, 3]), name="x")
        y = data(np.array([1, 2]), name="y")

        with self.assertRaises(ValueError):
            y.hexbin(x).paint()


if __name__ == "__main__":
    unittest.main()
