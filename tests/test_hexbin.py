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


    def test_hexbin_sets_aspect_from_axis_ranges(self):
        x = data(np.linspace(0, 10, 200), name="x")
        y = data(np.linspace(-2, 2, 200), name="y")

        artist = y.hexbin(x, gridsize=20, mincnt=1)

        with figure() as fig:
            axis = fig.axes[0]
            axis.ax.set_xlim(0, 20)
            axis.ax.set_ylim(-5, 5)
            artist.paint(axis)
            ax = axis.ax

        self.assertAlmostEqual(float(ax.get_aspect()), 2.0)

    def test_hexbin_raises_on_mismatched_lengths(self):
        x = data(np.array([1, 2, 3]), name="x")
        y = data(np.array([1, 2]), name="y")

        with self.assertRaises(ValueError):
            y.hexbin(x).paint()


if __name__ == "__main__":
    unittest.main()
