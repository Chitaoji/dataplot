import matplotlib

matplotlib.use("Agg")
import unittest

import numpy as np

from src.dataplot.core import data, figure


class TestHistogram(unittest.TestCase):
    def test_hist_fit_uses_dense_grid_with_few_bins(self):
        ds = data(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), name="sample")
        artist = ds.hist(bins=4, density=True, fit="norm")

        with figure() as fig:
            artist.paint(fig.axes[0])
            ax = fig.axes[0].ax

        fit_line = ax.lines[0]
        fit_x = fit_line.get_xdata()

        self.assertGreaterEqual(len(fit_x), 512)
        self.assertTrue(np.isclose(fit_x[0], ax.patches[0].get_x()))
        expected_right_edge = ax.patches[-1].get_x() + ax.patches[-1].get_width()
        self.assertTrue(np.isclose(fit_x[-1], expected_right_edge))


if __name__ == "__main__":
    unittest.main()
