import matplotlib

matplotlib.use("Agg")
import unittest

import numpy as np
from src.dataplot.container import _draw_reference_lines, _parse_linear_expression
from src.dataplot.core import data, figure


class TestCoreAndContainer(unittest.TestCase):
    def test_parse_linear_expression_for_x_and_y_forms(self):
        intercept, slope = _parse_linear_expression("1+2x-0.5x", "x")
        self.assertTrue(np.isclose(intercept, 1.0))
        self.assertTrue(np.isclose(slope, 1.5))

        intercept2, slope2 = _parse_linear_expression("-3+4y", "y")
        self.assertTrue(np.isclose(intercept2, -3.0))
        self.assertTrue(np.isclose(slope2, 4.0))

    def test_parse_linear_expression_invalid_input_raises(self):
        with self.assertRaises(ValueError):
            _parse_linear_expression("", "x")
        with self.assertRaises(ValueError):
            _parse_linear_expression("1+2z", "x")

    def test_draw_reference_lines_adds_visible_lines_only(self):
        figw = figure()
        with figw as fig:
            ax = fig.axes[0].ax
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            _draw_reference_lines(ax, ["y=x", "x=100"])  # only y=x is visible
            self.assertEqual(len(ax.lines), 1)

    def test_data_returns_single_and_multiple_datasets_with_labels(self):
        x = [1, 2, 3]
        ds_single = data(x, name="my_x")
        self.assertEqual(ds_single.name, "my_x")

        ds_multi = data([1, 2], [3, 4], name=["a", "b"])
        self.assertEqual([d.name for d in ds_multi.__multiobjects__], ["a", "b"])

    def test_data_label_validation_errors(self):
        with self.assertRaises(ValueError):
            data([1, 2], [3, 4], name="single-label")
        with self.assertRaises(ValueError):
            data([1, 2], name=["too", "many"])

    def test_figure_auto_grid_shape(self):
        fig1 = figure()
        self.assertEqual((fig1.nrows, fig1.ncols), (1, 1))

        fig2 = figure(nrows=1, ncols=1)
        self.assertEqual((fig2.nrows, fig2.ncols), (1, 1))
