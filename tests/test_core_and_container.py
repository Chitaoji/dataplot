import matplotlib

matplotlib.use("Agg")
import numpy as np

from dataplot.container import _draw_reference_lines, _parse_linear_expression
from dataplot.core import data, figure


def test_parse_linear_expression_for_x_and_y_forms():
    intercept, slope = _parse_linear_expression("1+2x-0.5x", "x")
    assert np.isclose(intercept, 1.0)
    assert np.isclose(slope, 1.5)

    intercept2, slope2 = _parse_linear_expression("-3+4y", "y")
    assert np.isclose(intercept2, -3.0)
    assert np.isclose(slope2, 4.0)


def test_draw_reference_lines_adds_visible_lines_only():
    figw = figure(active=True)
    with figw as fig:
        ax = fig.axes[0].ax
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        _draw_reference_lines(ax, ["y=x", "x=100"])  # only y=x is visible
        assert len(ax.lines) == 1


def test_data_returns_single_and_multiple_datasets_with_labels():
    x = [1, 2, 3]
    ds_single = data(x, label="my_x")
    assert ds_single.label == "my_x"

    ds_multi = data([1, 2], [3, 4], label=["a", "b"])
    assert [d.label for d in ds_multi.__multiobjects__] == ["a", "b"]
