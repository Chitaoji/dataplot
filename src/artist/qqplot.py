"""
Contains a plotter class: QQPlot.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import TYPE_CHECKING

import numpy as np
from attrs import define
from scipy import stats

from ..plotter import PlotSettable
from ..utils.math import get_quantile, linear_regression_1d
from .base import Plotter

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .._typing import DistStr
    from ..container import AxesWrapper
    from ..dataset import PlotDataSet

__all__ = ["QQPlot"]


@define
class QQPlot(Plotter):
    """
    A plotter class that creates a Q-Q plot.

    """

    dist_or_sample: "DistStr | NDArray | PlotDataSet"
    dots: int
    edge_precision: float
    fmt: str

    def paint(
        self, ax: "AxesWrapper", reflex: None = None, __multi_last_call__: bool = False
    ) -> None:
        ax.set_default(
            title="Quantile-Quantile Plot",
            xlabel="quantiles",
            ylabel="quantiles",
        )
        ax.loading(self.settings)
        self.__plot(ax)
        return reflex

    def __plot(self, ax: "AxesWrapper") -> None:
        xlabel, p, q1 = self._generate_dist()
        q2 = get_quantile(self.data, p)
        ax.ax.plot(q1, q2, self.fmt, zorder=2.1, label=f"{self.label} & {xlabel}")
        self._plot_fitted_line(ax, q1, q2)

    def _generate_dist(self) -> tuple[str, "NDArray", "NDArray"]:
        if self.edge_precision < 0 or self.edge_precision >= 0.5:
            raise ValueError(
                "edge_precision should be on the interval [0, 0.5), got "
                f"{self.edge_precision}"
            )
        p = np.linspace(self.edge_precision, 1 - self.edge_precision, self.dots)
        if isinstance(x := self.dist_or_sample, str):
            xlabel = x + "-distribution"
            q = self._get_ppf(x, p)
        elif isinstance(x, PlotSettable):
            xlabel = x.formatted_label()
            q = get_quantile(x.data, p)
        elif isinstance(x, (list, np.ndarray)):
            xlabel = "sample"
            q = get_quantile(x, p)
        else:
            raise TypeError(
                "argument 'dist_or_sample' expected to be str, NDArray, "
                f"or PlotDataSet, got {type(x)}"
            )
        return xlabel, p, q

    @staticmethod
    def _plot_fitted_line(ax: "AxesWrapper", x: "NDArray", y: "NDArray") -> None:
        a, b = linear_regression_1d(y, x)
        l, r = x.min(), x.max()
        ax.ax.plot(
            [l, r], [a + l * b, a + r * b], "--", label=f"y = {a:.3f} + {b:.3f}x"
        )

    @staticmethod
    def _get_ppf(dist: str, p: "NDArray") -> "NDArray":
        match dist:
            case "normal":
                return stats.norm.ppf(p)
            case "expon":
                return stats.expon.ppf(p)
