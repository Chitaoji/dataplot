"""
Contains a plotter class: QQPlot.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats
from validating import attr, dataclass

from .._typing import DistName
from ..setting import PlotSettable
from ..utils.math import get_quantile, linear_regression_1d
from .base import Plotter

if TYPE_CHECKING:
    from ..container import AxesWrapper
    from ..dataset import PlotDataSet

__all__ = ["QQPlot"]


@dataclass(validate_methods=True)
class QQPlot(Plotter):
    """
    A plotter class that creates a Q-Q plot.

    """

    dist_or_sample: "DistName | PlotDataSet | Any"
    dots: int
    edge_precision: float = attr(slb=0.0, sub=0.5)
    fmt: str

    def paint(
        self, ax: "AxesWrapper", __multi_prev_returned__: bool | None = None, **_
    ) -> bool:
        ax.set_axes(
            title=ax.get_setting("title", "Quantile-Quantile Plot"),
            xlabel=ax.get_setting("xlabel", "quantiles"),
            ylabel=ax.get_setting("ylabel", "quantiles"),
        )
        ax.load(self.settings)
        self.__plot(ax, __multi_prev_returned__)
        return True

    def __plot(self, ax: "AxesWrapper", is_multi: bool) -> None:
        xlabel, p, q1 = self._generate_dist()
        q2 = get_quantile(self.data, p)
        ax.ax.plot(q1, q2, self.fmt, zorder=2.1, label=f"{self.label} & {xlabel}")
        if is_multi:
            ax.ax.margins(x=0)
        else:
            ax.ax.margins(x=0.01)
        self._plot_fitted_line(ax, q1, q2)

    def _generate_dist(self) -> tuple[str, np.ndarray, np.ndarray]:
        p = np.linspace(self.edge_precision, 1 - self.edge_precision, self.dots)
        if isinstance(x := self.dist_or_sample, str):
            xlabel = x + "-distribution"
            q = self._get_ppf(x, p)
        elif isinstance(x, PlotSettable):
            xlabel = x.formatted_label()
            q = get_quantile(x.data, p)
        else:
            xlabel = "sample"
            q = get_quantile(np.array(x), p)
        return xlabel, p, q

    @staticmethod
    def _plot_fitted_line(ax: "AxesWrapper", x: np.ndarray, y: np.ndarray) -> None:
        a, b = linear_regression_1d(y, x)
        lb, ub = ax.ax.get_xlim()
        if lb == ub:
            lb, ub = x.min(), x.max()
        ax.ax.plot(
            [lb, ub], [a + lb * b, a + ub * b], "--", label=f"y = {a:.3f} + {b:.3f}x"
        )

    @staticmethod
    def _get_ppf(dist: DistName, p: np.ndarray) -> np.ndarray:
        match dist:
            case "normal":
                return stats.norm.ppf(p)
            case "expon":
                return stats.expon.ppf(p)
            case _:
                raise ValueError(f"no such distribution: {dist!r}")
