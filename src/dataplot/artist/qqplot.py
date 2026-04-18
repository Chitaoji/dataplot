"""
Contains a plotter class: QQPlot.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import TYPE_CHECKING

import numpy as np
from scipy import stats
from validating import dataclass

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

    dist_or_sample: "DistName | np.ndarray | PlotDataSet"
    dots: int
    edge_precision: float
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
        self._plot_fitted_line(ax, q1, q2, is_multi)

    def _generate_dist(self) -> tuple[str, np.ndarray, np.ndarray]:
        if not 0 <= self.edge_precision < 0.5:
            raise ValueError(
                "'edge_precision' should be on the interval [0, 0.5), got "
                f"{self.edge_precision} instead"
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
                f"'dist_or_sample' can not be instance of {x.__class__.__name__!r}"
            )
        return xlabel, p, q

    @staticmethod
    def _plot_fitted_line(
        ax: "AxesWrapper", x: np.ndarray, y: np.ndarray, is_multi: bool
    ) -> None:
        a, b = linear_regression_1d(y, x)
        if is_multi:
            ax.ax.margins(x=0)
        else:
            ax.ax.margins(x=0.01)
        lb, ub = ax.ax.get_xlim()
        print(lb, ub)
        if lb == ub:
            lb, ub = x.min(), x.max()
        ax.ax.plot(
            [lb, ub], [a + lb * b, a + ub * b], "--", label=f"y = {a:.3f} + {b:.3f}x"
        )
        ax.ax.margins(0)

    @staticmethod
    def _get_ppf(dist: str, p: np.ndarray) -> np.ndarray:
        match dist:
            case "normal":
                return stats.norm.ppf(p)
            case "expon":
                return stats.expon.ppf(p)
            case _:
                raise ValueError(f"no such distribution: {dist!r}")
