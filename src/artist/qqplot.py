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
from ..utils.math import linear_regression_1d
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
    A plotter class that creates a qqplot.

    """

    dist_or_sample: "DistStr | NDArray | PlotDataSet"
    dots: int
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
        if isinstance(x := self.dist_or_sample, str):
            xlabel = x + "-distribution"
            match x:
                case "normal":
                    p = np.linspace(0, 1, self.dots + 2)[1:-1]
                    q1 = stats.norm.ppf(p)
                case "expon":
                    p = np.linspace(0, 1, self.dots + 1)[0:-1]
                    q1 = stats.expon.ppf(p)
        elif isinstance(x, PlotSettable):
            xlabel = x.formatted_label()
            p = np.linspace(0, 1, self.dots)
            q1 = self.__get_quantile(x.data, p)
        elif isinstance(x, (list, np.ndarray)):
            xlabel = "sample"
            p = np.linspace(0, 1, self.dots)
            q1 = self.__get_quantile(x, p)
        else:
            raise TypeError(
                "argument 'dist_or_sample' expected to be str, NDArray, "
                f"or PlotDataSet, got {type(x)}"
            )
        q2 = self.__get_quantile(self.data, p)
        ax.ax.plot(q1, q2, self.fmt, zorder=2.1, label=f"{self.label} & {xlabel}")
        a, b = linear_regression_1d(q2, q1)
        l, r = q1.min(), q1.max()
        ax.ax.plot(
            [l, r], [a + l * b, a + r * b], "--", label=f"y = {a:.3f} + {b:.3f}x"
        )

    @staticmethod
    def __get_quantile(data, q):
        return np.nanquantile(np.nan_to_num(data, posinf=np.nan, neginf=np.nan), q)
