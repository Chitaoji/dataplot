"""
Contains an artist class: QQPlot.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import TYPE_CHECKING

import numpy as np
from attrs import define
from scipy import stats

from .artist import Artist, Plotter
from .container import AxesWrapper
from .utils.math import linear_regression_1d

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._typing import DistStr
    from .dataset import PlotDataSet

__all__ = ["QQPlot"]


@define
class QQPlot(Artist):
    """
    An artist class that creates a qqplot.

    """

    dist: "DistStr | NDArray | PlotDataSet" = "normal"
    quantiles: int = 30

    def paint(self, reflex: None = None) -> None:
        ax = self.prepare()
        ax.set_default(
            title="Quantile-Quantile Plot",
            ylabel=self.label + " quantiles",
        )
        self.__plot(ax.loading(self.settings))
        return reflex

    def __plot(self, ax: AxesWrapper) -> None:
        if isinstance(x := self.dist, str):
            ax.set_default(xlabel=x + " theoretical quantiles")
            match x:
                case "normal":
                    q = np.linspace(0, 1, self.quantiles + 2)[1:-1]
                    p1 = stats.norm.ppf(q)
                case "exponential":
                    q = np.linspace(0, 1, self.quantiles + 1)[0:-1]
                    p1 = stats.expon.ppf(q)
        elif isinstance(x, Plotter):
            ax.set_default(xlabel=x.label + " theoretical quantiles")
            q = np.linspace(0, 1, self.quantiles)
            p1 = self.__get_percentile(x.data, q)
        elif isinstance(x, (list, np.ndarray)):
            ax.set_default(xlabel="sample theoretical quantiles")
            q = np.linspace(0, 1, self.quantiles)
            p1 = self.__get_percentile(x, q)
        else:
            raise ValueError()

        p2 = self.__get_percentile(self.data, q)
        ax.ax.scatter(p1, p2, zorder=2.1)
        a, b = linear_regression_1d(p2, p1)
        l, r = p1.min(), p1.max()
        ax.ax.plot(
            [l, r], [a + l * b, a + r * b], "C1--", label=f"y = {a:.3f} + {b:.3f}x"
        )

    @staticmethod
    def __get_percentile(data, q):
        return np.nanpercentile(
            np.nan_to_num(data, posinf=np.nan, neginf=np.nan), q * 100
        )
