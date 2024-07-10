"""
Contains an artist class: QQPlot.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import TYPE_CHECKING

import numpy as np
from attrs import define
from scipy import stats

from .artist import Artist
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
            xlabel=self.dist + " theoretical quantiles",
            ylabel=self.label + " quantiles",
        )
        self.__plot(ax.loading(self.settings))
        return reflex

    def __plot(self, ax: AxesWrapper) -> None:
        match self.dist:
            case "normal":
                q = np.linspace(0, 1, self.quantiles + 2)[1:-1]
                p1 = stats.norm.ppf(q)
            case "exponential":
                q = np.linspace(0, 1, self.quantiles + 1)[0:-1]
                p1 = stats.expon.ppf(q)
            case _:
                raise NotImplementedError("not implemented yet")
        p2 = np.nanpercentile(
            np.nan_to_num(self.data, posinf=np.nan, neginf=np.nan), q * 100
        )
        ax.ax.scatter(p1, p2)
        a, b = linear_regression_1d(p2, p1)
        ax.ax.plot([l := p1.min(), r := p1.max()], [a + l * b, a + r * b])
