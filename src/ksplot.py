"""
Contains an artist class: KSPlot.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import TYPE_CHECKING

import numpy as np
from attrs import define
from scipy import stats

from .artist import Artist, Plotter
from .container import AxesWrapper

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._typing import DistStr
    from .dataset import PlotDataSet

__all__ = ["KSPlot"]


@define
class KSPlot(Artist):
    """
    An artist class that creates a KS plot.

    """

    dist_or_sample: "DistStr | NDArray | PlotDataSet"
    dots: int
    edge_precision: float

    def paint(self, reflex: None = None) -> None:
        ax = self.prepare()
        ax.set_default(
            title="Kolmogorov-Smirnov Plot",
            xlabel="value",
            ylabel="cummulative probability",
        )
        ax.loading(self.settings)
        self.__plot(ax)
        return reflex

    def __plot(self, ax: AxesWrapper) -> None:
        p = np.linspace(self.edge_precision, 1 - self.edge_precision, self.dots)
        if isinstance(x := self.dist_or_sample, str):
            xlabel = x
            match x:
                case "normal":
                    q1 = stats.norm.ppf(p)
                case "exponential":
                    q1 = stats.expon.ppf(p)
        elif isinstance(x, Plotter):
            xlabel = x.formatted_label()
            q1 = self.__get_quantile(x.data, p)
        elif isinstance(x, (list, np.ndarray)):
            xlabel = "sample"
            q1 = self.__get_quantile(x, p)
        else:
            raise TypeError(
                "argument 'dist_or_sample' expected to be str, NDArray, "
                f"or PlotDataSet, got {type(x)}"
            )
        q2 = self.__get_quantile(self.data, p)
        ax.ax.plot(q2, p, label=self.label)
        ax.ax.plot(q1, p, label=xlabel)

    @staticmethod
    def __get_quantile(data, q):
        return np.nanquantile(np.nan_to_num(data, posinf=np.nan, neginf=np.nan), q)
