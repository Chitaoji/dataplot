"""
Contains a plotter class: KSPlot.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..utils.math import get_quantile
from .qqplot import QQPlot

if TYPE_CHECKING:
    from ..container import AxesWrapper


__all__ = ["KSPlot"]


@dataclass(slots=True)
class KSPlot(QQPlot):
    """
    A plotter class that creates a K-S plot.

    """

    def paint(
        self, ax: "AxesWrapper", reflex: None = None, __multi_last_call__: bool = False
    ) -> None:
        ax.set_default(
            title="Kolmogorov-Smirnov Plot",
            xlabel="value",
            ylabel="cummulative probability",
        )
        ax.loading(self.settings)
        self.__plot(ax)
        return reflex

    def __plot(self, ax: "AxesWrapper") -> None:
        xlabel, p, q1 = self._generate_dist()
        q2 = get_quantile(self.data, p)
        ax.ax.plot(q1, p, self.fmt, label=xlabel)
        ax.ax.plot(q2, p, self.fmt, label=self.label)
