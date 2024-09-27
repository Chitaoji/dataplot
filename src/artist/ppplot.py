"""
Contains a plotter class: PPPlot.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..utils.math import get_prob
from .qqplot import QQPlot

if TYPE_CHECKING:
    from ..container import AxesWrapper

__all__ = ["PPPlot"]


@dataclass(slots=True)
class PPPlot(QQPlot):
    """
    A plotter class that creates a P-P plot.

    """

    def paint(
        self, ax: "AxesWrapper", reflex: None = None, __multi_last_call__: bool = True
    ) -> None:
        ax.set_default(
            title="Probability-Probability Plot",
            xlabel="cumulative probility",
            ylabel="cumulative probility",
        )
        ax.load(self.settings)
        self.__plot(ax)
        return reflex

    def __plot(self, ax: "AxesWrapper") -> None:
        xlabel, p1, q = self._generate_dist()
        p2 = get_prob(self.data, q)
        ax.ax.plot(p1, p2, self.fmt, zorder=2.1, label=f"{self.label} & {xlabel}")
        self._plot_fitted_line(ax, p1, p2)
