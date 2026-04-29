"""
Contains a plotter class: PPPlot.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import TYPE_CHECKING

from matplotlib.ticker import FuncFormatter
from validating import dataclass

from ..utils.math import get_prob
from .qqplot import QQPlot

if TYPE_CHECKING:
    from ..container import AxesWrapper

__all__ = ["PPPlot"]


@dataclass(validate_methods=True)
class PPPlot(QQPlot):
    """
    A plotter class that creates a P-P plot.

    """

    def paint(self, ax: "AxesWrapper", **_) -> None:
        ax.set_axes(
            title=ax.get_setting("title", "Probability-Probability Plot"),
            xlabel=ax.get_setting("xlabel", "baseline cumulative probility"),
            ylabel=ax.get_setting("ylabel", "sample cumulative probility"),
        )
        ax.load(self.settings)
        self.__plot(ax)
        return True

    def __plot(self, ax: "AxesWrapper") -> None:
        xlabel, p1, q = self._generate_dist(use_edge_precision=False)
        p2 = get_prob(self.data, q)
        ax.ax.plot(p1, p2, self.fmt, zorder=2.1, label=f"{self.name} & {xlabel}")
        ax.ax.set_xlim(0, 1)
        ax.ax.set_ylim(0, 1)
        ax.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: "" if abs(value) < 1e-12 else f"{value:.1f}")
        )
        self._plot_fitted_line(ax, p1, p2)
