"""
Contains a plotter class: KSPlot.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import TYPE_CHECKING

from validating import dataclass

from ..utils.math import get_quantile
from .qqplot import QQPlot

if TYPE_CHECKING:
    from ..container import AxesWrapper


__all__ = ["KSPlot"]


@dataclass(validate_methods=True)
class KSPlot(QQPlot):
    """
    A plotter class that creates a K-S plot.

    """

    def paint(self, ax: "AxesWrapper", **_) -> None:
        ax.set_axes(
            title=ax.get_setting("title", "Kolmogorov-Smirnov Plot"),
            xlabel=ax.get_setting("xlabel", "value"),
            ylabel=ax.get_setting("ylabel", "cummulative probability"),
        )
        ax.load(self.settings)
        self.__plot(ax)

    def __plot(self, ax: "AxesWrapper") -> None:
        xlabel, p, q1 = self._generate_dist()
        q2 = get_quantile(self.data, p)
        ax.ax.plot(q1, p, self.fmt, label=xlabel)
        ax.ax.plot(q2, p, self.fmt, label=self.label)
        ax.ax.margins(x=0)
