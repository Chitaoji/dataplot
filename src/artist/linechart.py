"""
Contains a plotter class: LineChart.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..plotter import PlotSettable
from .base import Plotter

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..container import AxesWrapper
    from ..dataset import PlotDataSet

__all__ = ["LineChart"]


@dataclass(slots=True)
class LineChart(Plotter):
    """
    A plotter class that creates a line chart.

    """

    xticks: Optional["NDArray | PlotDataSet"]
    fmt: str
    scatter: bool

    def paint(
        self, ax: "AxesWrapper", reflex: None = None, __multi_last_call__: bool = False
    ) -> None:
        ax.set_default(title="Line Chart")
        ax.loading(self.settings)
        self.__plot(ax)
        return reflex

    def __plot(self, ax: "AxesWrapper") -> None:
        if isinstance(self.xticks, PlotSettable):
            xticks = self.xticks.data
        else:
            xticks = self.xticks
        if xticks is None:
            xticks = range(len(self.data))
            ax.ax.plot(self.data, self.fmt, label=self.label)
        elif (len_t := len(xticks)) == (len_d := len(self.data)):
            ax.ax.plot(xticks, self.data, self.fmt, label=self.label)
        else:
            raise ValueError(
                "x-ticks and data must have the same length, but have "
                f"lengths {len_t} and {len_d}"
            )
        if self.scatter:
            ax.ax.scatter(xticks, self.data, zorder=2.0)
