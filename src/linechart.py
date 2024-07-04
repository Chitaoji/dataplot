"""
Contains an artist class: LineChart.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import TYPE_CHECKING, Optional

import numpy as np
from attrs import define

from .artist import Artist
from .container import AxesWrapper

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["LineChart"]


@define
class LineChart(Artist):
    """
    An artist class that creates a line chart.

    """

    ticks: Optional["NDArray[np.float64]"] = None
    scatter: bool = False

    def paint(self, reflex: None = None) -> None:
        ax = self.prepare()
        self.__plot(ax.loading(self.settings))
        return reflex

    def __plot(self, ax: AxesWrapper) -> None:
        if self.ticks is None:
            ax.ax.plot(self.data, label=self.label)
        elif (len_t := len(self.ticks)) == (len_d := len(self.data)):
            ax.ax.plot(self.ticks, self.data, label=self.label)
        else:
            raise ValueError(
                "Ticks and data must have the same length, but have "
                f"lengths {len_t} and {len_d}"
            )
        if self.scatter:
            ax.ax.scatter(self.data)
