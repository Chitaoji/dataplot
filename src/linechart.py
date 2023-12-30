"""
Contains a tool class for plotting: LineChart.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""
from typing import TYPE_CHECKING, Optional

import numpy as np
from attrs import define

from .setter import AxesWrapper, DataSetter

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["LineChart"]


@define
class LineChart(DataSetter):
    """
    A plotting class that creates a line chart.

    """

    ticks: Optional["NDArray[np.float64]"] = None
    scatter: bool = False
    figsize_adjust: bool = True

    def perform(self, reflex: None = None) -> None:
        """Do the plotting job."""
        with self.prepare() as ax:
            self.__plot(ax.loading(self.settings))
        return reflex

    def __plot(self, ax: AxesWrapper) -> None:
        if self.ticks is None:
            ax.ax.plot(self.data, label=self.label)
        elif (len_t := len(self.ticks)) < (len_d := len(self.data)):
            ax.ax.plot(self.ticks, self.data[:len_t], label=self.label)
        elif len_t == len_d:
            ax.ax.plot(self.ticks, self.data, label=self.label)
        else:
            ax.ax.plot(self.ticks[:len_d], self.data, label=self.label)
        if self.scatter:
            ax.ax.scatter(self.data)


class TicksLenError(Exception):
    """Raised when the length of ticks is shorter than the data length."""
