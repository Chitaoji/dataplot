"""
Contains a tool class for plotting: LineChart.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""
from typing import Any, List, NewType, Optional

from attrs import define, field

from .setter import AxesWrapper, DataSetter

DateLike = NewType("DateLike", Any)

__all__ = ["LineChart"]


@define
class LineChart(DataSetter):
    """
    A plotting class that creates a line chart.

    """

    timestamps: Optional[List[DateLike]] = field(default=None, init=False)
    scatter: bool = False
    figsize_adjust: bool = True

    # def q(self, timestamps):
    #     self.timestamps = timestamps
    #     self._date: pd.DatetimeIndex = pd.to_datetime([str(x) for x in self.timestamps])

    def perform(self, reflex: None = None) -> None:
        """Do the plotting job."""
        with self.prepare() as ax:
            self.__plot(ax.loading(self.settings))
        return reflex

    def __plot(self, ax: AxesWrapper) -> None:
        ax.ax.plot(self.data, label=self.label)
        if self.scatter:
            ax.ax.scatter(self.data)
