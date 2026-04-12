"""
Contains a plotter class: LineChart.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import TYPE_CHECKING, Optional

import numpy as np
from validating import dataclass

from ..setting import PlotSettable
from .base import Plotter

if TYPE_CHECKING:
    from ..container import AxesWrapper
    from ..dataset import PlotDataSet

__all__ = ["LineChart"]


@dataclass(validate_methods=True)
class LineChart(Plotter):
    """
    A plotter class that creates a line chart.

    """

    xticks: Optional["np.ndarray | PlotDataSet"]
    fmt: str
    scatter: bool
    sorted: bool
    rolling: Optional[int]

    def paint(self, ax: "AxesWrapper", **_) -> None:
        ax.set_axes(title=ax.get_setting("title", "Line Chart"))
        ax.load(self.settings)
        self.__plot(ax)

    def __plot(self, ax: "AxesWrapper") -> None:
        if isinstance(self.xticks, PlotSettable):
            xticks = self.xticks.data
        else:
            xticks = self.xticks
        if xticks is None:
            xticks = range(len(self.data))
        elif (len_t := len(xticks)) != (len_d := len(self.data)):
            raise ValueError(
                "x-ticks and data must have the same length, but have "
                f"lengths {len_t} and {len_d}"
            )

        if self.sorted:
            paired = sorted(
                zip(xticks, self.data, strict=True), key=lambda pair: pair[0]
            )
            xticks, data = zip(*paired, strict=True)
        else:
            data = self.data

        if self.rolling is not None:
            if self.rolling < 1:
                raise ValueError(
                    f"rolling must be a positive integer, got {self.rolling}"
                )
            data = np.convolve(
                np.asarray(data, dtype=float),
                np.ones(self.rolling, dtype=float),
                mode="full",
            )[: len(data)] / np.minimum(
                np.arange(1, len(data) + 1), self.rolling
            )

        ax.ax.plot(xticks, data, self.fmt, label=self.label)
        if self.scatter:
            ax.ax.scatter(xticks, data, zorder=2.0)
