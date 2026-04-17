"""
Contains a plotter class: LineChart.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
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
    rolling: Optional[int | Iterable[int]]

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
            data_array = np.array(data, dtype=float)
        else:
            data_array = self.data

        rolling_list = self.__normalize_rolling(self.rolling)
        if rolling_list is None:
            data_and_labels = [(data_array, self.label)]
        elif len(rolling_list) == 1:
            window = rolling_list[0]
            data_and_labels = [(self.__rolling_mean(data_array, window), self.label)]
        else:
            data_and_labels = [
                (
                    self.__rolling_mean(data_array, window),
                    f"{self.label} (rolling={window})",
                )
                for window in rolling_list
            ]

        for rolling_data, label in data_and_labels:
            ax.ax.plot(xticks, rolling_data, self.fmt, label=label)
            if self.scatter:
                ax.ax.scatter(xticks, rolling_data, zorder=2.0)

    def __normalize_rolling(
        self, rolling: Optional[int | Iterable[int]]
    ) -> Optional[list[int]]:
        if rolling is None:
            return None
        if isinstance(rolling, int):
            if rolling < 1:
                raise ValueError(f"rolling must be a positive integer, got {rolling}")
            return [rolling]
        rolling_list = list(rolling)
        if not rolling_list:
            raise ValueError("rolling list cannot be empty")
        for n in rolling_list:
            if not isinstance(n, int) or n < 1:
                raise ValueError(
                    f"rolling items must be positive integers, got {rolling_list!r}"
                )
        return rolling_list

    def __rolling_mean(self, data: np.ndarray, n: int) -> np.ndarray:
        if n == 1:
            return data
        return pd.Series(data).rolling(window=n, min_periods=1).mean().to_numpy()
