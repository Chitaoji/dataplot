"""
Contains a plotter class: LineChart.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd
from validating import dataclass

from ..database import Data
from ._ticks import ensure_rightmost_xtick_label
from .base import Plotter

if TYPE_CHECKING:
    from ..container import AxesWrapper
    from ..plottable import PlottableData

__all__ = ["LineChart"]


@dataclass(validate_methods=True)
class LineChart(Plotter):
    """
    A plotter class that creates a line chart.

    """

    xticks: Optional["PlottableData | Any"]
    fmt: str
    scatter: bool
    sorted: bool
    rolling: Optional[int | list[int]]

    def paint(self, ax: "AxesWrapper", **_) -> None:
            xlabel=ax.get_setting("xlabel", "count"),
        axes_settings = {
            "title": ax.get_setting("title", "Line Chart"),
            "ylabel": ax.get_setting("ylabel", "value"),
        }
        if self.xticks is None:
            axes_settings["xlabel"] = ax.get_setting("xlabel", "count")
        ax.set_axes(**axes_settings)
        ax.load(self.settings)
        self.__plot(ax)

    def __plot(self, ax: "AxesWrapper") -> None:
        if self.xticks is None:
            xticks = np.array(range(len(self.data)))
        elif isinstance(self.xticks, Data):
            xticks = self.xticks.data
        else:
            xticks = np.array(self.xticks)

        if (len_t := len(xticks)) != (len_d := len(self.data)):
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
            data_and_names = [(data_array, self.name)]
        else:
            data_and_names = [
                (
                    self.__rolling_mean(data_array, window),
                    f"rolling({self.name}, {window})" if window > 1 else self.name,
                )
                for window in rolling_list
            ]

        for rolling_data, name in data_and_names:
            ax.ax.plot(xticks, rolling_data, self.fmt, label=name)
            if self.scatter:
                ax.ax.scatter(xticks, rolling_data, zorder=2.0)

        # Disable matplotlib's default horizontal margins for tighter x-limits.
        ax.ax.margins(x=0)
        ensure_rightmost_xtick_label(ax, xticks)

    def __normalize_rolling(
        self, rolling: Optional[int | list[int]]
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
