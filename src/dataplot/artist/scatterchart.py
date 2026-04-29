"""
Contains a plotter class: ScatterChart.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from matplotlib.ticker import FixedLocator
from validating import dataclass

from ..setting import PlotSettable
from .base import Plotter

if TYPE_CHECKING:
    from ..container import AxesWrapper
    from ..plottable import PlottableData

__all__ = ["ScatterChart"]


@dataclass(validate_methods=True)
class ScatterChart(Plotter):
    """
    A plotter class that creates a scatter chart.

    """

    xticks: Optional["PlottableData | Any"]
    fmt: str

    def paint(self, ax: "AxesWrapper", **_) -> None:
        ax.set_axes(title=ax.get_setting("title", "Scatter Chart"))
        ax.load(self.settings)
        self.__plot(ax)

    def __plot(self, ax: "AxesWrapper") -> None:
        if self.xticks is None:
            xticks = np.array(range(len(self.data)))
        elif isinstance(self.xticks, PlotSettable):
            xticks = self.xticks.data
        else:
            xticks = np.array(self.xticks)

        if (len_t := len(xticks)) != (len_d := len(self.data)):
            raise ValueError(
                "x-ticks and data must have the same length, but have "
                f"lengths {len_t} and {len_d}"
            )

        ax.ax.plot(xticks, self.data, self.fmt, linestyle="None", label=self.label)
        self.__ensure_rightmost_xtick_label(ax, xticks)

    def __ensure_rightmost_xtick_label(self, ax: "AxesWrapper", xticks) -> None:
        xticks_array = np.asarray(list(xticks))
        if xticks_array.size == 0:
            return
        if not (
            np.issubdtype(xticks_array.dtype, np.number)
            or np.issubdtype(xticks_array.dtype, np.datetime64)
        ):
            return

        rightmost = float(ax.ax.convert_xunits(xticks_array[-1]))
        current_ticks = np.asarray(ax.ax.get_xticks(), dtype=float)
        if np.any(np.isclose(current_ticks, rightmost)):
            return

        merged_ticks = np.sort(np.append(current_ticks, rightmost))
        ax.ax.xaxis.set_major_locator(FixedLocator(merged_ticks))
