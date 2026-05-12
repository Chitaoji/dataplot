"""
Contains a plotter class: ScatterChart.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from validating import dataclass

from ..database import Data
from ._ticks import ensure_rightmost_xtick_label
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
        elif isinstance(self.xticks, Data):
            xticks = self.xticks.data
        else:
            xticks = np.array(self.xticks)

        if (len_t := len(xticks)) != (len_d := len(self.data)):
            raise ValueError(
                "x-ticks and data must have the same length, but have "
                f"lengths {len_t} and {len_d}"
            )

        ax.ax.plot(xticks, self.data, self.fmt, linestyle="None", label=self.name)
        ensure_rightmost_xtick_label(ax, xticks)
