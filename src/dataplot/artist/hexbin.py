"""
Contains a plotter class: HexBin.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from validating import dataclass

from ..database import Data
from .base import Plotter

if TYPE_CHECKING:
    from ..container import AxesWrapper
    from ..plottable import PlottableData

__all__ = ["HexBinMap"]


@dataclass(validate_methods=True)
class HexBinMap(Plotter):
    """A plotter class that creates a hexbin chart."""

    xticks: Optional["PlottableData | Any"]
    gridsize: int
    cmap: str
    mincnt: Optional[int]

    def paint(self, ax: "AxesWrapper", **_) -> None:
        ax.set_axes(title=ax.get_setting("title", "Hexbin Map"))
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

        ax.ax.hexbin(
            xticks,
            self.data,
            gridsize=self.gridsize,
            cmap=self.cmap,
            mincnt=self.mincnt,
            alpha=ax.settings.alpha,
        )
        # Scale aspect by the current data ranges so x/y limits keep a stable
        # visual proportion for hex bins.
        xmin, xmax = ax.ax.get_xlim()
        ymin, ymax = ax.ax.get_ylim()
        if xmax > xmin and ymax > ymin:
            ax.ax.set_aspect((xmax - xmin) / (ymax - ymin), adjustable="datalim")
