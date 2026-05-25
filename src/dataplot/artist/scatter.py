"""
Contains a plotter class: ScatterChart.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from validating import dataclass

from ..container import _is_date_xaxis
from ..database import Data
from ..utils.math import linear_regression_1d
from ._color import darken_color
from .base import Plotter

if TYPE_CHECKING:
    from ..container import AxesWrapper
    from ..plottable import PlottableData

__all__ = ["ScatterPlot"]


@dataclass(validate_methods=True)
class ScatterPlot(Plotter):
    """
    A plotter class that creates a scatter chart.

    """

    xticks: Optional["PlottableData | Any"]
    fmt: str
    fit: bool

    def paint(self, ax: "AxesWrapper", **_) -> None:
        ax.set_axes(title=ax.get_setting("title", "Scatter Plot"))
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

        scatter_line = ax.ax.plot(
            xticks,
            self.data,
            self.fmt,
            linestyle="None",
            label=self.name,
            alpha=ax.settings.alpha,
        )[0]
        if self.fit:
            if _is_date_xaxis(ax.ax):
                raise ValueError("fit=True requires numeric x-ticks")
            try:
                fit_xticks = np.asarray(xticks, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("fit=True requires numeric x-ticks") from exc
            self._plot_fitted_line(
                ax, fit_xticks, self.data, scatter_color=scatter_line.get_color()
            )

    @staticmethod
    def _plot_fitted_line(
        ax: "AxesWrapper", x: np.ndarray, y: np.ndarray, scatter_color: str
    ) -> None:
        a, b = linear_regression_1d(y, x)
        lb, ub = ax.ax.get_xlim()
        if lb == ub:
            lb, ub = x.min(), x.max()
        ax.ax.plot(
            [lb, ub],
            [a + lb * b, a + ub * b],
            "-",
            color=darken_color(scatter_color),
            label=f"y = {a:.3f} + {b:.3f}x",
            alpha=0.5 if ax.settings.alpha is None else ax.settings.alpha / 2,
        )
