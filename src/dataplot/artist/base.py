"""
Contains the core of artist: Artist, Plotter.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from collections import Counter
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from validating import attr, dataclass

from ..container import FigWrapper
from ..setting import PlotSettable
from ..utils.math import linear_regression_1d
from ..utils.multi import MultiObject, multiple
from ._color import darken_color

if TYPE_CHECKING:
    from ..container import AxesWrapper

__all__ = ["Artist", "Plotter"]


@dataclass(validate_methods=True)
class Artist(PlotSettable):
    """
    Paint the desired images on an axes.

    Parameters
    ----------
    plotter : Plotter
        Painting tool.

    """

    plotter: "Plotter | MultiObject"

    def __repr__(self) -> str:
        self.paint()
        names = (x.__class__.__name__ for x in multiple(self.plotter))
        s = ", ".join(f"{i} {x}s" for x, i in Counter(names).items())
        return f"<{self.__class__.__name__} with {s}>"

    def paint(self, ax: Optional["AxesWrapper"] = None) -> None:
        """
        Paint on the axes.

        Parameters
        ----------
        ax : AxesWrapper, optional
            Specifies the axes-wrapper on which the plot should be painted,
            by default None.

        """
        with self.customize(FigWrapper, 1, 1, active := ax is None) as fig:
            self.plotter.paint(fig.axes[0] if active else ax)


@dataclass(init=False, validate_methods=True)
class Plotter(PlotSettable):
    """
    Painting tool used by Artist - user can ignore this.

    """

    data: Optional[np.ndarray] = attr(repr=False, default=None, init=False)
    name: Optional[str] = attr(default=None, init=False)

    def paint(
        self,
        ax: "AxesWrapper",
        *,
        __multi_prev_returned__: Any = None,
        __multi_is_final__: bool = True,
    ) -> Any:
        """Paint."""

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
