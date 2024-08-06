"""
Contains container classes: FigWrapper and AxesWrapper.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self, Unpack

import matplotlib.pyplot as plt
import numpy as np

from .plotter import PlotSettable

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.pyplot import Axes

    from ._typing import AxesSettingDict, FigureSettingDict, SettingKey

__all__ = ["FigWrapper", "AxesWrapper"]


@dataclass(slots=True)
class FigWrapper(PlotSettable):
    """
    A wrapper of figure.

    Note that this should NEVER be instantiated directly, but always through the
    module-level function `dataplot.figure()`.

    """

    nrows: int = 1
    ncols: int = 1
    active: bool = field(repr=False, default=True)
    entered: bool = field(init=False, repr=False, default=False)
    fig: "Figure" = field(init=False, repr=False)
    axes: list["AxesWrapper"] = field(init=False, repr=False)

    def __enter__(self) -> Self:
        """
        Create subplots and set the style.

        Returns
        -------
        Self
            An instance of self.

        """
        if not self.active:
            return self
        if self.entered:
            raise DoubleEnteredError(
                f"can't enter an instance of {self.__class__.__name__!r} for twice; "
                "please do all the operations in one single context manager"
            )

        self.set_default(
            style="seaborn-v0_8-darkgrid",
            figsize=(10 * self.ncols, 5 * self.nrows),
            subplots_adjust={"hspace": 0.5},
            fontdict={"fontsize": "x-large"},
        )
        plt.style.use(self.settings.style)
        self.fig, axes = plt.subplots(
            self.nrows, self.ncols, figsize=self.settings.figsize
        )
        self.axes: list["AxesWrapper"] = [AxesWrapper(x) for x in np.reshape(axes, -1)]
        self.entered = True
        return self

    def __exit__(self, *args) -> None:
        """
        Set various properties for the figure and paint it.

        """
        if not self.active:
            return

        if len(self.axes) > 1:
            self.fig.suptitle(self.settings.title, **self.settings.fontdict)
        else:
            self.axes[0].ax.set_title(self.settings.title, **self.settings.fontdict)

        self.fig.set_size_inches(*self.settings.figsize)
        self.fig.subplots_adjust(**self.settings.subplots_adjust)
        self.fig.set_dpi(self.get_setting("dpi", 100))

        for ax in self.axes:
            ax.exit()
        plt.show()
        plt.close(self.fig)
        plt.style.use("default")

    def set_figure(self, **kwargs: Unpack["FigureSettingDict"]) -> None:
        """
        Set the settings of figure.

        Parameters
        ----------
        title : str, optional
            Title of figure.
        dpi : float, optional
            Sets the resolution of figure in dots-per-inch.
        style : StyleStr, optional
            A style specification.
        figsize : tuple[int, int], optional
            Figure size, this takes a tuple of two integers that specifies the
            width and height of the figure in inches.
        fontdict : FontDict, optional
            A dictionary controlling the appearance of the title text.
        subplots_adjust : SubplotsParams, optional
            Adjusts the subplot layout parameters including: left, right, bottom,
            top, wspace, and hspace. See `SubplotsParams` for more details.

        """
        self._set(inplace=True, **kwargs)

    def setting_check(self, key: "SettingKey", value: Any) -> None:
        if self.entered and key == "style":
            logging.warning(
                "setting the '%s' of a figure has no effect unless it's done "
                "before invoking context manager",
                key,
            )


@dataclass(slots=True)
class AxesWrapper(PlotSettable):
    """
    Serves as a wrapper for creating and customizing axes in matplotlib.

    Note that this should NEVER be instantiated directly, but always
    through the invoking of `FigWrapper.axes`.

    """

    ax: "Axes"

    def set_axes(self, **kwargs: Unpack["AxesSettingDict"]) -> None:
        """
        Set the settings of axes.

        Parameters
        ----------
        title : str, optional
            Title of axes.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        alpha : float, optional
            Controls the transparency of the plotted elements. It takes a float
            value between 0 and 1, where 0 means completely transparent and 1
            means completely opaque.
        grid : bool, optional
            Determines whether to show the grids or not.
        grid_alpha : float, optional
            Controls the transparency of the grid.
        fontdict : FontDict, optional
            A dictionary controlling the appearance of the title text.
        legend_loc : LegendLocStr, optional
            Location of the legend.

        """
        self._set(inplace=True, **kwargs)

    def exit(self) -> None:
        """
        Set various properties for the axes. This should be called only
        by `FigWrapper.__exit__()`.

        """
        self.ax.set_xlabel(self.settings.xlabel)
        self.ax.set_ylabel(self.settings.ylabel)
        if len(self.ax.get_legend_handles_labels()[0]):
            self.ax.legend(loc=self.settings.legend_loc)
        if self.get_setting("grid", True):
            alpha = self.get_setting("alpha", 1.0)
            self.ax.grid(alpha=self.get_setting("grid_alpha", alpha / 2))
        else:
            self.ax.grid(False)
        self.ax.set_title(self.settings.title, **self.get_setting("fontdict", {}))


class DoubleEnteredError(Exception):
    """Raised when entering a Figwrapper for twice."""
