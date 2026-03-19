"""
Contains container classes: FigWrapper and AxesWrapper.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

import logging
from typing import TYPE_CHECKING, Any, Self, Unpack

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes
from validating import attr, dataclass

from ._typing import AxesSettingDict, FigureSettingDict, SettingKey
from .setting import PlotSettable

if TYPE_CHECKING:
    from .artist import Artist

__all__ = ["FigWrapper", "AxesWrapper"]


@dataclass(validate_methods=True)
class AxesWrapper(PlotSettable):
    """
    Serves as a wrapper for creating and customizing axes in matplotlib.

    Note that this should NEVER be instantiated directly, but always
    through the invoking of `FigWrapper.axes`.

    """

    ax: Axes

    def set_axes(self, **kwargs: Unpack[AxesSettingDict]) -> None:
        """
        Set the settings of axes.

        Parameters
        ----------
        title : str, optional
            Title of axes. Please note that there's another parameter with
            the same name in `.set_figure()`.
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
        legend_loc : LegendLoc, optional
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


@dataclass(validate_methods=True)
class FigWrapper(PlotSettable):
    """
    A wrapper of figure.

    Note that this should NEVER be instantiated directly, but always through the
    module-level function `dataplot.figure()`.

    """

    nrows: int = 1
    ncols: int = 1
    active: bool = attr(repr=False, default=True)
    entered: bool = attr(init=False, repr=False, default=False)
    fig: Figure = attr(init=False, repr=False)
    axes: list[AxesWrapper] = attr(init=False, repr=False)
    artists: "list[Artist]" = attr(default_factory=list, init=False, repr=False)
    _entered_copy: "FigWrapper | None" = attr(init=False, repr=False, default=None)

    def __enter__(self) -> Self:
        """
        Create subplots and set the style.

        Returns
        -------
        Self
            An instance of self.

        """
        if self._entered_copy is not None:
            raise DoubleEnteredError(
                f"can't enter an instance of {self.__class__.__name__!r} for twice; "
                "please do all the operations in one single context manager"
            )

        figw = self.copy()
        if not figw.active:
            self._entered_copy = figw
            return figw

        figw.set_default(
            style="seaborn-v0_8-darkgrid",
            figsize=(10 * figw.ncols, 5 * figw.nrows),
            subplots_adjust={"hspace": 0.5},
            fontdict={"fontsize": "x-large"},
        )
        plt.style.use(figw.settings.style)
        figw.fig, axes = plt.subplots(figw.nrows, figw.ncols)
        figw.axes = [AxesWrapper(x) for x in np.reshape(axes, -1)]
        figw.entered = True
        self._entered_copy = figw
        return figw

    def __exit__(self, *args) -> None:
        """
        Set various properties for the figure and paint it.

        """
        figw = self._entered_copy
        if figw is None:
            figw = self

        if not figw.active:
            self._entered_copy = None
            return

        if len(figw.axes) > 1:
            figw.fig.suptitle(figw.settings.title, **figw.settings.fontdict)
        else:
            figw.axes[0].ax.set_title(figw.settings.title, **figw.settings.fontdict)

        figw.fig.set_size_inches(*figw.settings.figsize)
        figw.fig.subplots_adjust(**figw.settings.subplots_adjust)
        figw.fig.set_dpi(figw.get_setting("dpi", 100))

        for ax in figw.axes:
            ax.exit()
            if not ax.ax.has_data():
                figw.fig.delaxes(ax.ax)

        plt.show()
        plt.close(figw.fig)
        plt.style.use("default")

        figw.entered = False
        self._entered_copy = None

    def __repr__(self) -> str:
        with self as fig:
            for artist, ax in zip(self.artists, fig.axes[: len(self.artists)]):
                artist.paint(ax)
        return f"<{self.__class__.__name__}(nrows={self.nrows}, ncols={self.ncols})>"

    def set_figure(self, **kwargs: Unpack[FigureSettingDict]) -> None:
        """
        Set the settings of figure.

        Parameters
        ----------
        title : str, optional
            Title of figure. Please note that there's another parameter with
            the same name in `.set_axis()`.
        dpi : float, optional
            Sets the resolution of figure in dots-per-inch.
        style : StyleName, optional
            A style specification.
        figsize : tuple[int, int], optional
            Figure size, this takes a tuple of two integers that specifies the
            width and height of the figure in inches.
        fontdict : FontDict, optional
            A dictionary controlling the appearance of the title text.
        subplots_adjust : SubplotDict, optional
            Adjusts the subplot layout parameters including: left, right, bottom,
            top, wspace, and hspace. See `SubplotDict` for more details.

        """
        self._set(inplace=True, **kwargs)

    def setting_check(self, key: SettingKey, value: Any) -> None:
        entered = self._entered_copy is not None or self.entered
        if entered and key == "style":
            logging.warning(
                "setting the '%s' of a figure has no effect unless it's done "
                "before invoking context manager",
                key,
            )

    def copy(self) -> Self:
        obj = self.customize(
            self.__class__,
            nrows=self.nrows,
            ncols=self.ncols,
            active=self.active,
        )
        obj.entered = self.entered
        obj.artists = list(self.artists)
        obj._entered_copy = None
        return obj


class DoubleEnteredError(Exception):
    """Raised when entering a Figwrapper for twice."""
