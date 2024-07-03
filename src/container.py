"""
Contains container classes: FigWrapper and AxesWrapper.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

import logging
from typing import TYPE_CHECKING, Any, Optional, Self

import matplotlib.pyplot as plt
import numpy as np
from attrs import define, field

from .artist import Plotter

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.pyplot import Axes

    from ._typing import FontDict, LegendLocStr, SettingKey, StyleStr, SubplotDict

__all__ = ["FigWrapper", "AxesWrapper"]


@define
class FigWrapper(Plotter):
    """
    A wrapper of figure.

    Note that this should NEVER be instantiated directly, but always through the
    module-level function `dataplot.figure()`.

    """

    nrows: int = 1
    ncols: int = 1
    active: bool = True
    entered: bool = field(init=False, default=False)
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
        self.set_default(
            style="seaborn-v0_8-darkgrid",
            figsize=(10, 5),
            subplots_adjust={"hspace": 0.5},
            fontdict={"fontsize": "x-large"},
        )
        plt.style.use(self.settings.style)
        self.fig, axes = plt.subplots(
            self.nrows, self.ncols, figsize=self.settings.figsize
        )
        self.axes: list["AxesWrapper"] = [
            AxesWrapper(x) for x in np.array(axes).reshape(-1)
        ]
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

        self.active, self.entered = False, False

    def set_figure(
        self,
        title: Optional[str] = None,
        dpi: Optional[float] = None,
        figsize: Optional[tuple[int, int]] = None,
        style: Optional["StyleStr"] = None,
        fontdict: Optional["FontDict"] = None,
        subplots_adjust: Optional["SubplotDict"] = None,
    ) -> Self:
        """
        Set the settings of figure.

        Parameters
        ----------
        title : str, optional
            Title for the figure, by default None.
        dpi : float, optional
            Sets the resolution of the figure in dots-per-inch, by default None.
        figsize : tuple[int, int], optional
            Figure size, this takes a tuple of two integers that specifies the
            width and height of the figure in inches, by default None.
        style : StyleStr, optional
            A style specification, by default None.
        fontdict : FontDict, optional
            A dictionary controlling the appearance of the title text, by default
            None.
        subplots_adjust : SubplotsParams, optional
            Adjusts the subplot layout parameters including: left, right,
            bottom, top, wspace, and hspace, by default None. See `SubplotsParams`
            for more details.

        Returns
        -------
        Self
            An instance of self.

        """

        return self._set(
            title=title,
            dpi=dpi,
            figsize=figsize,
            style=style,
            fontdict=fontdict,
            subplots_adjust=subplots_adjust,
        )

    def setting_check(self, key: "SettingKey", value: Any) -> None:
        if self.entered and key == "style":
            logging.warning(
                "setting the '%s' of a figure has no effect unless it's done "
                "before invoking context manager",
                key,
            )


@define
class AxesWrapper(Plotter):
    """
    Serves as a wrapper for creating and customizing axes in matplotlib.

    Note that this should NEVER be instantiated directly, but always
    through the invoking of `FigWrapper.axes`.

    """

    ax: "Axes"

    def set_axes(
        self,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        alpha: Optional[float] = None,
        fontdict: Optional["FontDict"] = None,
        legend_loc: Optional["LegendLocStr"] = None,
    ) -> Self:
        """
        Set the settings of axes.

        Parameters
        ----------
        title : str, optional
            Title for the axes, by default None.
        xlabel : str, optional
            Label for the x-axis, by default None.
        ylabel : str, optional
            Label for the y-axis, by default None.
        alpha : float, optional
            Controls the transparency of the plotted elements. It takes a float
            value between 0 and 1, where 0 means completely transparent and 1
            means completely opaque, by default None.
        fontdict : FontDict, optional
            A dictionary controlling the appearance of the title text, by default
            None.
        legend_loc : LegendLocStr, optional
            Location of the legend, by default None.

        Returns
        -------
        Self
            An instance of self.

        """
        return self._set(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            alpha=alpha,
            fontdict=fontdict,
            legend_loc=legend_loc,
        )

    def exit(self) -> None:
        """
        Set various properties for the axes. This should be called only
        by `FigWrapper.__exit__()`.

        """
        self.set_default(fontdict={})

        self.ax.set_xlabel(self.settings.xlabel)
        self.ax.set_ylabel(self.settings.ylabel)
        if len(self.ax.get_legend_handles_labels()[0]):
            self.ax.legend(loc=self.settings.legend_loc)
        if (alpha := self.settings.alpha) is None:
            alpha = 1.0
        self.ax.grid(alpha=alpha / 2)
        self.ax.set_title(self.settings.title, **self.settings.fontdict)


class PlotterError(Exception):
    """Raised when data or labels are not set yet."""
