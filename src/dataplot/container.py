"""
Contains container classes: FigWrapper and AxesWrapper.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

import re
from typing import TYPE_CHECKING, Self, Unpack

import loggings
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes
from validating import attr, dataclass

from ._typing import AxesSettingDict, FigureSettingDict
from .setting import PlotSettable, defaults

if TYPE_CHECKING:
    from .artist import Artist

__all__ = ["FigWrapper", "AxesWrapper"]


def _parse_linear_expression(expr: str, var: str) -> tuple[float, float]:
    """
    Parse a linear expression into (intercept, slope), where
    expression = intercept + slope * var.
    """
    if not expr:
        raise ValueError("empty expression")
    if expr[0] not in "+-":
        expr = "+" + expr
    tokens = re.findall(r"[+-][^+-]+", expr)
    if "".join(tokens) != expr:
        raise ValueError(f"invalid expression: {expr!r}")
    intercept = 0.0
    slope = 0.0
    for token in tokens:
        sign = -1.0 if token[0] == "-" else 1.0
        body = token[1:]
        if body.endswith(var):
            coef = body[:-1]
            coef_value = 1.0 if coef == "" else float(coef)
            slope += sign * coef_value
            continue
        if "x" in body or "y" in body:
            raise ValueError(f"invalid expression: {expr!r}")
        intercept += sign * float(body)
    return intercept, slope


def _draw_reference_lines(ax: Axes, lines: list[str]) -> None:
    for text in lines:
        normalized = text.replace(" ", "")
        lhs, rhs = normalized.split("=")
        if lhs == "y":
            intercept, slope = _parse_linear_expression(rhs, "x")
            x0, x1 = ax.get_xlim()
            xs = np.linspace(x0, x1, 200)
            ys = intercept + slope * xs
            ax.plot(xs, ys, linestyle="--", linewidth=1.2, color="gray", alpha=0.85)
        else:
            intercept, slope = _parse_linear_expression(rhs, "y")
            y0, y1 = ax.get_ylim()
            ys = np.linspace(y0, y1, 200)
            xs = intercept + slope * ys
            ax.plot(xs, ys, linestyle="--", linewidth=1.2, color="gray", alpha=0.85)


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
            self.ax.legend(loc=self.get_setting("legend_loc"))
        if self.get_setting("grid"):
            alpha = self.get_setting("alpha")
            default_grid_alpha = (
                alpha / 2 if defaults.grid_alpha is None else defaults.grid_alpha
            )
            self.ax.grid(alpha=self.get_setting("grid_alpha", default_grid_alpha))
        else:
            self.ax.grid(False)
        self.ax.set_title(
            self.settings.title,
            **(self.get_setting("fontdict") or {}),
        )
        if lines := self.get_setting("reference_lines"):
            _draw_reference_lines(self.ax, lines)


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
    fig: Figure = attr(init=False, repr=False)
    axes: list[AxesWrapper] = attr(init=False, repr=False)
    artists: "list[Artist]" = attr(default_factory=list, init=False, repr=False)
    _copy: "FigWrapper | None" = attr(init=False, repr=False, default=None)

    def __repr__(self) -> str:
        with self as fig:
            for artist, ax in zip(self.artists, fig.axes[: len(self.artists)]):
                artist.paint(ax)
        return f"<{self.__class__.__name__} {self.nrows}x{self.ncols}>"

    def __enter__(self) -> Self:
        """
        Create subplots and set the style.

        Returns
        -------
        Self
            An instance of self.

        """
        if self._copy is not None:
            raise DoubleEnteredError(
                f"calling {self.__class__.__name__}.__enter__() for twice"
            )

        figw = self.copy()
        if not figw.active:
            self._copy = figw
            return figw

        plt.style.use(figw.get_setting("style"))
        figw.fig, axes = plt.subplots(figw.nrows, figw.ncols)
        figw.axes = [AxesWrapper(x) for x in np.reshape(axes, -1)]
        self._copy = figw
        return figw

    def __exit__(self, *args) -> None:
        """
        Set various properties for the figure and paint it.

        """
        figw = self._copy
        if figw is None:
            raise NotEnteredError(
                f"calling {self.__class__.__name__}.__exit__(...) before entering"
            )

        if not figw.active:
            self._copy = None
            return

        fontdict = figw.get_setting("fontdict") or {}
        if len(figw.axes) > 1:
            figw.fig.suptitle(figw.settings.title, **fontdict)
        else:
            figw.axes[0].ax.set_title(figw.settings.title, **fontdict)

        default_figsize = (
            defaults.figsize[0] * figw.ncols if defaults.figsize else 10 * figw.ncols,
            defaults.figsize[1] * figw.nrows if defaults.figsize else 5 * figw.nrows,
        )
        figw.fig.set_size_inches(*figw.get_setting("figsize", default_figsize))
        figw.fig.subplots_adjust(**(figw.get_setting("subplots_adjust") or {}))
        figw.fig.set_dpi(figw.get_setting("dpi"))

        for ax in figw.axes:
            ax.exit()
            if not ax.ax.has_data():
                figw.fig.delaxes(ax.ax)

        plt.show()
        plt.close(figw.fig)
        plt.style.use("default")

        self._copy = None

    def set_figure(self, **kwargs: Unpack[FigureSettingDict]) -> None:
        """
        Set the settings of figure.

        Parameters
        ----------
        title : str, optional
            Title of figure. Please note that there's another parameter with
            the same name in `.set_axes()`.
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
        if "style" in kwargs and (self._copy is not None):
            loggings.warning(
                "setting the 'style' of a figure has no effect unless it's done "
                "before invoking context manager",
            )
        self._set(inplace=True, **kwargs)

    def copy(self) -> Self:
        """Get a copy of self."""
        obj = self.customize(
            self.__class__,
            nrows=self.nrows,
            ncols=self.ncols,
            active=self.active,
        )
        obj.artists = self.artists
        obj._copy = None
        return obj

    def set_artists(self, *artist: "Artist") -> None:
        """Set artists."""
        self.artists = list(artist)


class DoubleEnteredError(Exception):
    """Raised when entering a Figwrapper for twice."""


class NotEnteredError(Exception):
    """Raised when exiting a Figwrapper that is not entered yet."""
