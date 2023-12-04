"""
Contains dataclasses: PlotSettings, PlotSetter and subclasses of PlotSetter.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

import matplotlib.pyplot as plt
import numpy as np
from attrs import Factory, define, field
from hintwith import hintwith
from typing_extensions import Self

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.pyplot import Axes
    from numpy.typing import NDArray


PlotSetterVar = TypeVar("PlotSetterVar", bound="PlotSetter")
DefaultVar = TypeVar("DefaultVar")

__all__ = ["PlotSettings", "PlotSetter", "DataSetter", "FigWrapper", "AxesWrapper"]


@define
class PlotSettings:
    """Stores and manages settings for plotting."""

    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    alpha: Optional[float] = None
    figsize: Optional[Tuple[int, int]] = None
    style: Optional[str] = None
    legend_loc: Optional[str] = None


@define(init=False)
class PlotSetter:
    """Sets the settings for plotting."""

    settings: PlotSettings = field(default=Factory(PlotSettings), init=False)

    @hintwith(PlotSettings.__init__)
    def set_plot(self, **kwargs) -> Self:
        """
        Sets the settings for plotting.

        Parameters
        ----------
        title : Union[str, None], optional
            The title for the axes, by default None.
        xlabel : Union[str, None], optional
            The label for the x-axis, by default None.
        ylabel : Union[str, None], optional
            The label for the y-axis, by default None.
        alpha : Union[float, None], optional
            Controls the transparency of the plotted elements. It takes a float
            value between 0 and 1, where 0 means completely transparent and 1
            means completely opaque. By default None.
        figsize : Optional[Tuple[int, int]], optional
            Figure size, this takes a tuple of two integers that specifies the
            width and height of the figure in inches, by default None.
        style : Optional[str], optional
            A style specification, by default None.
        legend_loc : Optional[str], optional
            Location of the legend, by default None.

        Returns
        -------
        Self
            An instance of self.

        """
        for key, value in kwargs.items():
            if value is not None:
                setattr(self.settings, key, value)
        return self

    @hintwith(PlotSettings.__init__)
    def set_plot_default(self, **kwargs) -> Self:
        """
        Set the default settings for plotting.

        Returns
        -------
        Self
            An instance of self.

        """
        for key, value in kwargs.items():
            if getattr(self.settings, key) is None:
                setattr(self.settings, key, value)
        return self

    def loading(self, settings: PlotSettings) -> Self:
        """
        Load in settings.

        Parameters
        ----------
        settings : PlotSettings
            An instance of `PlotSettings`.

        Returns
        -------
        Self
            An instance of self.
        """
        self.set_plot(
            **{
                key: getattr(settings, key)
                for key in settings.__init__.__code__.co_names
            }
        )
        return self

    def get_setting(
        self,
        key: Literal["title", "xlabel", "ylabel", "alpha", "figsize", "style"],
        default: Optional[DefaultVar] = None,
    ) -> DefaultVar:
        """
        Returns the value of a specified setting if it is not None, otherwise
        returns the default value.

        Parameters
        ----------
        key : Literal["title", "xlabel", "ylabel", "alpha", "figsize", "style"]
            Specifies the setting to be loaded.
        default : Optional[DefaultVar], optional
            Specifies the default value to be returned if the requested setting
            is None, by default None.

        Returns
        -------
        DefaultVar
            The value of the specified setting.

        """
        return default if (value := getattr(self.settings, key)) is None else value

    def customize(self, cls: Type[PlotSetterVar], *args, **kwargs) -> PlotSetterVar:
        """
        Initialize another instance with the same settings as `self`.

        Parameters
        ----------
        cls : Type[PlotSetableVar]
            Type of the new instance.
        *args :
            Positional arguments.
        **kwargs :
            Keyword arguments.

        Returns
        -------
        PlotSetableVar
            The new instance.

        """
        if issubclass(cls, PlotSetter):
            matched: Dict[str, Any] = {}
            unmatched: Dict[str, Any] = {}
            for k, v in kwargs.items():
                if k in cls.__init__.__code__.co_varnames[1:]:
                    matched[k] = v
                else:
                    unmatched[k] = v
            obj = cls(*args, **matched)
            obj.settings = deepcopy(self.settings)
            for k, v in unmatched.items():
                setattr(obj, k, v)
            return obj
        raise TypeError(f"type `{cls}` cannot be customized by `{self}`")


@define(init=False, slots=False)
class DataSetter(PlotSetter):
    """Sets the settings for data."""

    data: Optional["NDArray"] = field(default=None, init=False)
    label: Optional[str] = field(default=None, init=False)
    on: Optional["AxesWrapper"] = field(default=None, kw_only=True)

    def prepare(self) -> "AxesWrapper":
        """
        Prepares the data and labels and raises an error if they are not set
        correctly.

        Returns
        -------
        AxesWrapper
            An instance of `AxesWrapper`.

        Raises
        ------
        DataSetterError
            Raised when data or labels are not set yet.

        """
        for name in ["data", "label", "on"]:
            if getattr(self, name) is None:
                raise DataSetterError(f"'{name}' not set yet.")
        self.on: "AxesWrapper"
        return self.on


@define
class FigWrapper(PlotSetter):
    """
    Provides a context manager interface (`__enter__` and `__exit__` methods) for
    creating a figure with subplots and setting various properties for the figure.

    Parameters
    ----------
    nrows : int, optional
        Determines how many subplots can be arranged vertically in the figure,
        by default 1.
    ncols : int, optional
        Determines how many subplots can be arranged horizontally in the figure,
        by default 1.

    Returns
    -------
    FigWrapper
        A wrapper of figure.

    """

    nrows: int = 1
    ncols: int = 1
    fig: "Figure" = field(init=False)
    axes: List["AxesWrapper"] = field(init=False)

    def __enter__(self) -> Self:
        """
        Creates subplots and sets the style.

        Returns
        -------
            An instance of self.

        """
        self.set_plot_default(style="seaborn-v0_8-darkgrid", figsize=(10, 5))
        plt.style.use(self.settings.style)
        self.fig, axes = plt.subplots(
            self.nrows, self.ncols, figsize=self.settings.figsize
        )
        self.axes: List["AxesWrapper"] = [
            AxesWrapper(x) for x in np.array(axes).reshape(-1)
        ]
        return self

    def __exit__(self, *args) -> None:
        """
        Sets various properties for the figure and displays it.

        """
        self.axes[0].ax.set_title(self.settings.title)
        if self.settings.figsize is not None:
            self.fig.set_size_inches(*self.settings.figsize)
        plt.show()
        plt.close(self.fig)
        plt.style.use("default")


@define
class AxesWrapper(PlotSetter):
    """
    Serves as a wrapper for creating and customizing axes in matplotlib,
    providing a context manager interface (`__enter__` and `__exit__`
    methods) for setting various properties for the axes.

    """

    ax: "Axes"

    def __enter__(self) -> Self:
        """
        Enters the context manager.

        Returns
        -------
            An instance of self.

        """
        return self

    def __exit__(self, *args) -> None:
        """
        Sets various properties for the axes.

        """
        self.ax.set_xlabel(self.settings.xlabel)
        self.ax.set_ylabel(self.settings.ylabel)
        self.ax.legend(loc=self.settings.legend_loc)
        if (alpha := self.settings.alpha) is None:
            alpha = 1.0
        self.ax.grid(alpha=alpha / 2)
        self.ax.set_title(self.settings.title)


class DataSetterError(Exception):
    """Raised when data or labels are not set yet."""
