"""
Contains dataclasses: PlotSettings, PlotSetter and subclasses of PlotSetter.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    NotRequired,
    Optional,
    Type,
    TypedDict,
    TypeVar,
    Unpack,
    get_args,
)

import matplotlib.pyplot as plt
import numpy as np
from attrs import Factory, define, field
from typing_extensions import Self

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.pyplot import Axes
    from numpy.typing import NDArray


PlotSetterVar = TypeVar("PlotSetterVar", bound="PlotSetter")
DefaultVar = TypeVar("DefaultVar")
SettingKey = Literal[
    "title", "xlabel", "ylabel", "alpha", "figsize", "style", "legend_loc"
]
StyleStr = Literal[
    "Solarize_Light2",
    "_classic_test_patch",
    "_mpl-gallery",
    "_mpl-gallery-nogrid",
    "bmh",
    "classic",
    "dark_background",
    "fast",
    "fivethirtyeight",
    "ggplot",
    "grayscale",
    "seaborn-v0_8",
    "seaborn-v0_8-bright",
    "seaborn-v0_8-colorblind",
    "seaborn-v0_8-dark",
    "seaborn-v0_8-dark-palette",
    "seaborn-v0_8-darkgrid",
    "seaborn-v0_8-deep",
    "seaborn-v0_8-muted",
    "seaborn-v0_8-notebook",
    "seaborn-v0_8-paper",
    "seaborn-v0_8-pastel",
    "seaborn-v0_8-poster",
    "seaborn-v0_8-talk",
    "seaborn-v0_8-ticks",
    "seaborn-v0_8-white",
    "seaborn-v0_8-whitegrid",
    "tableau-colorblind10",
]
LegendLocStr = Literal[
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
]

__all__ = ["PlotSettings", "PlotSetter", "Plotter", "FigWrapper", "AxesWrapper"]


class SettingKwargs(TypedDict):
    """
    TypedDict for the keyword-arguments of settings.

    """

    title: NotRequired[str]
    xlabel: NotRequired[str]
    ylabel: NotRequired[str]
    alpha: NotRequired[float]
    figsize: NotRequired[tuple[int, int]]
    style: NotRequired[StyleStr]
    legend_loc: NotRequired[str]


@define
class PlotSettings:
    """Stores and manages settings for plotting."""

    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    alpha: Optional[float] = None
    figsize: Optional[tuple[int, int]] = None
    style: Optional[StyleStr] = None
    legend_loc: Optional[str] = None

    def __getitem__(self, __key: SettingKey) -> Any:
        return getattr(self, __key)

    def __setitem__(self, __key: SettingKey, __value: Any) -> None:
        setattr(self, __key, __value)

    def repr_not_none(self) -> str:
        """
        Returns a string representation of attributes with non-None values.

        Returns
        -------
        str
            String representation.

        """
        diff = [f"{k}={repr(v)}" for k, v in self.asdict().items() if v is not None]
        return ", ".join(diff)

    @classmethod
    def keys(cls) -> List[SettingKey]:
        """
        Keys of settings.

        Returns
        -------
        List[SettingAvailable]
            Names of the settings.

        """
        return get_args(SettingKey)

    def fromdict(self, d: Dict[SettingKey, Any]) -> None:
        """
        Reads settings from a dict.

        Parameters
        ----------
        d : Dict[SettingAvailable, Any]
            A dict of plot settings.

        """
        for k, v in d.items():
            setattr(self, k, v)

    def asdict(self) -> Dict[SettingKey, Any]:
        """
        Returns a dict of the settings.

        Returns
        -------
        Dict[SettingAvailable]
            A dict of plot settings.

        """
        return {x: getattr(self, x) for x in self.keys()}


@define(init=False)
class PlotSetter:
    """Sets the settings for plotting."""

    settings: PlotSettings = field(default=Factory(PlotSettings), init=False)

    # pylint: disable=unused-argument
    def _set(self, **kwargs: Unpack[SettingKwargs]) -> Self:
        """
        Set the settings.

        Parameters
        ----------
        **kwargs : Unpack[SettingKwargs]
            Specifies the settings.

        Returns
        -------
        Self
            An instance of self.

        """
        keys = self.settings.keys()
        for k, v in kwargs.items():
            if k in keys and v is not None:
                self.setting_check(k, v)
                self.settings[k] = v
        return self

    def setting_check(self, key: SettingKey, value: Any) -> None:
        """
        Checks if a new setting is legal.

        Parameters
        ----------
        key : SettingAvailable
            Key of the setting.
        value : Any
            Value of the setting.

        """

    def set_default(self, **kwargs: Unpack[SettingKwargs]) -> Self:
        """
        Set the default settings for plotting.

        Parameters
        ----------
        **kwargs : Unpack[SettingKwargs]
            Specifies the settings.

        Returns
        -------
        Self
            An instance of self.

        """
        keys = self.settings.keys()
        for k, v in kwargs.items():
            if k in keys and self.settings[k] is None:
                self.settings[k] = v
        return self

    # pylint: enable=unused-argument
    def loading(self, settings: PlotSettings) -> Self:
        """
        Load in the settings.

        Parameters
        ----------
        settings : PlotSettings
            An instance of `PlotSettings`.

        Returns
        -------
        Self
            An instance of self.
        """
        return self._set(**settings.asdict())

    def get_setting(
        self,
        key: SettingKey,
        default: Optional[DefaultVar] = None,
    ) -> DefaultVar:
        """
        Returns the value of a setting if it is not None, otherwise returns the
        default value.

        Parameters
        ----------
        key : SettingAvailable
            Key of the setting.
        default : Optional[DefaultVar], optional
            Specifies the default value to be returned if the requested value
            is None, by default None.

        Returns
        -------
        DefaultVar
            Value of the setting.

        """
        return default if (value := self.settings[key]) is None else value

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

        Raises
        ------
        ValueError
            Raised when `cls` cannot be customized.

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
            obj.settings = PlotSettings(**self.settings.asdict())
            for k, v in unmatched.items():
                setattr(obj, k, v)
            return obj
        raise ValueError(f"type {cls} cannot be customized")


@define(init=False, slots=False)
class Plotter(PlotSetter):
    """Plots the data and the label."""

    data: Optional["NDArray"] = field(default=None, init=False)
    label: Optional[str] = field(default=None, init=False)
    on: Optional["AxesWrapper"] = field(default=None, kw_only=True)

    def prepare(self) -> "AxesWrapper":
        """
        Prepares the data and the label. Raises an error if they are not
        passed in correctly.

        Returns
        -------
        AxesWrapper
            An instance of `AxesWrapper`.

        Raises
        ------
        DataSetterError
            Raised when the data or the label is not set yet.

        """
        for name in ["data", "label", "on"]:
            if getattr(self, name) is None:
                raise PlotterError(f"'{name}' not set yet.")
        return self.on

    def perform(self, reflex: None = None) -> None:
        """Do the plotting."""


@define
class FigWrapper(PlotSetter):
    """
    A wrapper of figure.

    Note that this should NEVER be instantiated directly, but always through the
    module-level function `dataplot.figure()`.

    """

    nrows: int = 1
    ncols: int = 1
    active: bool = True
    entered: bool = field(init=False, default=False)
    fig: "Figure" = field(init=False)
    axes: List["AxesWrapper"] = field(init=False)

    def __enter__(self) -> Self:
        """
        Creates subplots and sets the style.

        Returns
        -------
        Self
            An instance of self.

        """
        if not self.active:
            return self
        self.set_default(style="seaborn-v0_8-darkgrid", figsize=(10, 5))
        plt.style.use(self.settings.style)
        self.fig, axes = plt.subplots(
            self.nrows, self.ncols, figsize=self.settings.figsize
        )
        self.axes: List["AxesWrapper"] = [
            AxesWrapper(x) for x in np.array(axes).reshape(-1)
        ]
        self.entered = True
        return self

    def __exit__(self, *args) -> None:
        """
        Sets various properties for the figure and displays it.

        """
        if not self.active:
            return
        if len(self.axes) > 1:
            self.fig.suptitle(self.settings.title)
        else:
            self.axes[0].ax.set_title(self.settings.title)
        if self.settings.figsize is not None:
            self.fig.set_size_inches(*self.settings.figsize)

        for ax in self.axes:
            ax.exit()

        plt.show()
        plt.close(self.fig)
        plt.style.use("default")

    def set_figure(
        self,
        title: Optional[str] = None,
        figsize: Optional[tuple[int, int]] = None,
        style: Optional[StyleStr] = None,
    ) -> Self:
        """
        Sets the settings of figure.

        Parameters
        ----------
        title : str, optional
            The title for the figure, by default None.
        figsize : tuple[int, int], optional
            Figure size, this takes a tuple of two integers that specifies the
            width and height of the figure in inches, by default None.
        style : StyleStr, optional
            A style specification, by default None.

        Returns
        -------
        Self
            An instance of self.

        """

        return self._set(title=title, figsize=figsize, style=style)

    def setting_check(self, key: SettingKey, value: Any) -> None:
        if self.entered and key == "style":
            logging.warning(
                "setting the '%s' of a figure has no effect unless it's done "
                "before invoking context manager",
                key,
            )


@define
class AxesWrapper(PlotSetter):
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
        legend_loc: Optional[LegendLocStr] = None,
    ) -> Self:
        """
        Sets the settings of axes.

        Parameters
        ----------
        title : str, optional
            The title for the axes, by default None.
        xlabel : str, optional
            The label for the x-axis, by default None.
        ylabel : str, optional
            The label for the y-axis, by default None.
        alpha : float, optional
            Controls the transparency of the plotted elements. It takes a float
            value between 0 and 1, where 0 means completely transparent and 1
            means completely opaque, by default None.
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
            legend_loc=legend_loc,
        )

    def exit(self) -> None:
        """
        Sets various properties for the axes. This should be called only
        by `FigWrapper.__exit__()`.

        """
        self.ax.set_xlabel(self.settings.xlabel)
        self.ax.set_ylabel(self.settings.ylabel)
        self.ax.legend(loc=self.settings.legend_loc)
        if (alpha := self.settings.alpha) is None:
            alpha = 1.0
        self.ax.grid(alpha=alpha / 2)
        self.ax.set_title(self.settings.title)


class PlotterError(Exception):
    """Raised when data or labels are not set yet."""
