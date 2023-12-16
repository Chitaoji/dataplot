"""
Contains dataclasses: PlotSettings, PlotSetter and subclasses of PlotSetter.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""
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
SettingAvailable = Literal[
    "title", "xlabel", "ylabel", "alpha", "figsize", "style", "legend_loc"
]
StyleAvailable = Literal[
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
__all__ = ["PlotSettings", "PlotSetter", "DataSetter", "FigWrapper", "AxesWrapper"]


@define
class PlotSettings:
    """Stores and manages settings for plotting."""

    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    alpha: Optional[float] = None
    figsize: Optional[Tuple[int, int]] = None
    style: Optional[StyleAvailable] = None
    legend_loc: Optional[str] = None

    def __getitem__(self, __key: SettingAvailable) -> Any:
        return getattr(self, __key)

    def __setitem__(self, __key: SettingAvailable, __value: Any) -> None:
        setattr(self, __key, __value)

    @classmethod
    @property
    def available(cls) -> List[SettingAvailable]:
        """
        Available settings.

        Returns
        -------
        List[SettingAvailable]
            Names of the settings.

        """
        return get_args(SettingAvailable)

    def from_dict(self, d: Dict[SettingAvailable, Any]) -> None:
        """
        Reads settings from a dict.

        Parameters
        ----------
        d : Dict[SettingAvailable, Any]
            A dict of plot settings.

        """
        for k, v in d.items():
            setattr(self, k, v)

    def to_dict(self) -> Dict[SettingAvailable, Any]:
        """
        Returns a dict of the settings.

        Returns
        -------
        Dict[SettingAvailable]
            A dict of plot settings.

        """
        return {x: getattr(self, x) for x in self.available}


@define(init=False)
class PlotSetter:
    """Sets the settings for plotting."""

    settings: PlotSettings = field(default=Factory(PlotSettings), init=False)

    # pylint: disable=unused-argument
    def set_plot(
        self,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        alpha: Optional[float] = None,
        figsize: Optional[Tuple[int, int]] = None,
        style: Optional[StyleAvailable] = None,
        legend_loc: Optional[str] = None,
    ) -> Self:
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
        for key in self.settings.available:
            if (value := locals()[key]) is not None:
                self.settings[key] = value
        return self

    def set_plot_default(
        self,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        alpha: Optional[float] = None,
        figsize: Optional[Tuple[int, int]] = None,
        style: Optional[StyleAvailable] = None,
        legend_loc: Optional[str] = None,
    ) -> Self:
        """
        Set the default settings for plotting.

        Returns
        -------
        Self
            An instance of self.

        """
        for key in self.settings.available:
            if self.settings[key] is None:
                self.settings[key] = locals()[key]
        return self

    # pylint: enable=unused-argument
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
        self.set_plot(**settings.to_dict())
        return self

    def get_setting(
        self,
        key: SettingAvailable,
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
            obj.settings = PlotSettings(**self.settings.to_dict())
            for k, v in unmatched.items():
                setattr(obj, k, v)
            return obj
        raise ValueError(f"type {cls} cannot be customized by a PlotSetter")


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
    A wrapper of figure.

    Note that this should NEVER be instantiated directly, but always through the
    module-level function `dataplot.figure()`.

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
