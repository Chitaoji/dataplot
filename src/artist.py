"""
Contains dataclasses: PlotSettings, Plotter and Artist.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NotRequired,
    Optional,
    Type,
    TypedDict,
    TypeVar,
    Unpack,
    get_args,
)

from attrs import Factory, define, field
from typing_extensions import Self

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .container import AxesWrapper

PlotSetterVar = TypeVar("PlotSetterVar", bound="Plotter")
DefaultVar = TypeVar("DefaultVar")
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

SettingKey = Literal[
    "title",
    "xlabel",
    "ylabel",
    "alpha",
    "dpi",
    "figsize",
    "style",
    "legend_loc",
    "subplots_adjust",
]


class SettingKwargs(TypedDict):
    """
    TypedDict for the keyword-arguments of `PlotSetter._set()`.

    """

    title: NotRequired[str]
    xlabel: NotRequired[str]
    ylabel: NotRequired[str]
    alpha: NotRequired[float]
    dpi: NotRequired[float]
    figsize: NotRequired[tuple[int, int]]
    style: NotRequired[StyleStr]
    legend_loc: NotRequired[str]
    subplots_adjust: NotRequired["SubplotParams"]


class SubplotParams(TypedDict):
    """
    Adjusts the subplot layout parameters.

    Unset parameters are left unmodified; initial values are given in
    `FigWrapper`.

    Parameters
    ----------
    left : float, optional
        The position of the left edge of the subplots, as a fraction of the
        figure width.
    bottom : float, optional
        The position of the bottom edge of the subplots, as a fraction of the
        figure height.
    right : float, optional
        The position of the right edge of the subplots, as a fraction of the
        figure width.
    top : float, optional
        The position of the top edge of the subplots, as a fraction of the
        figure height.
    wspace : float, optional
        The width of the padding between subplots, as a fraction of the average
        Axes width.
    hspace : float, optional
        The height of the padding between subplots, as a fraction of the average
        Axes height.

    """

    left: NotRequired[float]
    bottom: NotRequired[float]
    right: NotRequired[float]
    top: NotRequired[float]
    wspace: NotRequired[float]
    hspace: NotRequired[float]


__all__ = ["PlotSettings", "Plotter", "Artist"]


@define
class PlotSettings:
    """Stores and manages settings for plotting."""

    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    alpha: Optional[float] = None
    dpi: Optional[float] = None
    figsize: Optional[tuple[int, int]] = None
    style: Optional[StyleStr] = None
    legend_loc: Optional[str] = None
    subplots_adjust: Optional[SubplotParams] = None

    def __getitem__(self, __key: SettingKey) -> Any:
        return getattr(self, __key)

    def __setitem__(self, __key: SettingKey, __value: Any) -> None:
        setattr(self, __key, __value)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(" + self.repr_not_none() + ")"

    def repr_not_none(self) -> str:
        """
        Returns a string representation of attributes with not-None values.

        Returns
        -------
        str
            String representation.

        """
        diff = [f"{k}={repr(v)}" for k, v in self.asdict().items() if v is not None]
        return ", ".join(diff)

    @classmethod
    def keys(cls) -> list[SettingKey]:
        """
        Keys of settings.

        Returns
        -------
        List[SettingAvailable]
            Names of the settings.

        """
        return get_args(SettingKey)

    def fromdict(self, d: dict[SettingKey, Any]) -> None:
        """
        Reads settings from a dict.

        Parameters
        ----------
        d : Dict[SettingAvailable, Any]
            A dict of plot settings.

        """
        for k, v in d.items():
            setattr(self, k, v)

    def asdict(self) -> dict[SettingKey, Any]:
        """
        Returns a dict of the settings.

        Returns
        -------
        Dict[SettingAvailable]
            A dict of plot settings.

        """
        return {x: getattr(self, x) for x in self.keys()}


@define(init=False)
class Plotter:
    """Contains an attribute of plot settings, and provides methods for
    handling these settings.

    """

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
                if isinstance(v, dict):
                    self.settings[k] = {**self.settings[k], **v}
                else:
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
        Sets the default settings.

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
        default : DefaultVar, optional
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
        if issubclass(cls, Plotter):
            matched: dict[str, Any] = {}
            unmatched: dict[str, Any] = {}
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
class Artist(Plotter):
    """Paints images on an axes, based on the data, labels, and other settings."""

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
                raise ArtistPrepareError(f"'{name}' not set yet.")
        return self.on

    def paint(self, reflex: None = None) -> None:
        """Paint on the axes."""


class ArtistPrepareError(Exception):
    """Raised when data or labels are not set yet."""
