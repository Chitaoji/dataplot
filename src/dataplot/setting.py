"""
Contains dataclasses: PlotSettings, PlotSettable.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from dataclasses import asdict
from typing import Any, Literal, Optional, Self, Unpack, overload

from validating import attr, dataclass

from ._typing import (
    FontDict,
    PlotSettableVar,
    SettingDict,
    SettingKey,
    StyleName,
    SubplotDict,
)

__all__ = ["PlotSettings", "PlotSettable", "defaults"]


@dataclass(validate_methods=True)
class PlotSettings:
    """Stores and manages settings for plotting."""

    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    alpha: Optional[float] = None
    dpi: Optional[int | float] = None
    grid: Optional[bool] = None
    grid_alpha: Optional[float] = None
    style: Optional[StyleName] = None
    figsize: Optional[tuple[int, int]] = None
    fontdict: Optional[FontDict] = None
    legend_loc: Optional[str] = None
    subplots_adjust: Optional[SubplotDict] = None

    @overload
    def __getitem__(
        self, __key: Literal["title", "xlabel", "ylabel"]
    ) -> Optional[str]: ...

    @overload
    def __getitem__(self, __key: Literal["alpha", "grid_alpha"]) -> Optional[float]: ...

    @overload
    def __getitem__(self, __key: Literal["dpi"]) -> Optional[int | float]: ...

    @overload
    def __getitem__(self, __key: Literal["grid"]) -> Optional[bool]: ...

    @overload
    def __getitem__(self, __key: Literal["style"]) -> Optional[StyleName]: ...

    @overload
    def __getitem__(self, __key: Literal["figsize"]) -> Optional[tuple[int, int]]: ...

    @overload
    def __getitem__(self, __key: Literal["fontdict"]) -> Optional[FontDict]: ...

    @overload
    def __getitem__(self, __key: Literal["legend_loc"]) -> Optional[str]: ...

    @overload
    def __getitem__(
        self, __key: Literal["subplots_adjust"]
    ) -> Optional[SubplotDict]: ...

    def __getitem__(self, __key: SettingKey) -> Any:
        return getattr(self, __key)

    @overload
    def __setitem__(
        self,
        __key: Literal["title", "xlabel", "ylabel", "legend_loc"],
        __value: Optional[str],
    ) -> None: ...

    @overload
    def __setitem__(
        self, __key: Literal["alpha", "grid_alpha"], __value: Optional[float]
    ) -> None: ...

    @overload
    def __setitem__(
        self, __key: Literal["dpi"], __value: Optional[int | float]
    ) -> None: ...

    @overload
    def __setitem__(self, __key: Literal["grid"], __value: Optional[bool]) -> None: ...

    @overload
    def __setitem__(
        self, __key: Literal["style"], __value: Optional[StyleName]
    ) -> None: ...

    @overload
    def __setitem__(
        self, __key: Literal["figsize"], __value: Optional[tuple[int, int]]
    ) -> None: ...

    @overload
    def __setitem__(
        self, __key: Literal["fontdict"], __value: Optional[FontDict]
    ) -> None: ...

    @overload
    def __setitem__(
        self, __key: Literal["subplots_adjust"], __value: Optional[SubplotDict]
    ) -> None: ...

    def __setitem__(self, __key: SettingKey, __value: Any) -> None:
        setattr(self, __key, __value)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(" + self._repr_changes() + ")"

    def _repr_changes(self) -> str:
        """
        Returns a string representation of attributes with not-None values.

        Returns
        -------
        str
            String representation.

        """
        diff = [f"{k}={repr(v)}" for k, v in asdict(self).items() if v is not None]
        return ", ".join(diff)

    def keys(self) -> list[SettingKey]:
        """
        Keys of settings.

        Returns
        -------
        list[SettingKey]
            Keys of the settings.

        """
        return getattr(self, "__match_args__")

    def reset(self) -> None:
        """
        Reset all the settings to None.

        """
        for k in self.keys():
            self[k] = None


defaults = PlotSettings(
    alpha=1.0,
    dpi=100,
    grid=True,
    grid_alpha=0.5,
    style="seaborn-v0_8-darkgrid",
    figsize=(10, 5),
    fontdict={"fontsize": "x-large"},
    subplots_adjust={"hspace": 0.5},
)


@dataclass(init=False)
class PlotSettable:
    """Contains an attribute of plot settings, and provides methods for
    handling these settings.

    """

    settings: PlotSettings = attr(default_factory=PlotSettings, init=False)

    def _set(
        self, *, inplace: bool = False, **kwargs: Unpack[SettingDict]
    ) -> Self | None:
        obj = self if inplace else self.copy()
        keys = obj.settings.keys()
        for k, v in kwargs.items():
            if v is None or k not in keys:
                continue
            if isinstance(v, dict) and isinstance(d := obj.settings[k], dict):
                d.update(v)
            else:
                obj.settings[k] = v
        if not inplace:
            return obj

    def load(self, settings: "PlotSettings | SettingDict") -> None:
        """
        Load in the settings.

        Parameters
        ----------
        settings : PlotSettings | SettingDict
            An instance of `PlotSettings` or a dict.

        """
        if isinstance(settings, PlotSettings):
            settings = asdict(settings)
        self._set(inplace=True, **settings)

    @overload
    def get_setting(
        self,
        key: Literal["title", "xlabel", "ylabel", "legend_loc"],
        default: Optional[str] = None,
    ) -> Optional[str]: ...

    @overload
    def get_setting(
        self, key: Literal["alpha", "grid_alpha"], default: Optional[float] = None
    ) -> Optional[float]: ...

    @overload
    def get_setting(
        self, key: Literal["dpi"], default: Optional[int | float] = None
    ) -> Optional[int | float]: ...

    @overload
    def get_setting(
        self, key: Literal["grid"], default: Optional[bool] = None
    ) -> Optional[bool]: ...

    @overload
    def get_setting(
        self, key: Literal["style"], default: Optional[StyleName] = None
    ) -> Optional[StyleName]: ...

    @overload
    def get_setting(
        self, key: Literal["figsize"], default: Optional[tuple[int, int]] = None
    ) -> Optional[tuple[int, int]]: ...

    @overload
    def get_setting(
        self, key: Literal["fontdict"], default: Optional[FontDict] = None
    ) -> Optional[FontDict]: ...

    @overload
    def get_setting(
        self, key: Literal["subplots_adjust"], default: Optional[SubplotDict] = None
    ) -> Optional[SubplotDict]: ...

    def get_setting(self, key: SettingKey, default: Any = ...) -> Any:
        """
        Returns the value of a setting if it is not None, otherwise returns the
        default value.

        Parameters
        ----------
        key : SettingKey
            Key of the setting.
        default : Any, optional
            Specifies the default value to be returned if the requested value
            is None. If omitted, falls back to ``defaults[key]``.

        Returns
        -------
        Any
            Value of the setting.

        """
        if default is ...:
            default = defaults[key]

        if (value := self.settings[key]) is None:
            return default
        if isinstance(value, dict) and isinstance(default, dict):
            return {**default, **value}
        return value

    def customize(
        self, cls: type["PlotSettableVar"], *args, **kwargs
    ) -> "PlotSettableVar":
        """
        Initialize another instance with the same settings as `self`.

        Parameters
        ----------
        cls : type[PlotSetableVar]
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
        if not issubclass(cls, PlotSettable):
            raise ValueError(f"type {cls} cannot be customized")
        matched: dict[str, Any] = {}
        unmatched: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in cls.__init__.__code__.co_varnames[1:]:
                matched[k] = v
            else:
                unmatched[k] = v
        obj = cls(*args, **matched)
        obj.settings = PlotSettings(**asdict(self.settings))
        for k, v in unmatched.items():
            setattr(obj, k, v)
        return obj

    def copy(self) -> Self:
        """
        Copy the instance of self (but not deepcopy).

        Returns
        -------
        Self
            A new instance of self.

        """
        raise TypeError(f"cannot copy instance of {self.__class__.__name__!r}")
