"""
Contains the dataset interface: PlotData.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from abc import ABCMeta
from dataclasses import dataclass, field
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Self,
    Unpack,
    overload,
)

import numpy as np
import pandas as pd

from .artist import Artist, CorrMap, Histogram, KSPlot, LineChart, PPPlot, QQPlot
from .plotter import PlotSettable, PlotSettings
from .utils.multi import MULITEM, REMAIN, MultiObject, multi, multipartial, single

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._typing import DistStr, ResampleRule, SettingDict
    from .artist import Plotter
    from .container import AxesWrapper


__all__ = ["PlotDataSet"]


@dataclass(slots=True)
class PlotDataSet(PlotSettable, metaclass=ABCMeta):
    """
    A dataset class providing methods for mathematical operations and plotting.

    Note that this should NEVER be instantiated directly, but always through the
    module-level function `dataplot.data()`.

    Parameters
    ----------
    data : NDArray
        Input data.
    label : str, optional
        Label of the data. If set to None, use "x1" as the label. By default None.

    Properties
    ----------
    fmt : str
        A string recording the mathmatical operations done on the data.
    original_data : NDArray
        Original input data.
    settings : PlotSettings
        Settings for plot (whether a figure or an axes).
    priority : int
        Priority of the latest mathmatical operation, where:
        0 : Highest priority, refering to `repr()` and some of unary operations;
        10 : Refers to binary operations that are prior to / (e.g., **);
        19 : Particularly refers to /;
        20 : Particularly refers to *;
        29 : Particularly refers to binary -;
        30 : Particularly refers to +;
        40 : Particularly refers to unary -.
            Note that / and binary - are distinguished from * or + because the
            former ones disobey the associative law.

    """

    data: "NDArray"
    label: Optional[str] = field(default=None)
    fmt_: str = field(init=False, default="{0}")
    original_data: "NDArray" = field(init=False)
    settings: PlotSettings = field(init=False, default_factory=PlotSettings)
    priority: int = field(init=False, default=0)

    @classmethod
    def __subclasshook__(cls, __subclass: type) -> bool:
        if __subclass is PlotDataSet or issubclass(__subclass, PlotDataSets):
            return True
        return False

    def __post_init__(self) -> None:
        self.label = "x1" if self.label is None else self.label
        self.original_data = self.data

    def __create(
        self, fmt: str, data: "NDArray", priority: int = 0, label: Optional[str] = None
    ) -> Self:
        obj = self.customize(
            self.__class__,
            self.original_data,
            self.label if label is None else label,
            fmt_=fmt,
            priority=priority,
        )
        obj.data = data
        return obj

    def __repr__(self) -> str:
        return self.__class__.__name__ + "\n- " + self.data_info()

    def data_info(self) -> str:
        """
        Information of dataset.

        Returns
        -------
        str
            A string indicating the data label and the plot settings.

        """
        not_none = self.settings.repr_not_none()
        return f"{self.formatted_label()}{': 'if not_none else ''}{not_none}"

    def __getitem__(self, __key: int) -> Self | Any:
        return MULITEM

    def __neg__(self) -> Self:
        new_fmt = f"(-{self.__auto_remove_brackets(self.fmt_, priority=28)})"
        new_data = -self.data
        return self.__create(new_fmt, new_data, priority=40)

    def __add__(self, __other: "float | int | PlotDataSet") -> Self:
        return self.__binary_operation(__other, "+", np.add, priority=30)

    def __radd__(self, __other: "float | int | PlotDataSet") -> Self:
        return self.__binary_operation(__other, "+", np.add, reverse=True, priority=30)

    def __sub__(self, __other: "float | int | PlotDataSet") -> Self:
        return self.__binary_operation(__other, "-", np.subtract, priority=29)

    def __rsub__(self, __other: "float | int | PlotDataSet") -> Self:
        return self.__binary_operation(
            __other, "-", np.subtract, reverse=True, priority=29
        )

    def __mul__(self, __other: "float | int | PlotDataSet") -> Self:
        return self.__binary_operation(__other, "*", np.multiply, priority=20)

    def __rmul__(self, __other: "float | int | PlotDataSet") -> Self:
        return self.__binary_operation(
            __other, "*", np.multiply, reverse=True, priority=20
        )

    def __truediv__(self, __other: "float | int | PlotDataSet") -> Self:
        return self.__binary_operation(__other, "/", np.true_divide, priority=19)

    def __rtruediv__(self, __other: "float | int | PlotDataSet") -> Self:
        return self.__binary_operation(
            __other, "/", np.true_divide, reverse=True, priority=19
        )

    def __pow__(self, __other: "float | int | PlotDataSet") -> Self:
        return self.__binary_operation(__other, "**", np.power)

    def __rpow__(self, __other: "float | int | PlotDataSet") -> Self:
        return self.__binary_operation(__other, "**", np.power, reverse=True)

    def __binary_operation(
        self,
        other: "float | int | PlotDataSet | Any",
        sign: str,
        func: Callable[[Any, Any], "NDArray"],
        reverse: bool = False,
        priority: int = 10,
    ) -> Self:
        if reverse:
            this_fmt = self.__auto_remove_brackets(self.fmt_, priority=priority)
            new_fmt = f"({other}{sign}{this_fmt})"
            new_data = func(other, self.data)
            return self.__create(new_fmt, new_data, priority=priority)

        this_fmt = self.__auto_remove_brackets(self.fmt_, priority=priority + 1)
        if isinstance(other, (float, int)):
            new_fmt = f"({this_fmt}{sign}{other})"
            new_data = func(self.data, other)
        elif isinstance(other, PlotDataSet):
            other_label = other.formatted_label(priority=priority)
            new_fmt = f"({this_fmt}{sign}{other_label})"
            new_data = func(self.data, other.data)
        else:
            raise ValueError(
                f"{sign!r} not supported between instances of 'PlotDataSet' and "
                f"{other.__class__.__name__!r}"
            )
        return self.__create(new_fmt, new_data, priority=priority)

    def __auto_remove_brackets(self, string: str, priority: int = 0):
        if priority == 0 or self.priority <= priority:
            return self.__remove_brackets(string)
        return string

    @staticmethod
    def __remove_brackets(string: str):
        if string.startswith("(") and string.endswith(")"):
            return string[1:-1]
        return string

    @property
    def fmt(self) -> str:
        """
        Return the format, but remove the pair of brackets at both ends of the
        string (if exists).

        Returns
        -------
        str
            Formatted label.

        """
        return self.__remove_brackets(self.fmt_)

    def formatted_label(self, priority: int = 0) -> str:
        """
        Return the formatted label, but remove the pair of brackets at both ends
        of the string if neccessary.

        Parameters
        ----------
        priority : int, optional
            Indicates whether to remove the brackets, by default 0.

        Returns
        -------
        str
            Formatted label.

        """
        if priority == self.priority and priority in (19, 29):
            priority -= 1
        return self.__auto_remove_brackets(
            self.fmt_.format(self.label), priority=priority
        )

    def join(self, *others: "PlotDataSet") -> Self:
        """
        Merge two or more `PlotDataSet` instances.

        Parameters
        ----------
        *others : PlotDataSet
            The instances to be merged.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        return PlotDataSets(self, *others)

    def resample(self, n: int, rule: "ResampleRule" = "head") -> Self:
        """
        Resample from the data.

        Parameters
        ----------
        n : int
            Length of new sample.
        rule : ResampleRule, optional
            Resample rule, by default "head".

        Returns
        -------
        Self
            A new instance of self.__class__.

        Raises
        ------
        ValueError
            Raised when receiving illegal rule.

        """
        new_fmt = f"resample({self.fmt}, {n})"
        match rule:
            case "random":
                idx = np.random.randint(0, len(self.data), n)
                new_data = self.data[idx]
            case "head":
                new_data = self.data[:n]
            case "tail":
                new_data = self.data[-n:]
            case _:
                raise ValueError(f"rule not supported: {rule!r}")
        return self.__create(new_fmt, new_data)

    def log(self) -> Self:
        """
        Perform a log operation on the data.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        new_fmt = f"log({self.fmt})"
        new_data = np.log(np.where(self.data > 0, self.data, np.nan))
        return self.__create(new_fmt, new_data)

    def signedlog(self) -> Self:
        """
        Perform a log operation on the data, but keep the sign.

        signedlog(x) =

        * log(x),   for x > 0;
        * 0,        for x = 0;
        * -log(-x), for x < 0.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        new_fmt = f"signedlog({self.fmt})"
        new_data = np.log(np.where(self.data > 0, self.data, np.nan))
        new_data[self.data < 0] = np.log(-self.data[self.data < 0])
        new_data[self.data == 0] = 0
        return self.__create(new_fmt, new_data)

    def signedpow(self, n: float) -> Self:
        """
        Perform a power operation on the data, but keep the sign.

        signedpow(x, n) =

        * x**n,     for x > 0;
        * 0,        for x = 0;
        * -x**(-n)  for x < 0.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        new_fmt = f"signedpow({self.fmt})"
        new_data = np.where(self.data > 0, self.data, np.nan) ** n
        new_data[self.data < 0] = -((-self.data[self.data < 0]) ** n)
        new_data[self.data == 0] = 0
        return self.__create(new_fmt, new_data)

    def rolling(self, n: int) -> Self:
        """
        Perform a rolling-mean operation on the data.

        Parameters
        ----------
        n : int
            Specifies the window size for calculating the rolling average of
            the data points.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        new_fmt = f"rolling({self.fmt}, {n})"
        new_data = pd.Series(self.data).rolling(n).mean().values
        return self.__create(new_fmt, new_data)

    def exp(self) -> Self:
        """
        Perform an exp operation on the data.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        new_fmt = f"exp({self.fmt})"
        new_data = np.exp(self.data)
        return self.__create(new_fmt, new_data)

    def abs(self) -> Self:
        """
        Perform an abs operation on the data.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        new_fmt = f"abs({self.fmt})"
        new_data = np.abs(self.data)
        return self.__create(new_fmt, new_data)

    def demean(self) -> Self:
        """
        Perform a demean operation on the data by subtracting its mean.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        new_fmt = f"({self.fmt}-mean({self.fmt}))"
        new_data = self.data - np.nanmean(self.data)
        return self.__create(new_fmt, new_data)

    def zscore(self) -> Self:
        """
        Perform a zscore operation on the data by subtracting its mean and then
        dividing by its standard deviation.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        new_fmt = f"zscore({self.fmt})"
        new_data = (self.data - np.nanmean(self.data)) / np.nanstd(self.data)
        return self.__create(new_fmt, new_data)

    def cumsum(self) -> Self:
        """
        Perform a cumsum operation on the data by calculating its cummulative
        sums.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        new_fmt = f"csum({self.fmt})"
        new_data = np.cumsum(self.data)
        return self.__create(new_fmt, new_data)

    def copy(self) -> Self:
        return self.__create(self.fmt_, self.data, priority=self.priority)

    def reset(self) -> Self:
        """
        Copy and reset the plot settings.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        obj = self.copy()
        obj.settings.reset()
        return obj

    def opclear(self) -> None:
        """
        Undo all the operations performed on the data and clean the records.

        """
        self.fmt_ = "{0}"
        self.data = self.original_data

    def opclear_records_only(self) -> None:
        """
        Clear the records of operations performed on the data. Differences to
        `opclear()` that the operations are not undone and the original data
        will be removed.

        """
        self.fmt_ = "{0}"
        self.original_data = self.data

    def set_label(self, __label: Optional[str] = None, **kwargs: str) -> Self:
        """
        Set the labels.

        Parameters
        ----------
        __label : str, optional
            The new label (if specified), by default None.
        **kwargs : str
            Works as a mapper to find the new label. If `self.label` is in
            `kwargs`, the label will be set to `kwargs[self.label]`.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        if isinstance(__label, str):
            new_label = __label
        elif self.label in kwargs:
            new_label = kwargs[self.label]
        else:
            new_label = self.label
        return self.__create(
            self.fmt_, self.data, priority=self.priority, label=new_label
        )

    @overload
    def set_plot(
        self, *, inplace: Literal[False] = False, **kwargs: Unpack["SettingDict"]
    ) -> Self: ...
    @overload
    def set_plot(
        self, *, inplace: Literal[True] = True, **kwargs: Unpack["SettingDict"]
    ) -> None: ...
    def set_plot(
        self, *, inplace: bool = False, **kwargs: Unpack["SettingDict"]
    ) -> Self | None:
        """
        Set the settings of a plot (whether a figure or an axes).

        Parameters
        ----------
        inplace : bool, optional
            Determines whether the changes of settings will happen in self or
            in a new copy of self, by default False.
        title : str, optional
            Title of plot.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        alpha : float, optional
            Controls the transparency of the plotted elements. It takes a float
            value between 0 and 1, where 0 means completely transparent and 1
            means completely opaque.
        dpi : float, optional
            Sets the resolution of figure in dots-per-inch.
        grid : bool, optional
            Determines whether to show the grids or not.
        grid_alpha : float, optional
            Controls the transparency of the grid.
        style : StyleStr, optional
            A style specification.
        figsize : tuple[int, int], optional
            Figure size, this takes a tuple of two integers that specifies the
            width and height of the figure in inches.
        fontdict : FontDict, optional
            A dictionary controlling the appearance of the title text.
        legend_loc : LegendLocStr, optional
            Location of the legend.

        Returns
        -------
        Self | None
            A new instance of self.__class__, or None.

        """
        return self._set(inplace=inplace, **kwargs)

    def batched(self, n: int = 1) -> Self:
        """
        If this instance is joined by multiple `PlotDataSet` objects, batch the
        objects into tuples of length n, otherwise return self.

        Use this together with `.plot()`, `.hist()`, etc.

        Parameters
        ----------
        n : int, optional
            Specifies the batch size, by default 1.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        if n <= 0:
            raise ValueError(f"batch size should be greater than 0, got {n} instead")
        return MultiObject([self])

    # pylint: disable=unused-argument
    def hist(
        self,
        bins: int | list[float] = 100,
        fit: bool = True,
        density: bool = True,
        same_bin: bool = True,
        stats: bool = True,
        *,
        ax: Optional["AxesWrapper"] = None,
    ) -> Artist:
        """
        Create a histogram of the data.

        Parameters
        ----------
        bins : int | list[float], optional
            Specifies the bins to divide the data into. If int, should be the number
            of bins. By default 100.
        fit : bool, optional
            Determines whether to fit a curve to the histogram, only available when
            `density=True`, by default True.
        density : bool, optional
            Determines whether to draw a probability density. If True, the histogram
            will be normalized such that the area under it equals to 1. By default
            True.
        same_bin : bool, optional
            Determines whether the bins should be the same for all sets of data, by
            default True.
        stats : bool, optional
            Determines whether to show the statistics, including the calculated mean,
            standard deviation, skewness, and kurtosis of the input, by default True.
        on : Optional[AxesWrapper], optional
            Specifies the axes-wrapper on which the plot should be painted. If
            not specified, the histogram will be plotted on a new axes in a new
            figure. By default None.

        Returns
        -------
        Artist
            An instance of Artist.

        """
        locals().update(only=self.__class__ is PlotDataSet)
        return self._get_artist(Histogram, locals())

    def plot(
        self,
        ticks: Optional["NDArray | PlotDataSet"] = None,
        fmt: str = "",
        scatter: bool = False,
        *,
        ax: Optional["AxesWrapper"] = None,
    ) -> Artist:
        """
        Create a line chart for the data. If there are more than one datasets, all of
        them should have the same length.

        Parameters
        ----------
        ticks : NDArray | PlotDataSet, optional
            Specifies the x-ticks for the line chart. If not provided, the x-ticks will
            be set to `range(len(data))`. By default None.
        fmt : str, optional
            A format string, e.g. 'ro' for red circles, by default ''.
        scatter : bool, optional
            Determines whether to include scatter points in the line chart, by default
            False.
        ax : AxesWrapper, optional
            Specifies the axes-wrapper on which the plot should be painted If
            not specified, the histogram will be plotted on a new axes in a new
            figure. By default None.

        Returns
        -------
        Artist
            An instance of Artist.

        """
        return self._get_artist(LineChart, locals())

    def qqplot(
        self,
        dist_or_sample: "DistStr | NDArray | PlotDataSet" = "normal",
        dots: int = 30,
        edge_precision: float = 1e-2,
        fmt: str = "o",
        *,
        ax: Optional["AxesWrapper"] = None,
    ) -> Artist:
        """
        Create a quantile-quantile plot.

        Parameters
        ----------
        dist_or_sample : DistStr | NDArray | PlotDataSet, optional
            Specifies the distribution to compare with. If str, specifies a
            theoretical distribution; if NDArray or PlotDataSet, specifies
            another real sample. By default 'normal'.
        dots : int, optional
            Number of dots, by default 30.
        edge_precision : float, optional
            Specifies the lowest quantile (`=edge_precision`) and the highest
            quantile (`=1-edge_precision`), by default 1e-2.
        fmt : str, optional
            A format string, e.g. 'ro' for red circles, by default 'o'.
        ax : AxesWrapper, optional
            Specifies the axes-wrapper on which the plot should be painted. If
            not specified, the histogram will be plotted on a new axes in a new
            figure. By default None.

        Returns
        -------
        Artist
            An instance of Artist.

        """
        return self._get_artist(QQPlot, locals())

    def ppplot(
        self,
        dist_or_sample: "DistStr | NDArray | PlotDataSet" = "normal",
        dots: int = 30,
        edge_precision: float = 1e-6,
        fmt: str = "o",
        *,
        ax: Optional["AxesWrapper"] = None,
    ) -> Artist:
        """
        Create a probability-probability plot.

        Parameters
        ----------
        dist_or_sample : DistStr | NDArray | PlotDataSet, optional
            Specifies the distribution to compare with. If str, specifies a
            theoretical distribution; if NDArray or PlotDataSet, specifies
            another real sample. By default 'normal'.
        dots : int, optional
            Number of dots, by default 30.
        edge_precision : float, optional
            Specifies the lowest quantile (`=edge_precision`) and the highest
            quantile (`=1-edge_precision`), by default 1e-6.
        fmt : str, optional
            A format string, e.g. 'ro' for red circles, by default 'o'.
        ax : AxesWrapper, optional
            Specifies the axes-wrapper on which the plot should be painted. If
            not specified, the histogram will be plotted on a new axes in a new
            figure. By default None.

        Returns
        -------
        Artist
            An instance of Artist.

        """
        return self._get_artist(PPPlot, locals())

    def ksplot(
        self,
        dist_or_sample: "DistStr | NDArray | PlotDataSet" = "normal",
        dots: int = 1000,
        edge_precision: float = 1e-6,
        fmt: str = "",
        *,
        ax: Optional["AxesWrapper"] = None,
    ) -> Artist:
        """
        Create a kolmogorov-smirnov plot.

        Parameters
        ----------
        dist_or_sample : DistStr | NDArray | PlotDataSet, optional
            Specifies the distribution to compare with. If str, specifies a
            theoretical distribution; if NDArray or PlotDataSet, specifies
            another real sample. By default 'normal'.
        dots : int, optional
            Number of dots, by default 1000.
        edge_precision : float, optional
            Specifies the lowest quantile (`=edge_precision`) and the highest
            quantile (`=1-edge_precision`), by default 1e-6.
        fmt : str, optional
            A format string, e.g. 'ro' for red circles, by default ''.
        ax : AxesWrapper, optional
            Specifies the axes-wrapper on which the plot should be painted. If
            not specified, the histogram will be plotted on a new axes in a new
            figure. By default None.

        Returns
        -------
        Artist
            An instance of Artist.

        """
        return self._get_artist(KSPlot, locals())

    def corrmap(
        self, annot: bool = True, *, ax: Optional["AxesWrapper"] = None
    ) -> Artist:
        """
        Create a correlation heatmap.

        Parameters
        ----------
        annot : bool, optional
            Specifies whether to write the data value in each cell, by default
            True.
        ax : AxesWrapper, optional
            Specifies the axes-wrapper on which the plot should be painted. If
            not specified, the histogram will be plotted on a new axes in a new
            figure. By default None.

        Returns
        -------
        Artist
            An instance of Artist.

        """
        return self._get_artist(CorrMap, locals())

    def _get_artist(self, cls: type["Plotter"], local: dict[str, Any]) -> Artist:
        params: dict[str, Any] = {}
        for key in cls.__init__.__code__.co_varnames[1:]:
            params[key] = local[key]
        plotter = self.customize(
            cls, data=self.data, label=self.formatted_label(), **params
        )
        artist = single(self.customize)(Artist, plotter=plotter)
        artist.paint(local["ax"])
        return artist

    # pylint: enable=unused-argument


class PlotDataSets(MultiObject[PlotDataSet]):
    """A duck subclass of `PlotDataSet`."""

    def __init__(self, *args: Any) -> None:
        if not args:
            raise ValueError("no args")
        objs: list[PlotDataSet] = []
        for a in args:
            if isinstance(a, self.__class__):
                objs.extend(a.__multiobjects__)
            elif isinstance(a, PlotDataSet):
                objs.append(a)
            else:
                raise TypeError(f"invalid type: {a.__class__.__name__!r}")
        super().__init__(objs, attr_reducer=self.__dataset_attr_reducer)

    def __repr__(self) -> str:
        data_info = "\n- ".join([x.data_info() for x in self.__multiobjects__])
        return f"{PlotDataSet.__name__}\n- {data_info}"

    def batched(self, n: int = 1) -> "MultiObject":
        """Overrides `PlotDataSet.batched()`."""
        PlotDataSet.batched(self, n)
        m = multi()
        for i in range(0, len(self.__multiobjects__), n):
            m.__multiobjects__.append(PlotDataSets(*self.__multiobjects__[i : i + n]))
        return m

    def __dataset_attr_reducer(self, n: str) -> Callable:
        match n:
            case (
                "hist"
                | "plot"
                | "ppplot"
                | "qqplot"
                | "ksplot"
                | "corrmap"
                | "join"
                | "_get_artist"
            ):
                return lambda _: partial(getattr(PlotDataSet, n), self)
            case "customize":
                return multipartial(
                    call_reducer=multipartial(
                        attr_reducer=multipartial(call_reflex="reflex")
                    )
                )
            case _ if n.startswith("_"):
                raise AttributeError(
                    f"cannot reach attribute '{n}' after dataset is joined"
                )
            case _:
                return multipartial(call_reducer=self.__join_if_dataset)

    @classmethod
    def __join_if_dataset(cls, x: list) -> Any:
        if x and isinstance(x[0], PlotDataSet):
            return cls(*x)
        return REMAIN
