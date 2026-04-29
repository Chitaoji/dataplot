"""
Contains the dataset interface: PlotData.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from abc import ABCMeta
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
from scipy.stats import norm
from validating import attr, dataclass

from ._typing import DistName, ResampleRule, SettingDict
from .artist import (
    Artist,
    CorrMap,
    Histogram,
    KSPlot,
    LineChart,
    PPPlot,
    QQPlot,
    ScatterChart,
)
from .setting import PlotSettable, PlotSettings
from .utils.multi import (
    REMAIN,
    UNSUBSCRIPTABLE,
    MultiObject,
    multipartial,
    single,
)

if TYPE_CHECKING:
    from .artist import Plotter


__all__ = ["PlotDataSet"]


@dataclass(validate_methods=True)
class PlotDataSet(PlotSettable, metaclass=ABCMeta):
    """
    A dataset class providing methods for mathematical operations and plotting.

    Note that this should NEVER be instantiated directly, but always through the
    module-level function `dataplot.data()`.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    label : str, optional
        Label of the data. If set to None, use "x" as the label. By default None.

    Properties
    ----------
    fmt : str
        A string recording the mathmatical operations done on the data.
    original_data : np.ndarray
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

    data: np.ndarray
    label: Optional[str] = attr(default=None)
    fmtb: str = attr(init=False, default="{0}")
    original_data: np.ndarray = attr(init=False)
    settings: PlotSettings = attr(init=False, default_factory=PlotSettings)
    priority: int = attr(init=False, default=0)

    @classmethod
    def __subclasshook__(cls, __subclass: type) -> bool:
        if __subclass is PlotDataSet or issubclass(__subclass, PlotDataSets):
            return True
        return False

    def __post_init__(self) -> None:
        self.label = "x" if self.label is None else self.label
        self.original_data = self.data

    def __create(
        self, fmt: str, data: np.ndarray, priority: int = 0, label: Optional[str] = None
    ) -> Self:
        obj = self.customize(
            self.__class__,
            self.original_data,
            self.label if label is None else label,
            fmtb=fmt,
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
        not_none = self.settings._repr_changes()
        return f"{self.formatted_label()}{': ' if not_none else ''}{not_none}"

    def __getitem__(self, __key: int) -> Self | Any:
        return UNSUBSCRIPTABLE

    def __neg__(self) -> Self:
        new_fmt = f"(-{self.__remove_brackets(self.fmtb, priority=28)})"
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
        func: Callable[[Any, Any], np.ndarray],
        reverse: bool = False,
        priority: int = 10,
    ) -> Self:
        if reverse:
            this_fmt = self.__remove_brackets(self.fmtb, priority=priority)
            new_fmt = f"({other}{sign}{this_fmt})"
            new_data = func(other, self.data)
            return self.__create(new_fmt, new_data, priority=priority)

        this_fmt = self.__remove_brackets(self.fmtb, priority=priority + 1)
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

    def __remove_brackets(self, string: str, priority: int = 0):
        if priority == 0 or self.priority <= priority:
            if string.startswith("(") and string.endswith(")"):
                return string[1:-1]
        return string

    @property
    def format(self) -> str:
        """
        Return the label format.

        Returns
        -------
        str
            Label format.

        """
        return self.__remove_brackets(self.fmtb)

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
        return self.__remove_brackets(self.fmtb.format(self.label), priority=priority)

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

    def resample(self, n: int, rule: ResampleRule = "head") -> Self:
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
        new_fmt = f"resample({self.format}, {n})"
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
        new_fmt = f"log({self.format})"
        new_data = np.log(np.where(self.data > 0, self.data, np.nan))
        return self.__create(new_fmt, new_data)

    def log10(self) -> Self:
        """
        Perform a log operation on the data (with base 10).

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        new_fmt = f"log10({self.format})"
        new_data = np.log10(np.where(self.data > 0, self.data, np.nan))
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
        new_fmt = f"signedlog({self.format})"
        new_data = np.log(np.where(self.data > 0, self.data, np.nan))
        new_data[self.data < 0] = np.log(-self.data[self.data < 0])
        new_data[self.data == 0] = 0
        return self.__create(new_fmt, new_data)

    def signedlog10(self) -> Self:
        """
        Perform a log operation on the data, but keep the sign.

        signedlog10(x) =

        * log10(x),   for x > 0;
        * 0,        for x = 0;
        * -log10(-x), for x < 0.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        new_fmt = f"signedlog({self.format})"
        new_data = np.log10(np.where(self.data > 0, self.data, np.nan))
        new_data[self.data < 0] = np.log10(-self.data[self.data < 0])
        new_data[self.data == 0] = 0
        return self.__create(new_fmt, new_data)

    def pow(self, n: int | float = 2) -> Self:
        """
        Perform a power operation on the data.

        Parameters
        ----------
        n : int | float, optional
            Power exponent, by default 2.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        new_fmt = f"pow({self.format}, {n})"
        new_data = self.data**n
        return self.__create(new_fmt, new_data)

    def signedpow(self, n: int | float) -> Self:
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
        new_fmt = f"signedpow({self.format})"
        new_data = np.where(self.data > 0, self.data, np.nan) ** n
        new_data[self.data < 0] = -((-self.data[self.data < 0]) ** n)
        new_data[self.data == 0] = 0
        return self.__create(new_fmt, new_data)

    def root(self, n: int = 2) -> Self:
        """
        Perform an n-th root operation on the data.

        Parameters
        ----------
        n : int, optional
            Root degree, by default 2.

        Returns
        -------
        Self
            A new instance of self.__class__.

        Raises
        ------
        ValueError
            Raised when n is zero.

        """
        if n == 0:
            raise ValueError("root degree must not be zero")
        new_fmt = f"root({self.format}, {n})"
        new_data = np.power(self.data, 1 / n)
        return self.__create(new_fmt, new_data)

    def sqrt(self) -> Self:
        """
        Perform a square-root operation on the data.

        Equivalent to calling `root(2)`.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        return self.root(2)

    def signedroot(self, n: int = 2) -> Self:
        """
        Perform an n-th root operation on the data, but keep the sign.

        signedroot(x, n) =

        * x**(1/n),       for x > 0;
        * 0,              for x = 0;
        * -((-x)**(1/n)), for x < 0.

        Parameters
        ----------
        n : int, optional
            Root degree, by default 2.

        Returns
        -------
        Self
            A new instance of self.__class__.

        Raises
        ------
        ValueError
            Raised when n is zero.

        """
        if n == 0:
            raise ValueError("root degree must not be zero")
        new_fmt = f"signedroot({self.format}, {n})"
        power = 1 / n
        new_data = np.power(np.where(self.data > 0, self.data, np.nan), power)
        new_data[self.data < 0] = -np.power(-self.data[self.data < 0], power)
        new_data[self.data == 0] = 0
        return self.__create(new_fmt, new_data)

    def signedsqrt(self) -> Self:
        """
        Perform a square-root operation on the data, but keep the sign.

        Equivalent to calling `signedroot(2)`.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        return self.signedroot(2)

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
        if n < 1:
            raise ValueError(f"rolling window must be a positive integer, got {n}")
        new_fmt = f"rolling({self.format}, {n})"
        new_data = np.convolve(
            np.asarray(self.data, dtype=float), np.ones(n, dtype=float), mode="full"
        )[: len(self.data)] / np.minimum(np.arange(1, len(self.data) + 1), n)
        return self.__create(new_fmt, new_data)

    def exp(self) -> Self:
        """
        Perform an exp operation on the data.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        new_fmt = f"exp({self.format})"
        new_data = np.exp(self.data)
        return self.__create(new_fmt, new_data)

    def exp10(self) -> Self:
        """
        Perform a 10-based exponential operation on the data.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        new_fmt = f"exp10({self.format})"
        new_data = 10**self.data
        return self.__create(new_fmt, new_data)

    def signedexp(self) -> Self:
        """
        Perform an exp operation on the data, but keep the sign.

        signedexp(x) =

        * exp(x),   for x > 0;
        * 0,        for x = 0;
        * -exp(-x), for x < 0.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        new_fmt = f"signedexp({self.format})"
        new_data = np.exp(np.where(self.data > 0, self.data, np.nan))
        new_data[self.data < 0] = -np.exp(-self.data[self.data < 0])
        new_data[self.data == 0] = 0
        return self.__create(new_fmt, new_data)

    def signedexp10(self) -> Self:
        """
        Perform a 10-based exponential operation on the data, but keep the sign.

        signedexp10(x) =

        * 10**x,    for x > 0;
        * 0,        for x = 0;
        * -10**(-x), for x < 0.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        new_fmt = f"signedexp10({self.format})"
        new_data = 10 ** np.where(self.data > 0, self.data, np.nan)
        new_data[self.data < 0] = -(10 ** (-self.data[self.data < 0]))
        new_data[self.data == 0] = 0
        return self.__create(new_fmt, new_data)

    def abs(self) -> Self:
        """
        Perform an abs operation on the data.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        new_fmt = f"abs({self.format})"
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
        new_fmt = f"({self.format}-mean({self.format}))"
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
        new_fmt = f"zscore({self.format})"
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
        new_fmt = f"csum({self.format})"
        new_data = np.cumsum(self.data)
        return self.__create(new_fmt, new_data)

    def rank(self, pct: bool = True) -> Self:
        """
        Rank the data values.

        Tied values receive the average rank. Non-finite values (nan/inf) are
        kept as nan in the output.

        Parameters
        ----------
        pct : bool, optional
            If True, return percentage ranks in (0, 1] by dividing ranks by the
            number of finite observations. By default True.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        valid_mask = np.isfinite(self.data)
        valid_data = self.data[valid_mask]
        ranks = np.full(self.data.shape, np.nan, dtype=float)

        if valid_data.size > 0:
            order = np.argsort(valid_data, kind="mergesort")
            sorted_data = valid_data[order]
            ranked = np.empty(valid_data.size, dtype=float)
            change_points = np.flatnonzero(np.diff(sorted_data) != 0) + 1
            starts = np.concatenate(([0], change_points))
            ends = np.concatenate((change_points, [valid_data.size]))
            avg_ranks = (starts + 1 + ends) / 2.0
            ranked[order] = np.repeat(avg_ranks, ends - starts)

            if pct:
                ranked /= valid_data.size
            ranks[valid_mask] = ranked

        new_fmt = f"rank({self.format}, pct=True)" if pct else f"rank({self.format})"
        return self.__create(new_fmt, ranks)

    def rank_normalize(self) -> Self:
        """
        Rank-normalize the data values into standard normal scores.

        This applies a rank-based inverse normal transformation:
        1) compute average ranks for finite observations;
        2) convert to plotting probabilities using (rank - 0.5) / n;
        3) map probabilities through the standard normal inverse CDF.

        Non-finite values (nan/inf) are kept as nan in the output.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        ranks = self.rank(pct=False).data
        valid_mask = np.isfinite(ranks)
        new_data = np.full(self.data.shape, np.nan, dtype=float)

        if np.any(valid_mask):
            n = np.count_nonzero(valid_mask)
            p = (ranks[valid_mask] - 0.5) / n
            new_data[valid_mask] = norm.ppf(p)

        new_fmt = f"rank_normalize({self.format})"
        return self.__create(new_fmt, new_data)

    def copy(self) -> Self:
        return self.__create(self.fmtb, self.data, priority=self.priority)

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

    def undo_all(self) -> None:
        """
        Undo all the operations performed on the data and clean the records.

        """
        self.fmtb = "{0}"
        self.data = self.original_data

    def set_label(
        self, label: Optional[str] = None, reset_format: bool = True, /, **kwargs: str
    ) -> Self:
        """
        Set the labels.

        Parameters
        ----------
        label : str, optional
            The new label (if specified), by default None.
        reset_format : bool, optional
            Determines whether to reset the format of the label (which shows
            the operations done on the data), by default True.
        **kwargs : str
            Works as a mapper to find the new label. If `self.label` is in
            `kwargs`, the label will be set to `kwargs[self.label]`.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        if isinstance(label, str):
            new_label = label
        elif self.label in kwargs:
            new_label = kwargs[self.label]
        else:
            new_label = self.label
        return self.__create(
            "{0}" if reset_format else self.fmtb,
            self.data,
            priority=self.priority,
            label=new_label,
        )

    @overload
    def set_plot(
        self, *, inplace: Literal[False] = False, **kwargs: Unpack[SettingDict]
    ) -> Self: ...
    @overload
    def set_plot(
        self, *, inplace: Literal[True] = True, **kwargs: Unpack[SettingDict]
    ) -> None: ...
    def set_plot(
        self, *, inplace: bool = False, **kwargs: Unpack[SettingDict]
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
        style : StyleName, optional
            A style specification.
        figsize : tuple[int, int], optional
            Figure size, this takes a tuple of two integers that specifies the
            width and height of the figure in inches.
        fontdict : FontDict, optional
            A dictionary controlling the appearance of the title text.
        legend_loc : LegendLoc, optional
            Location of the legend.
        format_label : bool, optional
            Determines whether to format the label (to show the operations done
            on the data).
        subplots_adjust : SubplotDict, optional
            Adjusts the subplot layout parameters including: left, right, bottom,
            top, wspace, and hspace. See `SubplotDict` for more details.
        reference_lines : list[str], optional
            Reference line expressions to draw on the axes. Each expression
            should use the format ``"y=..."`` or ``"x=..."`` (for example,
            ``"y=0"``, ``"x=10"``, ``"y=2x+1"``), and the lines are rendered
            as dashed gray guides.

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

    def hist(
        self,
        bins: int | list[float] = 100,
        density: bool = True,
        log: bool = False,
        same_bin: bool = True,
        fit: Literal["norm", "skew-norm", "t", "skew-t"] | None = "skew-t",
        stats: bool = True,
        **kwargs: Unpack[SettingDict],
    ) -> Artist:
        """
        Create a histogram of the data.

        Parameters
        ----------
        bins : int | list[float], optional
            Specifies the bins to divide the data into. If int, should be the number
            of bins. By default 100.
        density : bool, optional
            Determines whether to draw a probability density. If True, the histogram
            will be normalized such that the area under it equals to 1. By default
            True.
        log : bool, optional
            Determines whether to set the histogram axis to a log scale, by default
            False.
        same_bin : bool, optional
            Determines whether the bins should be the same for all sets of data, by
            default True.
        fit : Literal["norm", "skew-norm", "t", "skew-t"] | None, optional
            Distribution used to fit a curve to the histogram, only available when
            `density=True`. Set to ``None`` to disable fitting. By default
            ``"skew-t"``.
        stats : bool, optional
            Determines whether to show the statistics, including the calculated mean,
            standard deviation, skewness, and kurtosis of the input, by default True.
        **kwargs : **SettingDict
            Specifies the plot settings, see `.set_plot()` for more details.

        Returns
        -------
        Artist
            An instance of Artist.

        """
        return self._get_artist(Histogram, locals())

    def plot(
        self,
        xticks: Optional["PlotDataSet | Any"] = None,
        fmt: str = "",
        scatter: bool = False,
        sorted: bool = False,
        rolling: Optional[int | list[int]] = None,
        **kwargs: Unpack[SettingDict],
    ) -> Artist:
        """
        Create a line chart for the data. If there are more than one datasets, all of
        them should have the same length.

        Parameters
        ----------
        xticks : PlotDataSet | Any, optional
            Specifies the x-ticks for the line chart. If not provided, the x-ticks will
            be set to `range(len(data))`. By default None.
        fmt : str, optional
            A format string, e.g. 'ro' for red circles, by default ''.
        scatter : bool, optional
            Determines whether to include scatter points in the line chart, by default
            False.
        sorted : bool, optional
            Determines whether to sort by x-ticks before drawing the chart, by
            default False.
        rolling : int | list[int], optional
            Rolling window size(s). If provided as an integer, a single rolling
            mean with `rolling(rolling, min_periods=1)` is applied to y-values
            after optional sorting. If provided as a list, one line is drawn for
            each rolling window, by default None.
        **kwargs : **SettingDict
            Specifies the plot settings, see `.set_plot()` for more details.

        Returns
        -------
        Artist
            An instance of Artist.

        """
        if isinstance(xticks, PlotSettable) and "xlabel" not in kwargs:
            if kwargs.get("format_label", True):
                kwargs["xlabel"] = xticks.formatted_label()
            else:
                kwargs["xlabel"] = xticks.label
        return self._get_artist(LineChart, locals())

    def scatter(
        self,
        xticks: Optional["PlotDataSet | Any"] = None,
        fmt: str = "o",
        **kwargs: Unpack[SettingDict],
    ) -> Artist:
        """
        Create a scatter chart for the data. If there are more than one datasets,
        all of them should have the same length.

        Parameters
        ----------
        xticks : PlotDataSet | Any, optional
            Specifies the x-ticks for the chart. If not provided, the x-ticks will
            be set to `range(len(data))`. By default None.
        fmt : str, optional
            A format string, e.g. 'ro' for red circles, by default 'o'.
        **kwargs : **SettingDict
            Specifies the plot settings, see `.set_plot()` for more details.

        Returns
        -------
        Artist
            An instance of Artist.

        """
        if isinstance(xticks, PlotSettable) and "xlabel" not in kwargs:
            if kwargs.get("format_label", True):
                kwargs["xlabel"] = xticks.formatted_label()
            else:
                kwargs["xlabel"] = xticks.label
        return self._get_artist(ScatterChart, locals())

    def qqplot(
        self,
        baseline: "DistName | PlotDataSet | Any" = "normal",
        dots: int = 30,
        edge_precision: float = 1e-2,
        fmt: str = "o",
        **kwargs: Unpack[SettingDict],
    ) -> Artist:
        """
        Create a quantile-quantile plot.

        Parameters
        ----------
        baseline : DistName | PlotDataSet | Any, optional
            Specifies the distribution to compare with. If str, specifies a
            theoretical distribution; if PlotDataSet or Any, specifies another
            sample. By default 'normal'.
        dots : int, optional
            Number of dots, by default 30.
        edge_precision : float, optional
            Specifies the lowest quantile (`=edge_precision`) and the highest
            quantile (`=1-edge_precision`), by default 1e-2.
        fmt : str, optional
            A format string, e.g. 'ro' for red circles, by default 'o'.
        **kwargs : **SettingDict
            Specifies the plot settings, see `.set_plot()` for more details.

        Returns
        -------
        Artist
            An instance of Artist.

        """
        return self._get_artist(QQPlot, locals())

    def ppplot(
        self,
        baseline: "DistName | PlotDataSet | Any" = "normal",
        dots: int = 30,
        fmt: str = "o",
        **kwargs: Unpack[SettingDict],
    ) -> Artist:
        """
        Create a probability-probability plot.

        Parameters
        ----------
        baseline : DistName | PlotDataSet | Any, optional
            Specifies the distribution to compare with. If str, specifies a
            theoretical distribution; if PlotDataSet or Any, specifies another
            sample. By default 'normal'.
        dots : int, optional
            Number of dots, by default 30.
        fmt : str, optional
            A format string, e.g. 'ro' for red circles, by default 'o'.
        **kwargs : **SettingDict
            Specifies the plot settings, see `.set_plot()` for more details.

        Returns
        -------
        Artist
            An instance of Artist.

        """
        edge_precision = 1e-6
        return self._get_artist(PPPlot, locals())

    def ksplot(
        self,
        baseline: "DistName | PlotDataSet | Any" = "normal",
        dots: int = 1000,
        fmt: str = "",
        **kwargs: Unpack[SettingDict],
    ) -> Artist:
        """
        Create a kolmogorov-smirnov plot.

        Parameters
        ----------
        baseline : DistName | PlotDataSet | Any, optional
            Specifies the distribution to compare with. If str, specifies a
            theoretical distribution; if np.ndarray or PlotDataSet, specifies
            another real sample. By default 'normal'.
        dots : int, optional
            Number of dots, by default 1000.
        fmt : str, optional
            A format string, e.g. 'ro' for red circles, by default ''.
        **kwargs : **SettingDict
            Specifies the plot settings, see `.set_plot()` for more details.

        Returns
        -------
        Artist
            An instance of Artist.

        """
        edge_precision = 1e-6
        return self._get_artist(KSPlot, locals())

    def corrmap(
        self,
        annot: bool = True,
        **kwargs: Unpack[SettingDict],
    ) -> Artist:
        """
        Create a correlation heatmap.

        Parameters
        ----------
        annot : bool, optional
            Specifies whether to write the data value in each cell, by default
            True.
        **kwargs : **SettingDict
            Specifies the plot settings, see `.set_plot()` for more details.

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
        if "format_label" in local["kwargs"] and not local["kwargs"]["format_label"]:
            label = self.label
        else:
            label = self.formatted_label()
        plotter = self.customize(cls, data=self.data, label=label, **params)
        artist = single(self.customize)(Artist, plotter=plotter)
        if local["kwargs"]:
            artist.plotter.load(local["kwargs"])
            artist.load(local["kwargs"])
        return artist


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

    def batched(self, n: int = 1) -> MultiObject:
        """Overrides `PlotDataSet.batched()`."""
        PlotDataSet.batched(self, n)
        m = MultiObject()
        for i in range(0, len(self.__multiobjects__), n):
            m.__multiobjects__.append(PlotDataSets(*self.__multiobjects__[i : i + n]))
        return m

    def __dataset_attr_reducer(self, n: str) -> Callable:
        match n:
            case (
                "hist"
                | "plot"
                | "scatter"
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
                        attr_reducer=lambda x: multipartial(call_reflex=x == "paint")
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
        if all(i is None for i in x):
            return None
        return REMAIN
