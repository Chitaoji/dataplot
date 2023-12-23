"""
Contains the dataset interface: PlotData.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""
from abc import ABCMeta
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, TypeVar

import numpy as np
import pandas as pd
from attrs import define, field
from typing_extensions import Self

from .histogram import Histogram
from .linechart import LineChart
from .setter import AxesWrapper, FigWrapper, PlotSetter, PlotSettings
from .utils.multis import MultiObject, cleaner, single

if TYPE_CHECKING:
    from numpy.typing import NDArray


T = TypeVar("T")

__all__ = ["PlotData"]


@define
class PlotData(PlotSetter, metaclass=ABCMeta):
    """
    Provides methods for mathematical operations and plotting.

    Note that this should NEVER be instantiated directly, but always through the
    module-level function `dataplot.data()`.

    """

    data: "NDArray" = field(repr=False)
    label: Optional[str] = field(default=None)
    fmt: str = field(init=False, repr=lambda x: repr(x).format(""), default="{0}")
    fmtdata: "NDArray" = field(init=False, repr=False)
    settings: PlotSettings = field(
        init=False, factory=PlotSettings, repr=PlotSettings.repr_not_none
    )

    def __attrs_post_init__(self) -> None:
        self.label = "x1" if self.label is None else self.label
        self.fmtdata = self.data

    @classmethod
    def __subclasshook__(cls, __subclass: type) -> bool:
        if issubclass(__subclass, _PlotDatas):
            return True
        return super().__subclasshook__(__subclass)

    def __create(self, fmt: str, fmtdata: "NDArray") -> "PlotData":
        obj = self.customize(self.__class__, self.data, self.label)
        obj.fmt = fmt
        obj.fmtdata = fmtdata
        return obj

    def set_label(self, mapper: Optional[Mapping[str, str]] = None) -> Self:
        """
        Reset the labels according to the mapper.

        Parameters
        ----------
        mapper : Optional[Mapping], optional
            Mapper to apply to the labels, by default None.

        Returns
        -------
        Self
            An instance of self.

        """
        if self.label in mapper:
            self.label = mapper[self.label]
        return self

    def stage(self) -> Self:
        """
        Stage all the operations on the data while cleaning the records.

        Returns
        -------
        Self
            An instance of self.

        """
        self.fmt = "{0}"
        return self

    @property
    def fmtlabel(self) -> str:
        """
        Returns the formatted label.

        Returns
        -------
        str
            Formatted label.

        """
        return self.fmt.format(self.label)

    def join(self, *others: "PlotData") -> "PlotData":
        """
        Merge two or more `PlotData` instances, integrating the attributes
        of each.

        Parameters
        ----------
        *others : PlotData
            The instances to be merged.

        Returns
        -------
        PlotData
            A new instance of `PlotData`.

        """
        return _PlotDatas(self, *others)

    def log(self) -> "PlotData":
        """
        Perform a log operation on the data.

        Returns
        -------
        PlotData
            A new instance of `PlotData`.

        """
        new_fmt = f"log({self.fmt})"
        new_fmtdata = np.log(self.fmtdata)
        return self.__create(new_fmt, new_fmtdata)

    def rolling(self, n: int) -> "PlotData":
        """
        Perform a rolling-mean operation on the data.

        Parameters
        ----------
        n : int
            Specifies the window size for calculating the rolling average of
            the data points.

        Returns
        -------
        PlotData
            A new instance of `PlotData`.

        """
        new_fmt = f"rolling({self.fmt}, {n})"
        new_fmtdata = pd.Series(self.fmtdata).rolling(n).mean().values
        return self.__create(new_fmt, new_fmtdata)

    def exp(self) -> "PlotData":
        """
        Perform an exp operation on the data.

        Returns
        -------
        PlotData
            A new instance of `PlotData`.

        """
        new_fmt = f"exp({self.fmt})"
        new_fmtdata = np.exp(self.fmtdata)
        return self.__create(new_fmt, new_fmtdata)

    def demean(self) -> "PlotData":
        """
        Perform a demean operation on the data by subtracting its mean.

        Returns
        -------
        PlotData
            A new instance of `PlotData`.

        """
        new_fmt = f"{self.fmt} - mean({self.fmt})"
        new_fmtdata = self.fmtdata - np.nanmean(self.fmtdata)
        return self.__create(new_fmt, new_fmtdata)

    def zscore(self) -> "PlotData":
        """
        Perform a zscore operation on the data by subtracting its mean and then
        dividing by its standard deviation.

        Returns
        -------
        PlotData
            A new instance of `PlotData`.

        """
        new_fmt = f"({self.fmt} - mean({self.fmt})) / std({self.fmt})"
        new_fmtdata = (self.fmtdata - np.nanmean(self.fmtdata)) / np.nanstd(
            self.fmtdata
        )
        return self.__create(new_fmt, new_fmtdata)

    def cumsum(self) -> "PlotData":
        """
        Perform a cumsum operation on the data by calculating its cummulative
        sums.

        Returns
        -------
        PlotData
            A new instance of `PlotData`.

        """
        new_fmt = f"cumsum({self.fmt})"
        new_fmtdata = np.cumsum(self.fmtdata)
        return self.__create(new_fmt, new_fmtdata)

    def reset(self) -> Self:
        """
        Reset all the operations.

        Returns
        -------
        Self
            An instance of self.
        """
        self.fmt = "{0}"
        self.fmtdata = self.data
        return self

    # pylint: disable=unused-argument
    def hist(
        self,
        bins: int = 100,
        fit: bool = True,
        density: bool = True,
        same_bin: bool = True,
        stats: bool = True,
        *,
        on: Optional[AxesWrapper] = None,
    ) -> None:
        """
        Plot a histogram of the data.

        Parameters
        ----------
        bins : int, optional
            Specifies the number of bins to divide the data into for the histogram
            plot, by default 100.
        fit : bool, optional
            Fit a curve to the histogram or not, by default True.
        density : bool, optional
            Draw a probability density or not. If True, the histogram will be
            normalized such that the area under it equals to 1. By default True.
        same_bin : bool, optional
            Determines whether the bins should be the same for all sets of data, by
            default True.
        stats : bool, optional
            Determines whether to show the statistics, including the calculated mean,
            standard deviation, skewness, and kurtosis of the input, by default True.
        on : Optional[AxesWrapper], optional
            Specifies the axes wrapper on which the histogram should be plotted. If
            not specified, the histogram will be plotted on a new axes in a new
            figure. By default None.

        """
        with single(self.customize)(FigWrapper, 1, 1) as fig:
            on = fig.axes[0]
            kwargs: Dict[str] = {}
            for key in Histogram.__init__.__code__.co_varnames[1:]:
                kwargs[key] = locals()[key]
            self.customize(
                Histogram, data=self.fmtdata, label=self.fmtlabel, **kwargs
            ).perform()

    def plot(
        self,
        ticks: Optional["NDArray"] = None,
        scatter: bool = False,
        figsize_adjust: bool = True,
        *,
        on: Optional[AxesWrapper] = None,
    ) -> None:
        """
        Create a line chart for the data.

        Parameters
        ----------
        ticks : Optional[NDArray], optional
            Specifies the x-ticks for the line chart. If not provided, the x-ticks will
            be set to `range(len(data))`. By default None.
        scatter : bool, optional
            Determines whether to include scatter points in the line chart, by default
            False.
        figsize_adjust : bool, optional
            Determines whether the size of the figure should be adjusted automatically
            based on the data being plotted, by default True.
        on : Optional[AxesWrapper], optional
            Specifies the axes wrapper on which the line chart should be plotted. If
            not specified, the histogram will be plotted on a new axes in a new
            figure. By default None.

        """
        with single(self.customize)(FigWrapper, 1, 1) as fig:
            on = fig.axes[0]
            kwargs: Dict[str] = {}
            for key in LineChart.__init__.__code__.co_varnames[1:]:
                kwargs[key] = locals()[key]
            self.customize(
                LineChart, data=self.fmtdata, label=self.fmtlabel, **kwargs
            ).perform()

    def batched(self, n: int = 1) -> Self:
        """
        If this instance is joined by multiple PlotData objects, batch the objects into
        tuples of length n, otherwise return self.

        Parameters
        ----------
        n : int, optional
            Specifies the batch size, by default 1.

        Returns
        -------
        PlotData
            An instance of PlotData.

        """
        return self

    # pylint: enable=unused-argument


class _PlotDatas:
    def __init__(self, *args: Any) -> None:
        if not args:
            raise ValueError("number of data sets is 0")
        self.children: List[PlotData] = []
        for a in args:
            if isinstance(a, self.__class__):
                self.children.extend(a.children)
            else:
                self.children.append(a)

    def __getattr__(self, __name: str) -> Any:
        if __name in {"hist", "plot"}:
            return partial(getattr(PlotData, __name), self)
        attribs = (getattr(c, __name) for c in self.children)
        if __name in {"set_plot", "set_plot_default"}:
            return MultiObject(attribs, call_reducer=lambda x: self)
        if __name == "customize":
            return MultiObject(attribs, call_reflex="reflex")
        return MultiObject(attribs, call_reducer=lambda x: self.__class__(*x))

    def __repr__(self) -> str:
        return "[" + ",\n ".join([repr(x) for x in self.children]) + "]"

    def __getitem__(self, __key: str) -> PlotData:
        return self.children[__key]

    def batched(self, n: int = 1) -> "MultiObject":
        """Overrides `PlotData.batched()`."""
        if n <= 0:
            raise ValueError(f"batch size <= 0: {n}")
        if n > len(self.children):
            return self
        multi = MultiObject(call_reducer=cleaner)
        for i in range(0, len(self.children), n):
            multi.__multiobjects__.append(_PlotDatas(*self.children[i : i + n]))
        return multi
