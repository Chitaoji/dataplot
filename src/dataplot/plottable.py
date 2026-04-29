"""
Contains the dataset interface: PlottableData.

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
from validating import dataclass

from ._typing import DistName, SettingDict
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
from .database import Data
from .setting import PlotSettable
from .utils.multi import (
    REMAIN,
    MultiObject,
    multipartial,
    single,
)

if TYPE_CHECKING:
    from .artist import Plotter


__all__ = ["PlottableData"]


@dataclass(validate_methods=True)
class PlottableData(Data, PlotSettable, metaclass=ABCMeta):
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

    @classmethod
    def __subclasshook__(cls, __subclass: type) -> bool:
        if __subclass is PlottableData or issubclass(__subclass, PlottableDataSet):
            return True
        return False

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

    def join(self, *others: "PlottableData") -> Self:
        """
        Merge two or more `PlottableData` instances.

        Parameters
        ----------
        *others : PlottableData
            The instances to be merged.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        return PlottableDataSet(self, *others)

    def copy(self) -> Self:
        return self._create_data(self.fmtb, self.data, priority=self.priority)

    def reset(self) -> Self:
        """
        Return a copy of self with plot settings reset.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        obj = self.copy()
        obj.settings.reset()
        return obj

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
        return self._create_data(
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
        If this instance is joined by multiple `PlottableData` objects, batch the
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
        xticks: Self | Any = None,
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
        xticks : PlottableData | Any, optional
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
        xticks: Self | Any = None,
        fmt: str = "o",
        **kwargs: Unpack[SettingDict],
    ) -> Artist:
        """
        Create a scatter chart for the data. If there are more than one datasets,
        all of them should have the same length.

        Parameters
        ----------
        xticks : PlottableData | Any, optional
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
        baseline: DistName | Self | Any = "normal",
        dots: int = 30,
        edge_precision: float = 1e-2,
        fmt: str = "o",
        **kwargs: Unpack[SettingDict],
    ) -> Artist:
        """
        Create a quantile-quantile plot.

        Parameters
        ----------
        baseline : DistName | PlottableData | Any, optional
            Specifies the distribution to compare with. If str, specifies a
            theoretical distribution; if PlottableData or Any, specifies another
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
        baseline: DistName | Self | Any = "normal",
        dots: int = 30,
        fmt: str = "o",
        **kwargs: Unpack[SettingDict],
    ) -> Artist:
        """
        Create a probability-probability plot.

        Parameters
        ----------
        baseline : DistName | PlottableData | Any, optional
            Specifies the distribution to compare with. If str, specifies a
            theoretical distribution; if PlottableData or Any, specifies another
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
        baseline: DistName | Self | Any = "normal",
        dots: int = 1000,
        fmt: str = "",
        **kwargs: Unpack[SettingDict],
    ) -> Artist:
        """
        Create a kolmogorov-smirnov plot.

        Parameters
        ----------
        baseline : DistName | PlottableData | Any, optional
            Specifies the distribution to compare with. If str, specifies a
            theoretical distribution; if np.ndarray or PlottableData, specifies
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

    def _create_data(
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


class PlottableDataSet(MultiObject[PlottableData]):
    """A duck subclass of `PlottableData`."""

    def __init__(self, *args: Any) -> None:
        if not args:
            raise ValueError("no args")
        objs: list[PlottableData] = []
        for a in args:
            if isinstance(a, self.__class__):
                objs.extend(a.__multiobjects__)
            elif isinstance(a, PlottableData):
                objs.append(a)
            else:
                raise TypeError(f"invalid type: {a.__class__.__name__!r}")
        super().__init__(objs, attr_reducer=self.__dataset_attr_reducer)

    def __repr__(self) -> str:
        data_info = "\n- ".join([x.data_info() for x in self.__multiobjects__])
        return f"{PlottableData.__name__}\n- {data_info}"

    def batched(self, n: int = 1) -> MultiObject:
        """Overrides `PlottableData.batched()`."""
        PlottableData.batched(self, n)
        m = MultiObject()
        for i in range(0, len(self.__multiobjects__), n):
            m.__multiobjects__.append(
                PlottableDataSet(*self.__multiobjects__[i : i + n])
            )
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
                return lambda _: partial(getattr(PlottableData, n), self)
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
        if x and isinstance(x[0], PlottableData):
            return cls(*x)
        if all(i is None for i in x):
            return None
        return REMAIN
