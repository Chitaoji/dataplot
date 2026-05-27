"""
Contains the dataset interface: PlottableData.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

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

from ._typing import DistName, DistNameForHist, MarkerStyle, SampleRule, SettingDict
from .artist import (
    Artist,
    CorrMap,
    HexBinMap,
    Histogram,
    KSPlot,
    LineChart,
    PPPlot,
    QQPlot,
    ScatterPlot,
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
class PlottableData(Data, PlotSettable):
    """
    A dataset class providing methods for mathematical operations and plotting.

    Note that this should NEVER be instantiated directly, but always through the
    module-level function `dataplot.data()`.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    name : str, optional
        Name of the data. If set to None, use "x" as the default. By default None.

    """

    @classmethod
    def __subclasshook__(cls, __subclass: type) -> bool:
        if __subclass is PlottableData or issubclass(__subclass, PlottableDataSet):
            return True
        return False

    def __repr__(self) -> str:
        return self.__class__.__name__ + "\n- " + self.info()

    def info(self) -> str:
        """Information of data."""
        not_none = self.settings._repr_changes()
        return f"{self.formatted_name()}{': ' if not_none else ''}{not_none}"

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

    def set_names(
        self, name: Optional[str] = None, reset_format: bool = True, /, **kwargs: str
    ) -> Self:
        """
        Set the data names.

        Parameters
        ----------
        name : str, optional
            The new name (if specified), by default None.
        reset_format : bool, optional
            Determines whether to reset the format of the name (which shows
            the operations done on the data), by default True.
        **kwargs : str
            Works as a mapper to find the new name. If `self.name` is in
            `kwargs`, the name will be set to `kwargs[self.name]`.

        Returns
        -------
        Self
            A new instance of self.__class__.

        """
        if isinstance(name, str):
            new_name = name
        elif self.name in kwargs:
            new_name = kwargs[self.name]
        else:
            new_name = self.name
        return self._create_data(
            "{0}" if reset_format else self.fmtb,
            self.data,
            priority=self.priority,
            name=new_name,
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
        bins: int | list[int | float] = 100,
        density: bool = True,
        log: bool = False,
        same_bin: bool = True,
        stats: bool = True,
        fit: DistNameForHist | None = "norm",
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
        stats : bool, optional
            Determines whether to show the statistics, including the calculated mean,
            standard deviation, skewness, and kurtosis of the input, by default True.
        fit : DistNameForHist | None, optional
            Distribution used to fit a curve to the histogram, only available when
            `density=True`. Set to ``None`` to disable fitting. By default
            ``"norm"``.
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
        linestyle: Literal["-", "--", "-.", ":"] | list[str] = "-",
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
        linestyle : Literal["-", "--", "-.", ":"], optional
            Line style passed to matplotlib, by default "-".
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
        if isinstance(xticks, Data) and "xlabel" not in kwargs:
            kwargs["xlabel"] = xticks.formatted_name()
        fmt = self.__normalize_fmt(linestyle, len(self.__multiobjects__))
        return self._get_artist(LineChart, locals())

    def scatter(
        self,
        xticks: Self | Any = None,
        marker: MarkerStyle | list[str] = ".",
        fit: bool = False,
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
        marker : MarkerStyle, optional
            Marker style (matplotlib format string), e.g. '.' for point markers,
            by default '.'.
        fit : bool, optional
            Determines whether to fit and draw a straight trend line. Only numeric
            x-ticks are supported when fitting. By default False.
        **kwargs : **SettingDict
            Specifies the plot settings, see `.set_plot()` for more details.

        Returns
        -------
        Artist
            An instance of Artist.

        """
        if isinstance(xticks, Data) and "xlabel" not in kwargs:
            kwargs["xlabel"] = xticks.formatted_name()
        fmt = self.__normalize_fmt(marker, len(self.__multiobjects__))
        return self._get_artist(ScatterPlot, locals())

    def hexbin(
        self,
        xticks: Self | Any = None,
        gridsize: int = 30,
        cmap: str = "viridis",
        mincnt: int | None = 1,
        **kwargs: Unpack[SettingDict],
    ) -> Artist:
        """
        Create a hexbin chart for the data.

        Parameters
        ----------
        xticks : PlottableData | Any, optional
            Specifies the x-values for the chart. If not provided, x-values will
            be set to `range(len(data))`. By default None.
        gridsize : int, optional
            Number of hexagons in the x-direction, by default 30.
        cmap : str, optional
            Colormap used to color hexagons by counts, by default "viridis".
        mincnt : int | None, optional
            Minimum count required to display a hexagon. Set to None to disable
            filtering. By default 1.
        **kwargs : **SettingDict
            Specifies the plot settings, see `.set_plot()` for more details.

        Returns
        -------
        Artist
            An instance of Artist.

        """
        if isinstance(xticks, Data) and "xlabel" not in kwargs:
            kwargs["xlabel"] = xticks.formatted_name()
        return self._get_artist(HexBinMap, locals())

    def qqplot(
        self,
        baseline: DistName | Self | Any = "norm",
        dots: int = 30,
        edge_precision: float = 1e-2,
        marker: MarkerStyle = "o",
        **kwargs: Unpack[SettingDict],
    ) -> Artist:
        """
        Create a quantile-quantile plot.

        Parameters
        ----------
        baseline : DistName | PlottableData | Any, optional
            Specifies the distribution to compare with. If str, specifies a
            theoretical distribution; if PlottableData or Any, specifies another
            sample. By default 'norm'.
        dots : int, optional
            Number of dots, by default 30.
        edge_precision : float, optional
            Specifies the lowest quantile (`=edge_precision`) and the highest
            quantile (`=1-edge_precision`), by default 1e-2.
        marker : str, optional
            Marker style (matplotlib format string), e.g. '.' for point markers,
            by default 'o'.
        **kwargs : **SettingDict
            Specifies the plot settings, see `.set_plot()` for more details.

        Returns
        -------
        Artist
            An instance of Artist.

        """
        fmt = marker
        return self._get_artist(QQPlot, locals())

    def ppplot(
        self,
        baseline: DistName | Self | Any = "norm",
        dots: int = 30,
        marker: MarkerStyle = "o",
        **kwargs: Unpack[SettingDict],
    ) -> Artist:
        """
        Create a probability-probability plot.

        Parameters
        ----------
        baseline : DistName | PlottableData | Any, optional
            Specifies the distribution to compare with. If str, specifies a
            theoretical distribution; if PlottableData or Any, specifies another
            sample. By default 'norm'.
        dots : int, optional
            Number of dots, by default 30.
        marker : str, optional
            Marker style (matplotlib format string), e.g. '.' for point markers,
            by default 'o'.
        **kwargs : **SettingDict
            Specifies the plot settings, see `.set_plot()` for more details.

        Returns
        -------
        Artist
            An instance of Artist.

        """
        fmt = marker
        edge_precision = 1e-6
        return self._get_artist(PPPlot, locals())

    def ksplot(
        self,
        baseline: DistName | Self | Any = "norm",
        dots: int = 1000,
        linestyle: Literal["-", "--", "-.", ":"] = "-",
        **kwargs: Unpack[SettingDict],
    ) -> Artist:
        """
        Create a kolmogorov-smirnov plot.

        Parameters
        ----------
        baseline : DistName | PlottableData | Any, optional
            Specifies the distribution to compare with. If str, specifies a
            theoretical distribution; if np.ndarray or PlottableData, specifies
            another real sample. By default 'norm'.
        dots : int, optional
            Number of dots, by default 1000.
        linestyle : Literal["-", "--", "-.", ":"], optional
            Line style passed to matplotlib, by default "-".
        **kwargs : **SettingDict
            Specifies the plot settings, see `.set_plot()` for more details.

        Returns
        -------
        Artist
            An instance of Artist.

        """
        edge_precision = 1e-6
        fmt = linestyle
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
        plotter = self.customize(
            cls, data=self.data, name=self.formatted_name(), **params
        )
        artist = single(self.customize)(Artist, plotter=plotter)
        if local["kwargs"]:
            artist.plotter.load(local["kwargs"])
            artist.load(local["kwargs"])
        return artist

    def _create_data(
        self, fmt: str, data: np.ndarray, priority: int = 0, name: Optional[str] = None
    ) -> Self:
        obj = self.customize(
            self.__class__,
            self.original_data,
            self.name if name is None else name,
            fmtb=fmt,
            priority=priority,
        )
        obj.data = data
        return obj

    @staticmethod
    def __normalize_fmt(fmt: str | list[str], length: int) -> str | MultiObject[str]:
        if isinstance(fmt, list):
            if len(fmt) != length:
                raise ValueError(
                    "fmt list length must match number of datasets, but got "
                    f"{len(fmt)} and {length}"
                )
            return MultiObject(fmt)
        return fmt


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
        data_info = "\n- ".join([x.info() for x in self.__multiobjects__])
        return f"{PlottableData.__name__}\n- {data_info}"

    def sample(self, n: int = 100, rule: SampleRule = "head") -> "PlottableDataSet":
        """Sample every dataset with the same random positions when requested."""
        if rule != "random":
            return PlottableDataSet(
                *(obj.sample(n=n, rule=rule) for obj in self.__multiobjects__)
            )
        length = min(len(obj.data) for obj in self.__multiobjects__)
        index = np.random.randint(0, length, n).tolist()
        return self.idx(index)

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
                | "hexbin"
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
