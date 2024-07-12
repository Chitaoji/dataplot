"""
Contains the core of dataplot: figure(), data().

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from math import ceil, sqrt
from typing import TYPE_CHECKING, Optional, Unpack, overload

import numpy as np

from .container import FigWrapper
from .dataset import PlotDataSet, PlotDataSets

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._typing import FigureSettingDict
    from .artist import Artist


__all__ = ["figure", "data", "show"]


def figure(
    nrows: int = 1, ncols: int = 1, **kwargs: Unpack["FigureSettingDict"]
) -> FigWrapper:
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
    **kwargs : **FigureSettingDict
        Figure settings, see `FigWrapper.set_figure()` for more details.

    Returns
    -------
    FigWrapper
        A wrapper of figure.

    """
    fig = FigWrapper(nrows=nrows, ncols=ncols)
    fig.set_figure(**kwargs)
    return fig


@overload
def data(x: "NDArray", label: Optional[str] = None) -> PlotDataSet: ...
@overload
def data(x: list["NDArray"], label: Optional[list[str]] = None) -> PlotDataSet: ...
def data(
    x: "NDArray | list[NDArray]", label: Optional[str | list[str]] = None
) -> PlotDataSet:
    """
    Initializes a dataset interface which provides methods for mathematical
    operations and plotting.

    Parameters
    ----------
    x : NDArray | list[NDArray]
        Input values, this takes either a single array or a list of arrays, with
        each array representing a dataset.
    label : str | list[str], optional
        Label(s) of the data, this takes either a single string or a list of strings.
        If a list, should be the same length as `x`, with each element corresponding
        to a specific array in `x`. If set to None, use "x{i}" (i = 1, 2. 3, ...) as
        the label(s). By default None.

    Returns
    -------
    PlotDataSet
        Provides methods for mathematical operations and plotting.

    """
    if isinstance(x, list) and any(isinstance(i, (np.ndarray, list)) for i in x):
        if label is None:
            label = [f"x{i}" for i in range(1, 1 + len(x))]
        datas = [PlotDataSet(np.array(d), lb) for d, lb in zip(x, label)]
        return PlotDataSets(*datas)
    if isinstance(label, list):
        raise ValueError(
            "it seems not necessary to provide a list of labels, since "
            "the data has only one dimension"
        )
    return PlotDataSet(np.array(x), label=label)


def show(
    artist: "Artist | list[Artist]",
    ncols: Optional[int] = None,
    **kwargs: Unpack["FigureSettingDict"],
) -> None:
    """
    Paint the artist(a) on a new figure.

    Parameters
    ----------
    artist : Artist | list[Artist]
        Artist or list of artists.
    ncols : Optional[int], optional
        Number of columns. If None, will be set to `floor(sqrt(n))`, where `n`
        is the number of artist(s). By default None.
    **kwargs : **FigureSettingDict
        Figure settings, see `FigWrapper.set_figure()` for more details.

    """
    if not isinstance(artist, list):
        artist = [artist]
    len_a = len(artist)
    ncols = int(sqrt(len_a)) if ncols is None else min(ncols, len_a)
    with figure(ceil(len_a // ncols), ncols, **kwargs) as fig:
        for a, ax in zip(artist, fig.axes[:len_a]):
            a.paint(ax)
