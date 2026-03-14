"""
Contains the core of dataplot: figure(), data(), show(), etc.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""
from math import ceil, sqrt
import sys
from typing import TYPE_CHECKING, Any, Optional, Unpack, overload

import numpy as np

from .container import FigWrapper
from .dataset import PlotDataSet, PlotDataSets
from ._typing import FigureSettingDict

if TYPE_CHECKING:
    from .artist import Artist


__all__ = ["figure", "data", "show"]


def _infer_var_names(*values: Any) -> list[Optional[str]]:
    try:
        search_frame = sys._getframe(1)
    except ValueError:
        return [None] * len(values)

    labels: list[Optional[str]] = []
    try:
        for value in values:
            name = None
            current = search_frame
            while current is not None and name is None:
                local_items = list(current.f_locals.items())
                global_items = list(current.f_globals.items())
                name = next((k for k, v in local_items if v is value), None)
                if name is None:
                    name = next((k for k, v in global_items if v is value), None)
                current = current.f_back
            labels.append(name)
    finally:
        del search_frame
    return labels


def figure(
    nrows: int = 1, ncols: int = 1, **kwargs: Unpack[FigureSettingDict]
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
        Specifies the figure settings, see `FigWrapper.set_figure()` for more details.

    Returns
    -------
    FigWrapper
        A wrapper of figure.

    """
    fig = FigWrapper(nrows=nrows, ncols=ncols)
    fig.set_figure(**kwargs)
    return fig


def data(*x: Any, label: Optional[str | list[str]] = None) -> PlotDataSet:
    """
    Initializes a dataset interface which provides methods for mathematical
    operations and plotting.

    Parameters
    ----------
    *x : np.ndarray | Any
        Input values, this takes one or multiple arrays, with each array
        representing a dataset.
    label : str | list[str], optional
        Label(s) of the data, this takes either a single string or a list of strings.
        If a list, should be the same length as the number of input arrays, with
        each element corresponding to a specific array in `x`. If set to None,
        use "x{i}" (i = 1, 2. 3, ...) as the label(s). By default None.

    Returns
    -------
    PlotDataSet
        Provides methods for mathematical operations and plotting.

    """
    if not x:
        raise ValueError("at least one dataset should be provided")

    if len(x) > 1:
        if label is None:
            label = [
                lb if lb is not None else f"x{i}"
                for i, lb in enumerate(_infer_var_names(*x), start=1)
            ]
        elif isinstance(label, str):
            raise ValueError(
                "for multiple datasets, please provide labels as a list of strings"
            )
        elif len(label) != len(x):
            raise ValueError(
                f"label should have the same length as x ({len(x)}), got {len(label)}"
            )
        datas = [PlotDataSet(np.array(d), lb) for d, lb in zip(x, label)]
        return PlotDataSets(*datas)

    if isinstance(label, list):
        raise ValueError(
            "it seems not necessary to provide a list of labels, since "
            "the data has only one dimension"
        )
    if label is None:
        label = _infer_var_names(x[0])[0]
    return PlotDataSet(np.array(x[0]), label=label)


def show(
    artist: "Artist | list[Artist]",
    ncols: Optional[int] = None,
    **kwargs: Unpack[FigureSettingDict],
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
        Specifies the figure settings, see `FigWrapper.set_figure()` for more details.

    """
    if not isinstance(artist, list):
        artist = [artist]
    len_a = len(artist)
    ncols = int(sqrt(len_a)) if ncols is None else min(ncols, len_a)
    with figure(ceil(len_a / ncols), ncols, **kwargs) as fig:
        for a, ax in zip(artist, fig.axes[:len_a]):
            a.paint(ax)
        for ax in fig.axes[len_a:]:
            fig.fig.delaxes(ax.ax)
