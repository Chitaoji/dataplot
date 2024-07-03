"""
Contains the core of dataplot: figure(), data().

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import TYPE_CHECKING, Optional, Union, overload

from .container import FigWrapper
from .dataset import PlotDataSet, PlotDataSets

if TYPE_CHECKING:
    from numpy.typing import NDArray


__all__ = ["figure", "data"]


def figure(nrows: int = 1, ncols: int = 1) -> FigWrapper:
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

    Returns
    -------
    FigWrapper
        A wrapper of figure.

    """
    return FigWrapper(nrows=nrows, ncols=ncols)


@overload
def data(x: "NDArray", label: Optional[str] = None) -> PlotDataSet: ...


@overload
def data(x: list["NDArray"], label: Optional[list[str]] = None) -> PlotDataSet: ...


def data(
    x: Union["NDArray", list["NDArray"]], label: Union[str, list[str], None] = None
) -> PlotDataSet:
    """
    Initializes a dataset interface which provides methods for mathematical
    operations and plotting.

    Parameters
    ----------
    x : Union[NDArray, list[NDArray]]
        Input values, this takes either a single array or a list of arrays, with
        each array representing a dataset.
    label : Union[str, list[str], None], optional
        Label(s) of the data, this takes either a single string or a list of strings.
        If a list, should be the same length as `x`, with each element corresponding
        to a specific array in `x`. If set to None, use "x{i}" (i = 1, 2. 3, ...) as
        the label(s). By default None.

    Returns
    -------
    PlotDataSet
        Provides methods for mathematical operations and plotting.

    """
    if isinstance(x, list):
        if label is None:
            label = [f"x{i}" for i in range(1, 1 + len(x))]
        datas = [PlotDataSet(d, lb) for d, lb in zip(x, label)]
        return PlotDataSets(*datas)
    return PlotDataSet(x, label=label)
