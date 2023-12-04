"""
Contains the core of dataplot: figure(), data().

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""
from hintwith import hintwith

from .dataset import PlotData
from .setter import FigWrapper

__all__ = ["figure", "data"]


@hintwith(FigWrapper)
def figure(nrows: int = 1, ncols: int = 1) -> FigWrapper:
    """Calls `FigWrapper()`."""
    return FigWrapper(nrows=nrows, ncols=ncols)


@hintwith(PlotData)
def data(*args, **kwargs) -> PlotData:
    """Calls `PlotData()`."""
    return PlotData(*args, **kwargs)
