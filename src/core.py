"""
Contains the core of dataplot: figure(), data().

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""
from hintwith import hintwithmethod

from .dataset import PlotData
from .setter import FigWrapper

__all__ = ["figure", "data"]


@hintwithmethod(FigWrapper.__init__)
def figure(*args, **kwargs) -> FigWrapper:
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
    return FigWrapper(*args, **kwargs)


@hintwithmethod(PlotData.__new__)
def data(*args, **kwargs) -> PlotData:
    """Calls `PlotData()`."""
    return PlotData(*args, **kwargs)
