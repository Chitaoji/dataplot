"""
The core of math: linear_regression_1d(), get_quantile(), get_prob(), etc.

"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["linear_regression_1d", "get_quantile", "get_prob"]


def linear_regression_1d(y: "NDArray", x: "NDArray") -> tuple[float, float]:
    """
    Implements a 1-demensional linear regression of y on x (y = ax + b), and
    returns the regression coefficients (a, b). Nan-values and inf-values are
    handled smartly.

    Parameters
    ----------
    y : NDArray
        The dependent variable.
    x : NDArray
        The independent variable.

    Returns
    -------
    tuple[float, float]
        The regression coefficients (a, b).

    """
    x, y = np.nan_to_num(x, posinf=np.nan, neginf=np.nan), np.nan_to_num(
        y, posinf=np.nan, neginf=np.nan
    )
    xy_mean = np.nanmean(x * y)
    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)
    b = (xy_mean - x_mean * y_mean) / np.nanvar(x)
    a = y_mean - x_mean * b
    return a, b


def get_quantile(data: "NDArray", p: "NDArray") -> "NDArray":
    """
    Get quantiles from cummulative probabilities. Nan-values and inf-values
    are handled smartly.

    Parameters
    ----------
    data : NDArray
        Original data.
    p : NDArray
        Cummulative probabilities.

    Returns
    -------
    NDArray
        Quantiles.

    """
    return np.nanquantile(np.nan_to_num(data, posinf=np.nan, neginf=np.nan), p)


def get_prob(data: "NDArray", q: "NDArray") -> "NDArray":
    """
    Get cummulative probabilities from quantiles.

    Parameters
    ----------
    data : NDArray
        Original data.
    q : NDArray
        Quantiles.

    Returns
    -------
    NDArray
        Cummulative probabilities.

    """
    return np.array([np.sum(data <= x) for x in q]) / np.isfinite(data).sum()
