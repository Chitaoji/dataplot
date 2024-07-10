"""
The core of math: linear_regression_1d, etc.

"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["linear_regression_1d"]


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
