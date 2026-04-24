"""
The core of math: linear_regression_1d(), get_quantile(), get_prob(), etc.

"""

import numpy as np

__all__ = ["linear_regression_1d", "get_quantile", "get_prob"]


def linear_regression_1d(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """
    Implements a 1-demensional linear regression of y on x (y = ax + b), and
    returns the regression coefficients (a, b). Nan-values and inf-values are
    handled smartly.

    Parameters
    ----------
    y : np.ndarray
        The dependent variable.
    x : np.ndarray
        The independent variable.

    Returns
    -------
    tuple[float, float]
        The regression coefficients (a, b).

    """
    x, y = (
        np.nan_to_num(x, nan=np.nan, posinf=np.nan, neginf=np.nan),
        np.nan_to_num(y, nan=np.nan, posinf=np.nan, neginf=np.nan),
    )
    nanmask = np.isfinite(x) & np.isfinite(y)
    if nanmask.sum() < 2:
        raise ValueError("too few finite-values for x and y")
    x_mask, y_mask = x[nanmask], y[nanmask]

    xy_mean = (x_mask * y_mask).mean()
    x_mean = x_mask.mean()
    y_mean = y_mask.mean()
    b = (xy_mean - x_mean * y_mean) / x_mask.var()
    a = y_mean - x_mean * b
    return a, b


def get_quantile(data: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Get quantiles from cummulative probabilities. Nan-values and inf-values
    are handled smartly.

    Parameters
    ----------
    data : np.ndarray
        Original data.
    p : np.ndarray
        Cummulative probabilities.

    Returns
    -------
    np.ndarray
        Quantiles.

    """
    return np.nanquantile(
        np.nan_to_num(data, nan=np.nan, posinf=np.nan, neginf=np.nan), p
    )


def get_prob(data: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Get cummulative probabilities from quantiles.

    Parameters
    ----------
    data : np.ndarray
        Original data.
    q : np.ndarray
        Quantiles.

    Returns
    -------
    np.ndarray
        Cummulative probabilities.

    """
    data = np.nan_to_num(data, nan=np.nan, posinf=np.nan, neginf=np.nan)
    return np.array([np.sum(data <= x) for x in q]) / np.isfinite(data).sum()
