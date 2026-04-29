"""
Contains a plotter class: Histogram.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

import warnings
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
from scipy import stats
from validating import dataclass

from .base import Plotter

if TYPE_CHECKING:
    from ..container import AxesWrapper

__all__ = ["Histogram"]


@dataclass(validate_methods=True)
class Histogram(Plotter):
    """
    A plotter class that creates a histogram.

    """

    bins: int | list[float]
    fit: Literal["norm", "skew-norm", "t", "skew-t"] | None
    density: bool
    log: bool
    same_bin: bool
    stats: bool

    def paint(
        self,
        ax: "AxesWrapper",
        *,
        __multi_prev_returned__: Optional[tuple[str, np.ndarray]] = None,
        __multi_is_final__: bool = True,
    ) -> tuple[str, np.ndarray]:
        ax.set_axes(
            title=ax.get_setting("title", "Histogram"),
            alpha=ax.get_setting("alpha", 0.8),
            xlabel=ax.get_setting("xlabel", "value"),
            ylabel=ax.get_setting("ylabel", "density" if self.density else "count"),
        )
        ax.load(self.settings)
        if __multi_prev_returned__ is None:
            xlabel, bins = ax.settings.xlabel, None
        else:
            xlabel, bins = __multi_prev_returned__
        ds, new_bins = self.__hist(
            ax, bins=self.bins if (bins is None or not self.same_bin) else bins
        )
        if self.stats:
            xlabel += "\n" + ds
        if __multi_is_final__:
            ax.set_axes(xlabel=xlabel)
        return (xlabel, new_bins)

    def __hist(
        self, ax: "AxesWrapper", bins: int | list[float] = 100
    ) -> tuple[str, np.ndarray]:
        _, bin_list, _ = ax.ax.hist(
            self.data,
            bins=bins,
            density=self.density,
            log=self.log,
            alpha=ax.settings.alpha,
            label=self.name,
        )
        mean, std = np.nanmean(self.data), np.nanstd(self.data)
        skew: float = stats.skew(self.data, bias=False, nan_policy="omit")
        kurt: float = stats.kurtosis(self.data, bias=False, nan_policy="omit")
        if self.fit is not None and self.density:
            fit_curve = self.__fit_pdf(bin_list, self.data, dist=self.fit)
            ax.ax.plot(
                bin_list,
                fit_curve,
                alpha=ax.settings.alpha,
                label=f"{self.name} · fit",
            )

        # Disable matplotlib's default horizontal margins for tighter x-limits.
        ax.ax.margins(x=0)
        return (
            f"{self.name}: mean={mean:.3f}, std={std:.3f}, skew={skew:.3f}, "
            f"kurt={kurt:.3f}",
            bin_list,
        )

    @staticmethod
    def __fit_pdf(
        x: np.ndarray,
        data: np.ndarray,
        dist: Literal["norm", "skew-norm", "t", "skew-t"],
    ) -> np.ndarray:
        sample = np.asarray(data, dtype=float)
        sample = sample[np.isfinite(sample)]
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                if dist == "norm":
                    loc, scale = stats.norm.fit(sample)
                    return stats.norm.pdf(x, loc=loc, scale=scale)
                elif dist == "skew-norm":
                    a, loc, scale = stats.skewnorm.fit(sample)
                    return stats.skewnorm.pdf(x, a, loc=loc, scale=scale)
                elif dist == "t":
                    df, loc, scale = stats.t.fit(sample)
                    return stats.t.pdf(x, df, loc=loc, scale=scale)
                else:
                    # Jones-Faddy skew-t: captures skewness and heavy tails.
                    # a, b affect skewness and kurtosis; loc/scale shift/scale.
                    a, b, loc, scale = stats.jf_skew_t.fit(sample)
                    return stats.jf_skew_t.pdf(x, a, b, loc=loc, scale=scale)
        except Exception:
            return np.zeros_like(x, dtype=float)
        return np.zeros_like(x, dtype=float)
