"""
Contains a plotter class: Histogram.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import TYPE_CHECKING, Optional

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
    fit: bool
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
            label=self.label,
        )
        mean, std = np.nanmean(self.data), np.nanstd(self.data)
        skew: float = stats.skew(self.data, bias=False, nan_policy="omit")
        kurt: float = stats.kurtosis(self.data, bias=False, nan_policy="omit")
        if self.fit and self.density:
            fit_curve = self.__skew_t_pdf(bin_list, self.data)
            ax.ax.plot(
                bin_list,
                fit_curve,
                alpha=ax.settings.alpha,
                label=f"{self.label} · fit",
            )

        # Disable matplotlib's default horizontal margins for tighter x-limits.
        ax.ax.margins(x=0)
        return (
            f"{self.label}: mean={mean:.3f}, std={std:.3f}, skew={skew:.3f}, "
            f"kurt={kurt:.3f}",
            bin_list,
        )

    @staticmethod
    def __skew_t_pdf(x: np.ndarray, data: np.ndarray) -> np.ndarray:
        sample = np.asarray(data, dtype=float)
        sample = sample[np.isfinite(sample)]
        if len(sample) < 5:
            return np.zeros_like(x, dtype=float)

        # Jones-Faddy skew-t distribution: captures skewness and heavy tails.
        # a, b affect skewness and kurtosis; loc/scale shift and scale the fit.
        try:
            a, b, loc, scale = stats.jf_skew_t.fit(sample)
        except Exception:
            return np.zeros_like(x, dtype=float)
        if (
            (not np.isfinite(a))
            or (not np.isfinite(b))
            or (not np.isfinite(scale))
            or a <= 0
            or b <= 0
            or scale <= 0
        ):
            return np.zeros_like(x, dtype=float)
        return stats.jf_skew_t.pdf(x, a, b, loc=loc, scale=scale)
