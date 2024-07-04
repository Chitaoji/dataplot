"""
Contains an artist class: Histogram.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import List, Optional, Tuple, Union

import numpy as np
from attrs import define
from scipy import stats

from .artist import Artist
from .container import AxesWrapper

__all__ = ["Histogram"]


@define
class Histogram(Artist):
    """
    An artist class that creates a histogram.

    """

    bins: int = 100
    fit: bool = True
    density: bool = True
    same_bin: bool = True
    stats: bool = True

    def paint(self, reflex: Optional[List[float]] = None) -> List[float]:
        """Paint on the axes.

        Parameters
        ----------
        reflex : Optional[List[float]], optional
            If list of float, specifies the bins to divide the data into for
            the histogram plot, by default None.

        Returns
        -------
        List[float]
            The bins of the histogram plot.

        """
        ax = self.prepare()
        ax.set_default(
            alpha=0.5 + 0.25 * (len(self.data) == 1),
            xlabel="value",
            ylabel="density" if self.density else "count",
        ).loading(self.settings)
        ds, b = self.__hist(
            ax, bins=self.bins if (reflex is None or not self.same_bin) else reflex
        )
        if self.stats:
            ax.set_axes(xlabel=ax.settings.xlabel + "\n" + ds)
        return b

    def __hist(
        self, ax: AxesWrapper, bins: Union[int, List[float]] = 100
    ) -> Tuple[str, List[float]]:
        _, bin_list, _ = ax.ax.hist(
            self.data,
            bins=bins,
            density=self.density,
            alpha=ax.settings.alpha,
            label=self.label,
        )
        mean, std = np.nanmean(self.data), np.nanstd(self.data)
        skew: float = stats.skew(self.data, bias=False, nan_policy="omit")
        kurt: float = stats.kurtosis(self.data, bias=False, nan_policy="omit")
        if self.fit:
            ax.ax.plot(
                bin_list,
                stats.norm.pdf(bin_list, mean, std),
                alpha=ax.settings.alpha,
                label=f"{self.label} · fit",
            )
        return (
            f"{self.label}: mean={mean:.3f}, std={std:.3f}, skew={skew:.3f}, kurt={kurt:.3f}",
            bin_list,
        )
