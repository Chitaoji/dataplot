"""
Contains a plotter class: KSPlot.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import TYPE_CHECKING

import numpy as np
from validating import dataclass

from ..utils.math import get_quantile
from .qqplot import QQPlot

if TYPE_CHECKING:
    from ..container import AxesWrapper


__all__ = ["KSPlot"]


@dataclass(validate_methods=True)
class KSPlot(QQPlot):
    """
    A plotter class that creates a K-S plot.

    """

    def paint(self, ax: "AxesWrapper", **_) -> None:
        ax.set_axes(
            title=ax.get_setting("title", "Kolmogorov-Smirnov Plot"),
            xlabel=ax.get_setting("xlabel", "value"),
            ylabel=ax.get_setting("ylabel", "cummulative probability"),
        )
        ax.load(self.settings)
        self.__plot(ax)

    def __plot(self, ax: "AxesWrapper") -> None:
        if no_baseline := self.baseline is None:
            self.baseline = "normal"
        xlabel, p, q1 = self._generate_dist(use_edge_precision=False)
        q2 = get_quantile(self.data, p)
        q1_mask = q1[np.isfinite(q1)]
        q2_mask = q2[np.isfinite(q2)]
        qmin = min(q1_mask[0], q2_mask[0])
        qmax = max(q1_mask[-1], q2_mask[-1])
        q1 = np.concatenate(([qmin], q1, [qmax]))
        q2 = np.concatenate(([qmin], q2, [qmax]))
        p = np.concatenate(([0.0], p, [1.0]))

        new_q1_mask = np.isfinite(q1)
        new_q2_mask = np.isfinite(q2)
        if not no_baseline:
            ax.ax.plot(q1[new_q1_mask], p[new_q1_mask], self.fmt, label=xlabel)
        ax.ax.plot(q2[new_q2_mask], p[new_q2_mask], self.fmt, label=self.name)
        ax.ax.margins(x=0)
