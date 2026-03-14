"""
Contains a plotter class: CorrMap.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from validating import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

import pandas as pd
import seaborn as sns

from .base import Plotter

if TYPE_CHECKING:
    from ..container import AxesWrapper


__all__ = ["CorrMap"]


@dataclass(validate_methods=True)
class CorrMap(Plotter):
    """
    A plotter class that creates a correlation heatmap.

    """

    annot: bool

    def paint(
        self,
        ax: "AxesWrapper",
        *,
        __multi_prev_returned__: Optional[tuple[list["np.ndarray"], list[str]]] = None,
        __multi_is_final__: bool = True,
    ) -> tuple[list["np.ndarray"], list[str]]:
        if __multi_prev_returned__ is None:
            arrays, labels = [], []
        else:
            arrays, labels = __multi_prev_returned__
        arrays.append(self.data)
        labels.append(self.label)
        if __multi_is_final__:
            ax.set_default(title="Correlation Heatmap")
            ax.load(self.settings)
            self.__plot(ax, arrays, labels)
        return arrays, labels

    def __plot(
        self, ax: "AxesWrapper", arrays: list["np.ndarray"], labels: list[str]
    ) -> None:
        corr = pd.DataFrame(arrays, index=labels).T.corr()
        sns.heatmap(corr, ax=ax.ax, annot=self.annot)
