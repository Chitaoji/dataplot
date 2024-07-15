"""
Contains a plotter class: CorrMap.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import TYPE_CHECKING, Optional

import pandas as pd
import seaborn as sns
from attrs import define

from .base import Plotter

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..container import AxesWrapper


__all__ = ["CorrMap"]


@define
class CorrMap(Plotter):
    """
    A plotter class that creates a correlation heatmap.

    """

    annot: bool

    def paint(
        self,
        ax: "AxesWrapper",
        reflex: Optional[tuple[list["NDArray"], list[str]]] = None,
        __multi_last_call__: bool = False,
    ) -> tuple[list["NDArray"], list[str]]:
        if reflex is None:
            reflex = ([], [])
        arrays, labels = reflex
        arrays.append(self.data)
        labels.append(self.label)
        if __multi_last_call__:
            ax.set_default(title="Correlation Heatmap")
            ax.loading(self.settings)
            self.__plot(ax, arrays, labels)
        return arrays, labels

    def __plot(
        self, ax: "AxesWrapper", arrays: list["NDArray"], labels: list[str]
    ) -> None:
        corr = pd.DataFrame(arrays, index=labels).T.corr()
        sns.heatmap(corr, ax=ax.ax, annot=self.annot)
