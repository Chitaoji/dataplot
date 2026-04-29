"""
Contains a plotter class: CorrMap.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from validating import dataclass

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
        __multi_prev_returned__: Optional[tuple[list[np.ndarray], list[str]]] = None,
        __multi_is_final__: bool = True,
    ) -> tuple[list[np.ndarray], list[str]]:
        if __multi_prev_returned__ is None:
            arrays, names = [], []
        else:
            arrays, names = __multi_prev_returned__
        arrays.append(self.data)
        names.append(self.name)
        if __multi_is_final__:
            ax.set_axes(title=ax.get_setting("title", "Correlation Heatmap"))
            ax.load(self.settings)
            self.__plot(ax, arrays, names)
        return arrays, names

    def __plot(
        self, ax: "AxesWrapper", arrays: list[np.ndarray], names: list[str]
    ) -> None:
        corr = pd.DataFrame(arrays, index=names).T.corr()
        sns.heatmap(corr, ax=ax.ax, annot=self.annot)
