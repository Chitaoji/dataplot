"""
Contains typing classes.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

import logging
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict, TypeVar

if TYPE_CHECKING:
    from .artist import Plotter

logging.warning(
    "importing from '._typing'. This module is not aimed to be imported directly, "
    "so unexpected behaviors may be discoverd"
)

PlotSetterVar = TypeVar("PlotSetterVar", bound="Plotter")
DefaultVar = TypeVar("DefaultVar")
StyleStr = Literal[
    "Solarize_Light2",
    "_classic_test_patch",
    "_mpl-gallery",
    "_mpl-gallery-nogrid",
    "bmh",
    "classic",
    "dark_background",
    "fast",
    "fivethirtyeight",
    "ggplot",
    "grayscale",
    "seaborn-v0_8",
    "seaborn-v0_8-bright",
    "seaborn-v0_8-colorblind",
    "seaborn-v0_8-dark",
    "seaborn-v0_8-dark-palette",
    "seaborn-v0_8-darkgrid",
    "seaborn-v0_8-deep",
    "seaborn-v0_8-muted",
    "seaborn-v0_8-notebook",
    "seaborn-v0_8-paper",
    "seaborn-v0_8-pastel",
    "seaborn-v0_8-poster",
    "seaborn-v0_8-talk",
    "seaborn-v0_8-ticks",
    "seaborn-v0_8-white",
    "seaborn-v0_8-whitegrid",
    "tableau-colorblind10",
]
LegendLocStr = Literal[
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
]
FontSizeStr = Literal[
    "xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"
]
FontWeightStr = Literal[
    "ultralight",
    "light",
    "normal",
    "regular",
    "book",
    "medium",
    "roman",
    "semibold",
    "demibold",
    "demi",
    "bold",
    "heavy",
    "extra bold",
    "black",
]
FontStyleStr = Literal["normal", "italic", "oblique"]
ColorStr = Literal["b", "g", "r", "c", "m", "y", "k", "w"]
VerticalAlignmentStr = Literal["baseline", "bottom", "center", "center_baseline", "top"]
HorizontalAlignmentStr = Literal["left", "center", "right"]
SettingKey = Literal[
    "title",
    "xlabel",
    "ylabel",
    "alpha",
    "dpi",
    "figsize",
    "style",
    "legend_loc",
    "subplots_adjust",
]


class SettingDict(TypedDict):
    """
    Dict for the keyword-arguments of `PlotSetter._set()`.

    """

    title: NotRequired[str]
    xlabel: NotRequired[str]
    ylabel: NotRequired[str]
    alpha: NotRequired[float]
    dpi: NotRequired[float]
    figsize: NotRequired[tuple[int, int]]
    style: NotRequired[StyleStr]
    fontdict: NotRequired["FontDict"]
    legend_loc: NotRequired[str]
    subplots_adjust: NotRequired["SubplotDict"]


class FontDict(TypedDict):
    """
    A dictionary controlling the appearance of the title text.

    Unset parameters are left unmodified; initial values are given by
    `matplotlib.rcParams`.

    Parameters
    ----------
    fontsize : float | FontSizeStr, optional
        The font size.
    fontweight : float | FontWeightStr, optional
        The font weight. If a float, should be in range 0-1000.
    fontstyle : FontStyleStr, optional
        The font style.
    color : ColorStr, optional
        The font color.
    verticalalignment : VerticalAlignmentStr, optional
        The vertical alignment relative to the anchor point.
    horizontalalignment : HorizontalAlignmentStr, optional
        The horizontal alignment relative to the anchor point.

    """

    fontsize: NotRequired[float | FontSizeStr]
    fontweight: NotRequired[float | FontWeightStr]
    fontstyle: NotRequired[FontStyleStr]
    color: NotRequired[ColorStr]
    verticalalignment: NotRequired[VerticalAlignmentStr]
    horizontalalignment: NotRequired[HorizontalAlignmentStr]


class SubplotDict(TypedDict):
    """
    A dictionary controlling the subplot layout parameters.

    Unset parameters are left unmodified; initial values are given by
    `matplotlib.rcParams`, whose recommended values are {"left": 0.125,
    "bottom": 0.11, "right": 0.9, "top": 0.88, "wspace": 0.2, "hspace":
    0.2}.

    Parameters
    ----------
    left / right / bottom / top : float, optional
        The position of the left / right / bottom / top edge of the
        subplots, as a fraction of the figure width.
    wspace / hspace : float, optional
        The width / height of the padding between subplots, as a fraction
        of the average Axes width / height.

    """

    left: NotRequired[float]
    bottom: NotRequired[float]
    right: NotRequired[float]
    top: NotRequired[float]
    wspace: NotRequired[float]
    hspace: NotRequired[float]
