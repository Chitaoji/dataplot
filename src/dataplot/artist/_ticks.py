"""
Shared tick helpers for chart plotters.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.ticker import FixedLocator

if TYPE_CHECKING:
    from ..container import AxesWrapper


def ensure_rightmost_xtick_label(ax: "AxesWrapper", xticks: Any) -> None:
    """Preserve the rightmost x tick label without crowding nearby labels."""
    xticks_array = np.asarray(list(xticks))
    if xticks_array.size == 0:
        return
    if not (
        np.issubdtype(xticks_array.dtype, np.number)
        or np.issubdtype(xticks_array.dtype, np.datetime64)
    ):
        return

    converted_xticks = np.asarray(ax.ax.convert_xunits(xticks_array), dtype=float)
    rightmost = float(np.max(converted_xticks))
    current_ticks = np.asarray(ax.ax.get_xticks(), dtype=float)
    if current_ticks.size == 0:
        ax.ax.xaxis.set_major_locator(FixedLocator([rightmost]))
        return

    if np.any(np.isclose(current_ticks, rightmost)):
        return

    x_min, x_max = sorted(ax.ax.get_xlim())
    visible_ticks = np.sort(
        current_ticks[(current_ticks >= x_min) & (current_ticks <= x_max)]
    )
    tick_steps = np.diff(visible_ticks)
    tick_steps = tick_steps[tick_steps > 0]

    if tick_steps.size > 0:
        min_gap = float(np.median(tick_steps)) * 0.8
        keep_mask = np.abs(current_ticks - rightmost) >= min_gap
        current_ticks = current_ticks[keep_mask]

    merged_ticks = np.sort(np.append(current_ticks, rightmost))
    ax.ax.xaxis.set_major_locator(FixedLocator(merged_ticks))
