"""
Contains the core of dataplot: figure(), data(), show(), etc.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

import dis
import re
import sys
from math import ceil, sqrt
from typing import TYPE_CHECKING, Any, Optional, Unpack

import numpy as np

from ._typing import FigureSettingDict
from .container import FigWrapper
from .dataset import PlotDataSet, PlotDataSets

if TYPE_CHECKING:
    from .artist import Artist


__all__ = ["data", "figure"]


def _infer_var_names(*values: Any) -> list[Optional[str]]:
    try:
        search_frame = sys._getframe(1)
    except ValueError:
        return [None] * len(values)

    labels: list[Optional[str]] = []
    try:
        for value in values:
            name = None
            current = search_frame
            while current is not None and name is None:
                local_items = list(current.f_locals.items())
                global_items = list(current.f_globals.items())
                name = next((k for k, v in local_items if v is value), None)
                if name is None:
                    name = next((k for k, v in global_items if v is value), None)
                current = current.f_back
            labels.append(name)
    finally:
        del search_frame
    return labels


def _infer_assigned_name() -> Optional[str]:
    """Try inferring assignment target name from call-site."""
    try:
        frame = sys._getframe(2)
    except ValueError:
        return None

    # Bytecode inspection works in REPL/notebook contexts where source code file
    # is unavailable.
    try:
        instructions = list(dis.get_instructions(frame.f_code))
        store_index = next(
            (
                i
                for i, ins in enumerate(instructions)
                if ins.offset > frame.f_lasti
                and ins.opname in {"STORE_NAME", "STORE_FAST", "STORE_GLOBAL"}
            ),
            None,
        )
        if store_index is not None:
            # `a = b = data(...)` compiles to STORE_* b then STORE_* a;
            # returning the last one better matches user expectation.
            last_store = store_index
            while last_store + 1 < len(instructions) and instructions[
                last_store + 1
            ].opname in {"STORE_NAME", "STORE_FAST", "STORE_GLOBAL"}:
                last_store += 1
            return str(instructions[last_store].argval)
    except Exception:
        pass

    try:
        context = frame.f_code.co_filename
        lineno = frame.f_lineno
        with open(context, "r", encoding="utf-8") as f:
            lines = f.readlines()
        line = lines[lineno - 1].strip()
    except (OSError, IndexError):
        return None
    finally:
        del frame

    if "=" not in line:
        return None
    lhs = line.split("=", 1)[0].strip()
    if not lhs:
        return None
    m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", lhs)
    return m.group(1) if m else None


def data(*x: Any, label: Optional[str | list[str]] = None) -> PlotDataSet:
    """
    Initializes a dataset interface which provides methods for mathematical
    operations and plotting.

    Parameters
    ----------
    *x : np.ndarray | Any
        Input values, this takes one or multiple arrays, with each array
        representing a dataset.
    label : str | list[str], optional
        Label(s) of the data, this takes either a single string or a list of strings.
        If a list, should be the same length as the number of input arrays, with
        each element corresponding to a specific array in `x`. If set to None,
        use "x{i}" (i = 1, 2. 3, ...) as the label(s). By default None.

    Returns
    -------
    PlotDataSet
        Provides methods for mathematical operations and plotting.

    """
    if not x:
        raise ValueError("at least one dataset should be provided")

    expanded_data: list[Any] = []
    expanded_names: list[Optional[str]] = []
    inferred_names = _infer_var_names(*x)
    for i, value in enumerate(x):
        if isinstance(value, PlotDataSets):
            expanded_data.extend(value.__multiobjects__)
            expanded_names.extend([None] * len(value.__multiobjects__))
        else:
            expanded_data.append(value)
            expanded_names.append(inferred_names[i])

    normalized_data: list[np.ndarray] = []
    for value in expanded_data:
        if isinstance(value, PlotDataSet):
            normalized_data.append(np.array(value.data))
        else:
            normalized_data.append(np.array(value))

    if len(expanded_data) > 1:
        if label is None:
            label = []
            for i, (d, inferred_name) in enumerate(
                zip(expanded_data, expanded_names), start=1
            ):
                if isinstance(d, PlotDataSet):
                    label.append(d.formatted_label())
                else:
                    label.append(
                        inferred_name if inferred_name is not None else f"x{i}"
                    )
        elif isinstance(label, str):
            raise ValueError(
                "for multiple datasets, please provide labels as a list of strings"
            )
        elif len(label) != len(expanded_data):
            raise ValueError(
                "label should have the same length as x "
                f"({len(expanded_data)}), got {len(label)}"
            )
        datas = [PlotDataSet(d, lb) for d, lb in zip(normalized_data, label)]
        return PlotDataSets(*datas)

    if isinstance(label, list):
        raise ValueError(
            "it seems not necessary to provide a list of labels, since "
            "the data has only one dimension"
        )
    if label is None:
        original_label = (
            expanded_data[0].label if isinstance(expanded_data[0], PlotDataSet) else None
        )
        label = (
            original_label
            or _infer_assigned_name()
            or expanded_names[0]
            or "x1"
        )
    return PlotDataSet(normalized_data[0], label=label)


def figure(
    artist: "Artist | list[Artist]",
    nrows: int | None = None,
    ncols: int | None = None,
    **kwargs: Unpack[FigureSettingDict],
) -> FigWrapper:
    """
    Provides a context manager interface (`__enter__` and `__exit__` methods) for
    creating a figure with subplots and setting various properties for the figure.

    Parameters
    ----------
    artist : Artist | list[Artist]
        Artist or list of artists.
    nrows : int, optional
        Determines how many subplots can be arranged vertically in the figure,
        If None, will be automatically set according to ``len(artist)``. By default
        None.
    ncols : int, optional
        Determines how many subplots can be arranged horizontally in the figure.
        If None, will be automatically set according to ``len(artist)``. By default
        None.
    **kwargs : **FigureSettingDict
        Specifies the figure settings, see `FigWrapper.set_figure()` for more details.

    Returns
    -------
    FigWrapper
        A wrapper of figure.

    """
    if not isinstance(artist, list):
        artist = [artist]
    len_a = max(len(artist), 1)
    if nrows is None and ncols is None:
        ncols = int(sqrt(len_a))
        nrows = ceil(len_a / ncols)
    elif ncols is None:
        nrows = min(nrows, len_a)
        ncols = ceil(len_a / ncols)
    else:
        ncols = min(ncols, len_a)
        nrows = ceil(len_a / ncols)
    figw = FigWrapper(nrows=nrows, ncols=ncols)
    figw.set_figure(**kwargs)
    figw.artists = artist
    return figw
