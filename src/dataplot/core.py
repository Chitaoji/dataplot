"""
Contains the core of dataplot: data(), figure(), etc.

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

import dis
import re
import sys
from math import ceil, sqrt
from types import FrameType
from typing import TYPE_CHECKING, Any, Optional, Unpack

import numpy as np
from validating import validate

from ._typing import FigureSettingDict
from .container import FigWrapper
from .plottable import PlottableData, PlottableDataSet

if TYPE_CHECKING:
    from .artist import Artist


__all__ = ["data", "figure", "randn"]


def _find_user_frame(start_depth: int = 1) -> FrameType | None:
    """
    Find the nearest frame belonging to user code.

    This skips internal dataplot core frames and validating wrappers so helper
    functions keep working when `data()`/`figure()` are decorated.
    """
    try:
        current = sys._getframe(start_depth)
    except ValueError:
        return None

    try:
        while current is not None:
            module_name = str(current.f_globals.get("__name__", ""))
            if module_name != __name__ and not module_name.startswith("validating"):
                return current
            current = current.f_back
        return None
    finally:
        del current


def _infer_var_names(*values: Any) -> list[Optional[str]]:
    search_frame = _find_user_frame(start_depth=2)
    if search_frame is None:
        return [None] * len(values)

    names: list[Optional[str]] = []
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
            names.append(name)
    finally:
        del search_frame
    return names


def _infer_assigned_name() -> Optional[str]:
    """Try inferring assignment target name from call-site."""
    frame = _find_user_frame(start_depth=2)
    if frame is None:
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


@validate
def data(
    *x: Any, name: Optional[str | list[str]] = None, copy: bool = True
) -> PlottableData:
    """
    Initializes a dataset interface which provides methods for mathematical
    operations and plotting.

    Parameters
    ----------
    *x : np.ndarray | Any
        Input values, this takes one or multiple arrays, with each array
        representing a dataset.
    name : str | list[str], optional
        Name(s) of the data, this takes either a single string or a list of strings.
        If a list, should be the same length as the number of input arrays, with
        each element corresponding to a specific array in `x`. If set to None,
        use "x{i}" (i = 1, 2. 3, ...) as the name(s). By default None.
    copy : bool, optional
        Whether to copy input values during normalization. When True, values are
        normalized with ``np.array(..., copy=True)``; when False, ``np.array`` may
        return a view if possible. By default True.

    Returns
    -------
    PlottableData
        Provides methods for mathematical operations and plotting.

    """
    if not x:
        raise ValueError("at least one dataset should be provided")

    expanded_data: list[Any] = []
    expanded_names: list[Optional[str]] = []
    inferred_names = _infer_var_names(*x)
    for i, value in enumerate(x):
        if isinstance(value, PlottableDataSet):
            expanded_data.extend(value.__multiobjects__)
            expanded_names.extend([None] * len(value.__multiobjects__))
        else:
            expanded_data.append(value)
            expanded_names.append(inferred_names[i])

    normalized_data: list[np.ndarray] = []
    for value in expanded_data:
        if isinstance(value, PlottableData):
            normalized_data.append(np.array(value.data, copy=copy).reshape(-1))
        else:
            normalized_data.append(np.array(value, copy=copy).reshape(-1))

    if len(expanded_data) > 1:
        if name is None:
            name = []
            for i, (d, inferred_name) in enumerate(
                zip(expanded_data, expanded_names), start=1
            ):
                if isinstance(d, PlottableData):
                    name.append(d.formatted_name())
                else:
                    name.append(inferred_name if inferred_name is not None else f"x{i}")
        elif isinstance(name, str):
            raise ValueError(
                "for multiple datasets, please provide names as a list of strings"
            )
        elif len(name) != len(expanded_data):
            raise ValueError(f"expected {len(expanded_data)} names, got {len(name)}")
        datas = [PlottableData(d, lb) for d, lb in zip(normalized_data, name)]
        return PlottableDataSet(*datas)

    if isinstance(name, list):
        raise ValueError(
            "it seems not necessary to provide a list of names, since "
            "the data has only one dimension"
        )
    if name is None:
        original_name = (
            expanded_data[0].name
            if isinstance(expanded_data[0], PlottableData)
            else None
        )
        name = original_name or _infer_assigned_name() or expanded_names[0] or "x1"
    return PlottableData(normalized_data[0], name=name)


@validate
def randn(
    n: int, mean: int | float = 0, std: int | float = 1, seed: int = 0
) -> PlottableData:
    """Generate normal-distributed random values as a :class:`PlottableData`."""
    values = np.random.default_rng(seed).normal(loc=mean, scale=std, size=n)
    return data(values, copy=False)


@validate
def figure(
    *artists: "Artist",
    nrows: int | None = None,
    ncols: int | None = None,
    **kwargs: Unpack[FigureSettingDict],
) -> FigWrapper:
    """
    Provides a context manager interface (`__enter__` and `__exit__` methods) for
    creating a figure with subplots and setting various properties for the figure.

    Parameters
    ----------
    *artist : Artist
        List of artists.
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
    len_a = max(len(artists), 1)
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
    figw.set_artists(*artists)
    return figw
