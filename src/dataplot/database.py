"""Contains calculation-only data behaviors."""

from abc import ABCMeta
from typing import Any, Callable, Optional, Self

import numpy as np
from scipy.stats import norm
from validating import attr, dataclass

from ._typing import ResampleRule
from .utils.multi import UNSUBSCRIPTABLE

__all__ = ["Data"]


@dataclass(validate_methods=True)
class Data(metaclass=ABCMeta):
    """Calculation-focused base class for plottable datasets."""

    data: np.ndarray
    label: Optional[str] = attr(default=None)
    fmtb: str = attr(init=False, default="{0}")
    original_data: np.ndarray = attr(init=False)
    priority: int = attr(init=False, default=0)

    def __post_init__(self) -> None:
        self.label = "x" if self.label is None else self.label
        self.original_data = self.data

    def __getitem__(self, __key: int):
        return UNSUBSCRIPTABLE

    def _create_data(
        self, fmt: str, data: np.ndarray, priority: int = 0, label: Optional[str] = None
    ) -> Self:
        obj = self.customize(
            self.__class__,
            self.original_data,
            self.label if label is None else label,
            fmtb=fmt,
            priority=priority,
        )
        obj.data = data
        return obj

    def __neg__(self) -> Self:
        new_fmt = f"(-{self.__remove_brackets(self.fmtb, priority=28)})"
        return self._create_data(new_fmt, -self.data, priority=40)

    def __add__(self, __other: float | int | Self) -> Self:
        return self.__binary_operation(__other, "+", np.add, priority=30)

    def __radd__(self, __other: float | int | Self) -> Self:
        return self.__binary_operation(__other, "+", np.add, reverse=True, priority=30)

    def __sub__(self, __other: float | int | Self) -> Self:
        return self.__binary_operation(__other, "-", np.subtract, priority=29)

    def __rsub__(self, __other: float | int | Self) -> Self:
        return self.__binary_operation(
            __other, "-", np.subtract, reverse=True, priority=29
        )

    def __mul__(self, __other: float | int | Self) -> Self:
        return self.__binary_operation(__other, "*", np.multiply, priority=20)

    def __rmul__(self, __other: float | int | Self) -> Self:
        return self.__binary_operation(
            __other, "*", np.multiply, reverse=True, priority=20
        )

    def __truediv__(self, __other: float | int | Self) -> Self:
        return self.__binary_operation(__other, "/", np.true_divide, priority=19)

    def __rtruediv__(self, __other: float | int | Self) -> Self:
        return self.__binary_operation(
            __other, "/", np.true_divide, reverse=True, priority=19
        )

    def __pow__(self, __other: float | int | Self) -> Self:
        return self.__binary_operation(__other, "**", np.power)

    def __rpow__(self, __other: float | int | Self) -> Self:
        return self.__binary_operation(__other, "**", np.power, reverse=True)

    def __binary_operation(
        self,
        other: float | int | Self,
        sign: str,
        func: Callable[[Any, Any], np.ndarray],
        reverse: bool = False,
        priority: int = 10,
    ) -> Self:
        if reverse:
            this_fmt = self.__remove_brackets(self.fmtb, priority=priority)
            return self._create_data(
                f"({other}{sign}{this_fmt})", func(other, self.data), priority
            )
        this_fmt = self.__remove_brackets(self.fmtb, priority=priority + 1)
        if isinstance(other, (float, int)):
            return self._create_data(
                f"({this_fmt}{sign}{other})", func(self.data, other), priority
            )
        if isinstance(other, Data):
            return self._create_data(
                f"({this_fmt}{sign}{other.formatted_label(priority=priority)})",
                func(self.data, other.data),
                priority,
            )
        raise ValueError(
            f"{sign!r} not supported between instances of 'Data' and {other.__class__.__name__!r}"
        )

    def __remove_brackets(self, string: str, priority: int = 0):
        if priority == 0 or self.priority <= priority:
            if string.startswith("(") and string.endswith(")"):
                return string[1:-1]
        return string

    @property
    def format(self) -> str:
        return self.__remove_brackets(self.fmtb)

    def formatted_label(self, priority: int = 0) -> str:
        if priority == self.priority and priority in (19, 29):
            priority -= 1
        return self.__remove_brackets(self.fmtb.format(self.label), priority=priority)

    def resample(self, n: int, rule: ResampleRule = "head") -> Self:
        match rule:
            case "random":
                idx = np.random.randint(0, len(self.data), n)
                new_data = self.data[idx]
            case "head":
                new_data = self.data[:n]
            case "tail":
                new_data = self.data[-n:]
            case _:
                raise ValueError(f"rule not supported: {rule!r}")
        return self._create_data(f"resample({self.format}, {n})", new_data)

    def rank(self, pct: bool = True) -> Self:
        valid_mask = np.isfinite(self.data)
        valid_data = self.data[valid_mask]
        ranks = np.full(self.data.shape, np.nan, dtype=float)
        if valid_data.size > 0:
            order = np.argsort(valid_data, kind="mergesort")
            sorted_data = valid_data[order]
            ranked = np.empty(valid_data.size, dtype=float)
            change_points = np.flatnonzero(np.diff(sorted_data) != 0) + 1
            starts = np.concatenate(([0], change_points))
            ends = np.concatenate((change_points, [valid_data.size]))
            avg_ranks = (starts + 1 + ends) / 2.0
            ranked[order] = np.repeat(avg_ranks, ends - starts)
            if pct:
                ranked /= valid_data.size
            ranks[valid_mask] = ranked
        new_fmt = f"rank({self.format}, pct=True)" if pct else f"rank({self.format})"
        return self._create_data(new_fmt, ranks)

    def normrank(self) -> Self:
        ranks = self.rank(pct=False).data
        valid_mask = np.isfinite(ranks)
        new_data = np.full(self.data.shape, np.nan, dtype=float)
        if np.any(valid_mask):
            n = np.count_nonzero(valid_mask)
            p = (ranks[valid_mask] - 0.5) / n
            new_data[valid_mask] = norm.ppf(p)
        return self._create_data(f"rank_normalize({self.format})", new_data)

    def ranknorm(self) -> Self:
        return self.normrank()

    def log(self) -> Self:
        return self._create_data(
            f"log({self.format})", np.log(np.where(self.data > 0, self.data, np.nan))
        )

    def log10(self) -> Self:
        return self._create_data(
            f"log10({self.format})",
            np.log10(np.where(self.data > 0, self.data, np.nan)),
        )

    def signedlog(self) -> Self:
        new_data = np.log(np.where(self.data > 0, self.data, np.nan))
        new_data[self.data < 0] = np.log(-self.data[self.data < 0])
        new_data[self.data == 0] = 0
        return self._create_data(f"signedlog({self.format})", new_data)

    def signedlog10(self) -> Self:
        new_data = np.log10(np.where(self.data > 0, self.data, np.nan))
        new_data[self.data < 0] = np.log10(-self.data[self.data < 0])
        new_data[self.data == 0] = 0
        return self._create_data(f"signedlog({self.format})", new_data)

    def pow(self, n: int | float = 2) -> Self:
        return self._create_data(f"pow({self.format}, {n})", self.data**n)

    def signedpow(self, n: int | float) -> Self:
        new_data = np.where(self.data > 0, self.data, np.nan) ** n
        new_data[self.data < 0] = -((-self.data[self.data < 0]) ** n)
        new_data[self.data == 0] = 0
        return self._create_data(f"signedpow({self.format})", new_data)

    def root(self, n: int = 2) -> Self:
        if n == 0:
            raise ValueError("root degree must not be zero")
        return self._create_data(
            f"root({self.format}, {n})", np.power(self.data, 1 / n)
        )

    def sqrt(self) -> Self:
        return self.root(2)

    def signedroot(self, n: int = 2) -> Self:
        if n == 0:
            raise ValueError("root degree must not be zero")
        power = 1 / n
        new_data = np.power(np.where(self.data > 0, self.data, np.nan), power)
        new_data[self.data < 0] = -np.power(-self.data[self.data < 0], power)
        new_data[self.data == 0] = 0
        return self._create_data(f"signedroot({self.format}, {n})", new_data)

    def signedsqrt(self) -> Self:
        return self.signedroot(2)

    def rolling(self, n: int) -> Self:
        if n < 1:
            raise ValueError(f"rolling window must be a positive integer, got {n}")
        new_data = np.convolve(
            np.asarray(self.data, dtype=float), np.ones(n, dtype=float), mode="full"
        )[: len(self.data)] / np.minimum(np.arange(1, len(self.data) + 1), n)
        return self._create_data(f"rolling({self.format}, {n})", new_data)

    def exp(self) -> Self:
        return self._create_data(f"exp({self.format})", np.exp(self.data))

    def exp10(self) -> Self:
        return self._create_data(f"exp10({self.format})", 10**self.data)

    def signedexp(self) -> Self:
        new_data = np.exp(np.where(self.data > 0, self.data, np.nan))
        new_data[self.data < 0] = -np.exp(-self.data[self.data < 0])
        new_data[self.data == 0] = 0
        return self._create_data(f"signedexp({self.format})", new_data)

    def signedexp10(self) -> Self:
        new_data = 10 ** np.where(self.data > 0, self.data, np.nan)
        new_data[self.data < 0] = -(10 ** (-self.data[self.data < 0]))
        new_data[self.data == 0] = 0
        return self._create_data(f"signedexp10({self.format})", new_data)

    def abs(self) -> Self:
        return self._create_data(f"abs({self.format})", np.abs(self.data))

    def demean(self) -> Self:
        return self._create_data(
            f"({self.format}-mean({self.format}))", self.data - np.nanmean(self.data)
        )

    def zscore(self) -> Self:
        return self._create_data(
            f"zscore({self.format})",
            (self.data - np.nanmean(self.data)) / np.nanstd(self.data),
        )

    def cumsum(self) -> Self:
        return self._create_data(f"csum({self.format})", np.cumsum(self.data))
