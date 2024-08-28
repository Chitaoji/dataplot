"""
The core of multi: multi(), multipartial(), etc.

"""

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    LiteralString,
    Optional,
    TypeVar,
    overload,
)

if TYPE_CHECKING:
    from hintwith import hintwith

T = TypeVar("T")
S = TypeVar("S", bound=LiteralString)


__all__ = [
    "MultiObject",
    "multi",
    "multipartial",
    "single",
    "multiple",
    "REMAIN",
    "UNSUBSCRIPTABLE",
]


class MultiObject(Generic[T]):
    """
    A basic object that enables multi-element attribute-getting,
    attribute-setting, calling, etc. This object maintains a list of items,
    and if a method (including some magic methods, see below) is called, each
    item's method with the same name will be called instead, and the results
    come as a new MultiObject.

    Here are the methods that will be overloaded:
    * __getattr__()
    * __setattr__()
    * __call__()
    * __getitem__()
    * __setitem__()
    * All public methods
    * All private methods that starts with only one "_"

    And here is the only property that is exposed outside:
    * __multiobjects__ : returns the items

    Parameters
    ----------
    __iterable : Iterable, optional
        If not given, the constructor creates a new empty MultiObject. If
        specified, the argument must be an iterable (the same as what is needed
        for creating a list). By default None.
    call_reducer : Callable[[list], Any], optional
        Specifies a reducer for the returns of `__call__()`. If specified,
        should be a callable that receives the list of original returns, and
        gives back a reduced value. If None, the reduced value will always be a
        new MultiObject. By default None.
    call_reflex : str, optional
        If str, the returns of a previous element's `__call__()` will be
        provided to the next element as a keyword argument named by it, by
        default None.
    attr_reducer: Callable[[str], Callable[[list], Any]],  optional
        Specifies a reducer for the returns of `__getattr__()`. If specified,
        should be a callable that receives the attribute name, and gives back a
        new callable. The new callable will receive the list of original returns,
        and gives back a reduced value. If None, the reduced value will always be
        a new MultiObject. By default None.

    """

    def __init__(
        self,
        __iterable: Optional[Iterable] = None,
        *,
        call_reducer: Optional[Callable[[list], Any]] = None,
        call_reflex: Optional[str] = None,
        attr_reducer: Optional[Callable[[str], Callable[[list], Any]]] = None,
    ) -> None:
        self.__call_reducer = call_reducer
        self.__call_reflex = call_reflex
        self.__attr_reducer = attr_reducer
        self.__items: list[S] = [] if __iterable is None else list(__iterable)

    def __getattr__(self, __name: str) -> "MultiObject | Any":
        if __name.startswith("__"):
            raise AttributeError(f"cannot reach attribute '{__name}'")
        attrs = [getattr(x, __name) for x in self.__items]
        if self.__attr_reducer:
            reduced = self.__attr_reducer(__name)(attrs)
            if reduced != REMAIN:
                return reduced
        return MultiObject(attrs)

    def __setattr__(self, __name: Any, __value: Any) -> None:
        if isinstance(__name, str) and __name.startswith("_"):
            super().__setattr__(__name, __value)
        else:
            for i, obj in enumerate(self.__items):
                setattr(obj, single(__name, n=i), single(__value, n=i))

    def __call__(self, *args: Any, **kwargs: Any) -> "MultiObject | Any":
        returns = []
        len_items = len(self.__items)
        for i, obj in enumerate(self.__items):
            a = [single(x, n=i) for x in args]
            kwd = {k: single(v, n=i) for k, v in kwargs.items()}
            if self.__call_reflex and i > 0:
                kwd["__multi_last_call__"] = i == len_items - 1
                kwd[self.__call_reflex] = r
            returns.append(r := obj(*a, **kwd))
        if self.__call_reducer:
            reduced = self.__call_reducer(returns)
            if reduced != REMAIN:
                return reduced
        return MultiObject(returns, call_reflex=self.__call_reflex)

    def __getitem__(self, __key: Any) -> T | "MultiObject":
        items = [x[__key] for x in self.__items]
        if isinstance(__key, int) and UNSUBSCRIPTABLE in items:
            return self.__items[__key]
        return MultiObject(items)

    def __setitem__(self, __key: Any, __value: Any) -> None:
        for i, obj in enumerate(self.__items):
            obj[single(__key, n=i)] = single(__value, n=i)

    def __repr__(self) -> str:
        return ("\n").join("- " + repr(x).replace("\n", "\n  ") for x in self.__items)

    def __str__(self) -> str:
        signature = self.__class__.__name__ + repr_not_none(self)
        return f"{signature}"

    @property
    def __multiobjects__(self) -> list[T]:
        return self.__items


def repr_not_none(x: MultiObject) -> str:
    """
    Returns a string representation of the MultiObject's attributes with
    not-None values. Attributes with values of None are ignored.

    Parameters
    ----------
    x : MultiObject
        Any object.

    Returns
    -------
    str
        String representation.

    """
    namelist = [n for n in x.__init__.__code__.co_varnames[1:] if not n.startswith("_")]
    not_nones: list[str] = []
    for n in namelist:
        if not hasattr(x, p := f"_{type(x).__name__}__{n}"):
            continue
        if (v := getattr(x, p)) is None:
            continue
        if isinstance(v, Callable):
            v = v.__name__
        not_nones.append(f"{n}={v}")
    return "" if len(not_nones) == 0 else "(" + ", ".join(not_nones) + ")"


@dataclass(slots=True)
class MultiFlag(Generic[S]):
    """Flag for MultiObjects."""

    flag: int
    name: S
    err: Optional[type[Exception]] = None
    errmsg: str = ""

    def __repr__(self) -> str:
        if self.err is not None:
            raise self.err(self.errmsg)
        return self.name

    def __eq__(self, __value: Any) -> bool:
        if isinstance(__value, MultiFlag):
            return self.flag == __value.flag
        return False


REMAIN = MultiFlag(0, "REMAIN")
UNSUBSCRIPTABLE = MultiFlag(
    1, "UNSUBSCRIPTABLE", TypeError, "object is not subscriptable"
)

if TYPE_CHECKING:

    @hintwith(MultiObject)
    def multi() -> MultiObject:
        """Same to `MultiObject()`."""

else:

    def multi(*args, **kwargs) -> MultiObject:
        """Magic happens."""
        return MultiObject(*args, **kwargs)


def multipartial(**kwargs) -> Callable[[list | str], Any]:
    """
    Returns a MultiObject constructor with partial application of the
    given arguments and keywords.

    Returns
    -------
    Callable[[list | str], Any]
        A MultiObject constructor.

    """

    def multi_constructor(x: list | str):
        if isinstance(x, str):
            return multi_constructor
        return MultiObject(x, **kwargs)

    return multi_constructor


def single(x: S, n: int = -1) -> S:
    """
    If a MultiObject is provided, return its n-th element, otherwise return
    the input itself.

    Parameters
    ----------
    x : T
        Can be a MultiObject or anything else.
    n : int, optional
        Specifies which element to return if a MultiObject is provided, by
        default -1.

    Returns
    -------
    T
        A single object.

    """
    return x.__multiobjects__[n] if isinstance(x, MultiObject) else x


def multiple(x: S) -> list[S]:
    """
    If a MultiObject is provided, return a list of its elements, otherwise
    return `[x]`.

    Parameters
    ----------
    x : T
        Can be a MultiObject or anything else.

    Returns
    -------
    list[T]
        List of elements.

    """
    return x.__multiobjects__ if isinstance(x, MultiObject) else [x]


@overload
def cleaner(x: list) -> list | None: ...
@overload
def cleaner(x: str) -> Callable[[list], list | None]: ...
def cleaner(x: list | str) -> list | None:
    """
    If the list is consist of None's only, return None, otherwise return
    a MultiObject instantiated by the list.

    Parameters
    ----------
    x : list | str
        May be a list.

    Returns
    -------
    Any
        May be None or a MultiObject instantiated by the list.

    """
    if isinstance(x, str):
        return cleaner
    if all(i is None for i in x):
        return None
    return MultiObject(x, call_reducer=cleaner, attr_reducer=cleaner)
