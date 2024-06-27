"""
The core of multis: MultiObject, etc.

"""

from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    Optional,
    TypeVar,
    overload,
)

from attrs import define

T = TypeVar("T")


__all__ = ["MultiObject", "REMAIN", "multi", "multi_partial", "cleaner", "single"]


class MultiObject:
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
    *args : Iterable if specified
        An iterable of the items if specified (the same as what is needed for
        initializing a list). If no argument is given, the constructor creates
        a new empty MultiObject.
    call_reducer : Optional[Callable[[list], Any]], optional
        Specifies a reducer for the returns of `__call__()`. If specified,
        should be a callable that receives the list of original returns, and
        gives back a new return. If None, the return will be a new MultiObject.
        By default None.
    call_reflex : Optional[str], optional
        If str, the returns of a previous element's `__call__()` will be
        provided to the next element as a keyword argument named by it, by
        default None.
    attr_reducer: Optional[Callable[[list, str], Any]] = None,
        Specifies a reducer for the returns of `__getattr__()`. If specified,
        should be a callable that receives 2 positional arguments: the list of
        original returns and the attribute name, and gives back a new return. If
        None, the return will be a new MultiObject. By default None.
    """

    @overload
    def __init__(
        self,
        call_reducer: Optional[Callable[[list], Any]] = None,
        call_reflex: Optional[str] = None,
        attr_reducer: Optional[Callable[[list, str], Any]] = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        __iterable: Iterable,
        /,
        call_reducer: Optional[Callable[[list], Any]] = None,
        call_reflex: Optional[str] = None,
        attr_reducer: Optional[Callable[[list, str], Any]] = None,
    ) -> None: ...

    def __init__(
        self,
        *args,
        call_reducer: Optional[Callable[[list], Any]] = None,
        call_reflex: Optional[str] = None,
        attr_reducer: Optional[Callable[[list, str], Any]] = None,
    ) -> None:
        self.__call_reducer = call_reducer
        self.__call_reflex = call_reflex
        self.__attr_reducer = attr_reducer
        self.__items = list(*args)

    def __getattr__(self, __name: str) -> "MultiObject":
        attrs = [getattr(x, __name) for x in self.__items]
        if self.__attr_reducer:
            reduced = self.__attr_reducer(attrs, __name)
            if isinstance(reduced, MultiFlag) and reduced == REMAIN:
                pass
            else:
                return reduced
        return MultiObject(attrs)

    def __setattr__(self, __name: Any, __value: Any) -> None:
        if isinstance(__name, str) and __name.startswith("_"):
            super().__setattr__(__name, __value)
        else:
            for i, obj in enumerate(self.__items):
                setattr(obj, single(__name, n=i), single(__value, n=i))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        returns = []
        for i, obj in enumerate(self.__items):
            clean_args = [single(a, n=i) for a in args]
            clean_kwargs = {k: single(v, n=i) for k, v in kwargs.items()}
            if self.__call_reflex and i > 0:
                clean_kwargs[self.__call_reflex] = r
            returns.append(r := obj(*clean_args, **clean_kwargs))
        if self.__call_reducer:
            reduced = self.__call_reducer(returns)
            if isinstance(reduced, MultiFlag) and reduced == REMAIN:
                pass
            else:
                return reduced
        return self.__class__(returns, call_reflex=self.__call_reflex)

    def __getitem__(self, __key: str) -> "MultiObject":
        return MultiObject((x[__key] for x in self.__items))

    def __setitem__(self, __key: str, __value: Any) -> "MultiObject":
        for i, obj in enumerate(self.__items):
            obj[single(__key, n=i)] = single(__value, n=i)

    def __repr__(self) -> str:
        items = ("\n- ").join(repr(x).replace("\n", "\n  ") for x in self.__items)
        call_reducer = self.__call_reducer.__name__ if self.__call_reducer else None
        signature = (
            self.__class__.__name__
            + f"(call_reducer={call_reducer}, call_reflex={self.__call_reflex!r}, "
            f"attr_reducer={self.__attr_reducer})"
        )
        return f"{signature}\n- {items}"

    @property
    def __multiobjects__(self):
        return self.__items


@define
class MultiFlag(Generic[T]):
    """Flag for MultiObjects."""

    flag: T

    def __eq__(self, __value: "MultiFlag") -> bool:
        return self.flag == __value.flag


REMAIN: MultiFlag[Literal[0]] = MultiFlag(0)


def multi(*args, **kwargs) -> MultiObject:
    """
    Same to `MultiObject()`.

    Returns
    -------
    MultiObject
        A MultiObject.

    """
    return MultiObject(*args, **kwargs)


def multi_partial(*args, **kwargs) -> Callable[[list], MultiObject]:
    """
    Returns a MultiObject constructor with partial application of the
    given arguments and keywords.

    Returns
    -------
    Callable[[list], MultiObject]
        A MultiObject constructor.

    """

    def multi_constructor(x, *_, **__):
        return MultiObject(x, *args, **kwargs)

    return multi_constructor


def cleaner(x: list) -> Optional[list]:
    """
    If the list is consist of None's only, return None, otherwise return
    a MultiObject instantiated by the list.

    Parameters
    ----------
    x : MultiObject
        A list.

    Returns
    -------
    Optional[list]
        None or a MultiObject instantiated by the list.

    """
    if all(i is None for i in x):
        return None
    return MultiObject(x, call_reducer=cleaner)


def single(x: T, n: int = -1) -> T:
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
