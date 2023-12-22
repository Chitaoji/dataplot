"""
The core of multis: MultiObject, etc.

"""
from typing import Any, Callable, Iterable, Optional, TypeVar, overload

T = TypeVar("T")

__all__ = ["MultiObject", "cleaner", "single"]


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
    * __getitem__()
    * __setitem__()
    * __call__()
    * All public methods
    * All private methods that starts with only one "_"

    And here is the only property that is exposed outside:
    * __multiobjects__ : returns the items

    Parameters
    ----------
    *args : Iterable if specified
        An iterable of the items if specified (refering to what is needed for
        initializing a list). If no argument is given, the constructor creates
        a new empty MultiObject.
    call_reducer : Optional[Callable[[list], Any]], optional
        Specifies a reducer for the returns of `__call__()`. If specified,
        should be a callable that receives the list of original returns, and
        gives back a new return value. If None, the return value of `__call__()`
        will always be a new MultiObject. By default None.
    call_reflex : Optional[str], optional
        If str, the returns of a previous element's `__call__()` will be
        provided to the next element as a keyword argument named by it, by
        default None.

    """

    @overload
    def __init__(
        self,
        call_reducer: Callable[[list], Any] = None,
        call_reflex: Optional[str] = None,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        __iterable: Iterable,
        /,
        call_reducer: Callable[[list], Any] = None,
        call_reflex: Optional[str] = None,
    ) -> None:
        ...

    def __init__(
        self,
        *args,
        call_reducer: Callable[[list], Any] = None,
        call_reflex: Optional[str] = None,
    ) -> None:
        self.__call_reducer = call_reducer
        self.__call_reflex = call_reflex
        self.__items = list(*args)

    def __getattr__(self, __name: str) -> "MultiObject":
        return MultiObject(
            (getattr(x, __name) for x in self.__items),
            call_reducer=self.__call_reducer,
            call_reflex=self.__call_reflex,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        returns = []
        for i, obj in enumerate(self.__items):
            clean_args = [
                a.__multiobjects__[i] if isinstance(a, self.__class__) else a
                for a in args
            ]
            clean_kwargs = {
                k: v.__multiobjects__[i] if isinstance(v, self.__class__) else v
                for k, v in kwargs.items()
            }
            if self.__call_reflex and i > 0:
                clean_kwargs[self.__call_reflex] = r
            returns.append(r := obj(*clean_args, **clean_kwargs))
        if self.__call_reducer:
            return self.__call_reducer(returns)
        return self.__class__(returns, call_reflex=self.__call_reflex)

    def __repr__(self) -> str:
        items_repr = ("\n- ").join(repr(x).replace("\n", "\n  ") for x in self.__items)
        sig_repr = (
            self.__class__.__name__
            + f"(call_reducer={self.__call_reducer.__name__}, call_reflex={self.__call_reflex})"
        )
        return f"{sig_repr} of {len(self.__items)}:\n- {items_repr}"

    @property
    def __multiobjects__(self):
        return self.__items


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


def single(x: T) -> T:
    """
    If a MultiObject is provided, return its last element, otherwise return
    the input itself.

    Parameters
    ----------
    x : T
        Can be a MultiObject or anything else.

    Returns
    -------
    T
        A single object.

    """
    return x.__multiobjects__[-1] if isinstance(x, MultiObject) else x
