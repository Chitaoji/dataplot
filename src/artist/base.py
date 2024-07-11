"""
Contains the core of artist: figure(), data().

NOTE: this module is private. All functions and objects are available in the main
`dataplot` namespace - use that instead.

"""

from typing import TYPE_CHECKING, Optional

from attrs import define, field

from ..plotter import Plotter

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..container import AxesWrapper

__all__ = ["Artist"]


@define(init=False, slots=False)
class Artist(Plotter):
    """Paints images on an axes, based on the data, labels, and other settings."""

    data: Optional["NDArray"] = field(default=None, init=False)
    label: Optional[str] = field(default=None, init=False)
    on: Optional["AxesWrapper"] = field(default=None, kw_only=True)

    def prepare(self) -> "AxesWrapper":
        """
        Prepares the data and the label. Raises an error if they are not
        passed in correctly.

        Returns
        -------
        AxesWrapper
            An instance of `AxesWrapper`.

        Raises
        ------
        DataSetterError
            Raised when the data or the label is not set yet.

        """
        for name in ["data", "label", "on"]:
            if getattr(self, name) is None:
                raise ArtistPrepareError(f"'{name}' not set yet.")
        return self.on

    def paint(self, reflex: None = None) -> None:
        """Paint on the axes."""


class ArtistPrepareError(Exception):
    """Raised when data or labels are not set yet."""
