"""Color helpers for artist modules."""

from matplotlib.colors import to_rgba


def darken_color(color: object, factor: float = 0.75) -> tuple[float, ...]:
    """Return a darker RGBA color by scaling RGB channels."""
    r, g, b, a = to_rgba(color)
    return (r * factor, g * factor, b * factor, a)
