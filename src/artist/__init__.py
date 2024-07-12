"""
Contains artists.

"""

from . import base, histogram, ksplot, linechart, qqplot
from .base import *
from .histogram import *
from .ksplot import *
from .linechart import *
from .qqplot import *

__all__: list[str] = []
__all__.extend(base.__all__)
__all__.extend(histogram.__all__)
__all__.extend(ksplot.__all__)
__all__.extend(ksplot.__all__)
__all__.extend(qqplot.__all__)
