"""
Contains artists.

"""

from . import base, corrmap, histogram, ksplot, linechart, ppplot, qqplot, scatterchart
from .base import *
from .corrmap import *
from .histogram import *
from .ksplot import *
from .linechart import *
from .ppplot import *
from .qqplot import *
from .scatterchart import *

__all__: list[str] = []
__all__.extend(base.__all__)
__all__.extend(corrmap.__all__)
__all__.extend(histogram.__all__)
__all__.extend(ksplot.__all__)
__all__.extend(linechart.__all__)
__all__.extend(ppplot.__all__)
__all__.extend(qqplot.__all__)
__all__.extend(scatterchart.__all__)
