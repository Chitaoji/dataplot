"""dataplot"""
import lazyr

VERBOSE = 0

lazyr.register("numpy", verbose=VERBOSE)
lazyr.register("pandas", verbose=VERBOSE)
lazyr.register("scipy.stats", verbose=VERBOSE)
lazyr.register("matplotlib.pyplot", verbose=VERBOSE)

# pylint: disable=wrong-import-position
from . import core, dataset, histogram, linechart, setter
from .__version__ import __version__
from .core import *
from .dataset import *
from .histogram import *
from .linechart import *
from .setter import *

__all__ = []
__all__.extend(core.__all__)
__all__.extend(dataset.__all__)
__all__.extend(histogram.__all__)
__all__.extend(linechart.__all__)
__all__.extend(setter.__all__)
