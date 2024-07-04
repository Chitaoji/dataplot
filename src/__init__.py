"""
# dataplot
Provides plotters useful in datascience.

## See Also
### Github repository
* https://github.com/Chitaoji/dataplot/

### PyPI project
* https://pypi.org/project/dataplot/

## License
This project falls under the BSD 3-Clause License.

"""

from typing import List

import lazyr

VERBOSE = 0

lazyr.register("numpy", verbose=VERBOSE)
lazyr.register("pandas", verbose=VERBOSE)
lazyr.register("scipy.stats", verbose=VERBOSE)
lazyr.register("matplotlib.pyplot", verbose=VERBOSE)

# pylint: disable=wrong-import-position
from . import artist, container, core, dataset, histogram, linechart
from .__version__ import __version__
from .artist import *
from .container import *
from .core import *
from .dataset import *
from .histogram import *
from .linechart import *

__all__: List[str] = []
__all__.extend(core.__all__)
__all__.extend(artist.__all__)
__all__.extend(container.__all__)
__all__.extend(dataset.__all__)
__all__.extend(histogram.__all__)
__all__.extend(linechart.__all__)
