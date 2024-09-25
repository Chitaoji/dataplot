"""
# dataplot
Provides plotting tools useful in datascience.

## See Also
### Github repository
* https://github.com/Chitaoji/dataplot/

### PyPI project
* https://pypi.org/project/dataplot/

## License
This project falls under the BSD 3-Clause License.

"""

import lazyr

VERBOSE = 0

lazyr.register("matplotlib.pyplot", verbose=VERBOSE)
lazyr.register("numpy", verbose=VERBOSE)
lazyr.register("pandas", verbose=VERBOSE)
lazyr.register("scipy.stats", verbose=VERBOSE)
lazyr.register("seaborn", verbose=VERBOSE)

# pylint: disable=wrong-import-position
from . import container, core, dataset, plotter
from .__version__ import __version__
from .container import *
from .core import *
from .dataset import *
from .plotter import *

__all__: list[str] = []
__all__.extend(core.__all__)
__all__.extend(plotter.__all__)
__all__.extend(container.__all__)
__all__.extend(dataset.__all__)
