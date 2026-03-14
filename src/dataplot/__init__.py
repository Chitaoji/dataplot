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

with lazyr.setverbose(0):
    lazyr.register("matplotlib.pyplot")
    lazyr.register("pandas")
    lazyr.register("scipy.stats")
    lazyr.register("seaborn")

from . import container, core, dataset, setting
from .container import *
from .core import *
from .dataset import *
from .setting import *

__all__: list[str] = []
__all__.extend(core.__all__)
__all__.extend(setting.__all__)
__all__.extend(container.__all__)
__all__.extend(dataset.__all__)
