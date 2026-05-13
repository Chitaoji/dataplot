"""
# dataplot
Provides plotting tools useful in datascience.

A lightweight plotting library for data science that unifies data transformation and
plotting in a single chainable API. `dataplot` is designed for fast exploration,
teaching demos, and script-based analysis workflows.

## ✨ Features
`dataplot` focuses on an **analysis-first plotting workflow**: data processing and
visual diagnostics are written in one concise chain.

| Capability | What it gives you | Typical API |
| --- | --- | --- |
| 📦 Data-as-object API | Treat raw arrays/series as first-class plotting objects with
metadata and settings. | `dp.data(...)` |
| 🔗 Composable transforms | Build reproducible feature pipelines before plotting. |
`.rolling().demean().zscore().rank()` |
| 📊 Statistical chart set | Switch between distribution, trend, and diagnostic views
quickly. | `.hist()`, `.plot()`, `.scatter()`, `.qqplot()` |
| 🧩 Artist-first composition | Compose multiple plots into one figure in a clean,
deferred style. | `dp.figure(artist1, artist2, ...)` |
| 🎛️ Layered settings model | Configure defaults at dataset / axes / figure scope with
consistent fallback. | `set_plot()`, `set_axes()`, `set_figure()` |
| 🐍 Scientific Python stack | Keep compatibility with familiar numeric and plotting
ecosystems. | `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn` |

In short: **less boilerplate, clearer analysis flow, and more reusable plotting code**.

## 🚀 Quick Start
```py
import dataplot as dp
import numpy as np

raw = np.random.randn(300)
x = dp.data(raw, name="daily_return")

artist1 = x.hist(bins=30, alpha=0.7)
artist2 = x.qqplot(baseline="norm")

fig = dp.figure(artist1, artist2, title="Distribution diagnostics")
fig
```

## 🧠 Core Concepts
### 📥 `dp.data(...)`
`dp.data(...)` is the entry point of `dataplot`.
It converts array-like input (for example `list`, `numpy.ndarray`, `pandas.Series`, or
another `PlottableData`) into a unified `PlottableData` object.

`PlottableData` is the core abstraction that carries:
- **data values**
- **transformation history**
- **plot-level settings** (labels, style hints, etc.)

This design keeps analysis context attached to the data itself.

```py
import dataplot as dp
import numpy as np

raw = np.random.randn(300)
x = dp.data(raw, name="daily_return")
```

### 🔄 Typical Workflow
```text
Raw data -> dp.data(...) -> transform chain -> artist(s) -> dp.figure(...)
```

You can think of `dataplot` as a 4-step loop:
1. **Wrap** your data with `dp.data(...)`.
2. **Transform** it with chainable operations (`rolling`, `zscore`, `rank`, ...).
3. **Render** one or more `Artist` objects via plot methods.
4. **Compose** artists into a figure and apply final figure/axes settings.

### 🔢 Data Operations
`PlottableData` supports both arithmetic operators and built-in transforms:
- **Arithmetic**: `+ - * / **`
- **Log / power family**: `log()` / `log10()` / `signedlog()` / `signedlog10()` /
`pow()` / `root()` / `sqrt()` / ...
- **Statistical transforms**: `rolling()` / `demean()` / `zscore()` / `rank(pct=True)` /
`cumsum()` / `abs()`
- **State management**: `copy()` / `reset()` / `undo_all()` / `sample()`

Operations are chainable, which is useful for quick experimentation:

```py
x.zscore().rolling(5).rank(pct=True)
```

### 📈 Plot Methods
Every plot method returns an `Artist` object instead of drawing immediately.
This enables deferred composition and clean multi-panel figure assembly:
- **Distribution**: `hist(...)`
- **Trend / relationship**: `plot(...)`, `scatter(...)`
- **Goodness-of-fit diagnostics**: `qqplot(...)`, `ppplot(...)`, `ksplot(...)`
- **Structure overview**: `corrmap(...)`

```py
artist1 = x.hist(bins=30, alpha=0.7)
artist1  # show one plot
```
```py
artist2 = x.qqplot(baseline="norm")
fig = dp.figure(artist1, artist2, title="Distribution diagnostics")
fig  # show both plots
```

### ⚙️ Plot Settings
Common settings:
- `title`, `xlabel`, `ylabel`
- `alpha`, `grid`, `grid_alpha`
- `style`, `figsize`, `dpi`
- `fontdict`, `legend_loc`
- `subplots_adjust`
- `reference_lines` (for example: `"y=x"`, `"y=0"`, `"x=1"`)

## 🔗 See Also
### Github repository
* https://github.com/Chitaoji/dataplot/

### PyPI project
* https://pypi.org/project/dataplot/

## ⚖️ License
This project falls under the BSD 3-Clause License.

"""

import lazyr

with lazyr.setverbose(0):
    lazyr.register("pandas")
    lazyr.register("scipy.stats")
    lazyr.register("seaborn")

from . import container, core, database, plottable, setting
from .container import *
from .core import *
from .database import *
from .plottable import *
from .setting import *

__all__: list[str] = []
__all__.extend(core.__all__)
__all__.extend(database.__all__)
__all__.extend(setting.__all__)
__all__.extend(container.__all__)
__all__.extend(plottable.__all__)
