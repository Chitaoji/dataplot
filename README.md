# dataplot
Provides plotting tools useful in datascience.

A lightweight plotting library for data science that unifies data transformation and plotting in a single chainable API. `dataplot` is designed for fast exploration, teaching demos, and script-based analysis workflows.

## Installation
```sh
$ pip install dataplot
```

## Features
`dataplot` focuses on an **analysis-first plotting workflow**: data processing and visual diagnostics are written in one concise chain.

| Capability | What it gives you | Typical API |
| --- | --- | --- |
| 📦 Data-as-object API | Treat raw arrays/series as first-class plotting objects with metadata and settings. | `dp.data(...)` |
| 🔗 Composable transforms | Build reproducible feature pipelines before plotting. | `.rolling().demean().zscore().rank()` |
| 📊 Statistical chart set | Switch between distribution, trend, and diagnostic views quickly. | `.hist()`, `.plot()`, `.scatter()`, `.qqplot()` |
| 🧩 Artist-first composition | Compose multiple plots into one figure in a clean, deferred style. | `dp.figure(artist1, artist2, ...)` |
| 🎛️ Layered settings model | Configure defaults at dataset / axes / figure scope with consistent fallback. | `set_plot()`, `set_axes()`, `set_figure()` |
| 🐍 Scientific Python stack | Keep compatibility with familiar numeric and plotting ecosystems. | `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn` |

In short: **less boilerplate, clearer analysis flow, and more reusable plotting code**.

## Quick Start
```py
import dataplot as dp
import numpy as np

x = dp.data(np.random.randn(500))

artist1 = x.hist(bins=30)
artist2 = x.qqplot(baseline="normal")

fig = dp.figure(artist1, artist2, title="Distribution diagnostics")
fig
```

## Core Concepts
### `dp.data(...)`
`dp.data(...)` is the entry point of `dataplot`.
It converts array-like input (for example `list`, `numpy.ndarray`, `pandas.Series`, or another `PlotDataSet`) into a unified `PlotDataSet` object.

`PlotDataSet` is the core abstraction that carries:
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

### Typical Workflow
```text
Raw data -> dp.data(...) -> transform chain -> artist(s) -> dp.figure(...)
```

You can think of `dataplot` as a 4-step loop:
1. **Wrap** your data with `dp.data(...)`.
2. **Transform** it with chainable operations (`rolling`, `zscore`, `rank`, ...).
3. **Render** one or more `Artist` objects via plot methods.
4. **Compose** artists into a figure and apply final figure/axes settings.

### Data Operations
`PlotDataSet` supports both arithmetic operators and built-in transforms:
- **Arithmetic**: `+ - * / **`
- **Log / power family**: `log()` / `log10()` / `signedlog()` / `signedlog10()` / `pow()` / `root()` / `sqrt()` / ...
- **Statistical transforms**: `rolling()` / `demean()` / `zscore()` / `rank(pct=True)` / `cumsum()` / `abs()`
- **State management**: `copy()` / `reset()` / `undo_all()` / `resample()`

Operations are chainable, which is useful for quick experimentation:

```py
y = x.rolling(5).demean().zscore().rank(pct=True)
```

### Plot Methods
Every plot method returns an `Artist` object instead of drawing immediately.
This enables deferred composition and clean multi-panel figure assembly:
- **Distribution**: `hist(...)`
- **Trend / relationship**: `plot(...)`, `scatter(...)`
- **Goodness-of-fit diagnostics**: `qqplot(...)`, `ppplot(...)`, `ksplot(...)`
- **Structure overview**: `corrmap(...)`

```py
a1 = x.hist(bins=40, alpha=0.7)
a2 = x.qqplot(baseline="normal")
fig = dp.figure(a1, a2, title="Distribution Check")
```

## Plot Settings
Common settings:
- `title`, `xlabel`, `ylabel`
- `alpha`, `grid`, `grid_alpha`
- `style`, `figsize`, `dpi`
- `fontdict`, `legend_loc`
- `subplots_adjust`
- `reference_lines` (for example: `"y=x"`, `"y=0"`, `"x=1"`)

## Requirements
```txt
validating
lazyr
loggings
matplotlib
numpy
pandas
scipy
seaborn
```

## See Also
### Github repository
* https://github.com/Chitaoji/dataplot/

### PyPI project
* https://pypi.org/project/dataplot/

## License
This project falls under the BSD 3-Clause License.

## History
### v0.1.9
* New method `PlotDataSet.rank(pct=True)` for rank transformation, supporting `pct=False` to return raw ranks.
* Updated scatter behavior with improved defaults and removed implicit x-axis sorting.
* Refactored plotting APIs to remove the `ax` constructor argument for a cleaner artist construction flow.

### v0.1.8
* Improved naming inference in `dp.data(...)` when decorators from `validating` are involved.
* Refined module exports around artist helpers to make wildcard-style artist imports behave consistently.
* Refactored plot-setting fallback behavior to use `dp.defaults` as the global source of defaults.
* Enhanced `get_setting(...)` (including overload/type-hint coverage) so default values align with the corresponding setting types, and dict defaults are handled more safely.
* Removed legacy `set_default()` usage and simplified setting application paths across artists/containers.
* Added dependency `loggings` and updated warning emission in figure setting flows.

### v0.1.7
* `dp.data(...)` can accept `PlotDataSet` objects now.
* `FigWrapper.__enter__()` now returns a copy safely via `_entered_copy`.
* Internal maintenance and stability refinements.

### v0.1.6
* New method `PlotDataSet.scatter()` to draw true scatter charts while keeping `PlotDataSet.plot()` as line chart behavior.
* Improved automatic label inference for `dp.data(...)`, plotting labels, and x-axis labels in interactive contexts.
* Plot builders now use deferred drawing; removed `dp.show()` and improved axis/figure rendering in object representations.
* Refined rendering stability with fixes for empty-axis cleanup and figure re-rendering in `FigWrapper.__repr__`.
* Updated minimum required Python version to >=3.13.

### v0.1.5
* Fixed issue: unworking figure settings in the artist methods.

### v0.1.4
* Fixed issue: incorrectly displayed histogram statistics when the x-label had been modified by the user.

### v0.1.3
* Allowed users to set the plot-settings by kwargs in artist methods like `PlotDataSet.hist()`, `PlotDataSet.plot()`, etc.
* New operation methods `PlotDataSet.signedpow()` and `PlotDataSet.log10()`.
* Renamed `PlotDataSet.signlog()` to `.signedlog()`; renamed `PlotDataSet.opclear()` to `.undo_all()`; removed `PlotDataSet.opclear_records_only()`.
* New optional parameter `format_label=` for `PlotDataSet.set_plot()` to decide whether to format the label when painting on the axes.
* When defining the data classes, used *dataclasses* instead of *attrs* for a faster import.

### v0.1.2
* New methods `PlotDataSet.corrmap()`, `PlotDataSet.ppplot()`, and `PlotDataSet.resample()`.
* New optional parameter `fmt=` for `PlotDataSet.plot()`, `PlotDataSet.qqplot()`, `PlotDataSet.ppplot()`, and `PlotDataSet.ksplot()`.
* Bugfix.

### v0.1.1
* New module-level function `dp.show()`.
* New methods `PlotDataSet.qqplot()`, `PlotDataSet.ksplot()` and `PlotDataSet.abs()`.
* All the plotting method (e.g., `.hist()`) will now return an `Artist` object instead of None.
* New plot settings: `grid` and `grid_alpha`.
* Parameters of `FigWrapper.set_figure()`, `AxesWrapper.set_axes()` and `PlotDataSet.set_plot()` are keyword-only now.
* The returns of `.set_figure()` and `.set_axes()` will be None (instead of `self`) to avoid misunderstandings.
* New optional parameter `inplace=` for `PlotDataSet.set_plot()` to decide whether the changes will happen in-place (which is the only option before) or in a new copy.
* Parameter `ticks=` for `PlotDataSet.plot()` can be set to a `PlotDataSet` object now.

### v0.1.0
* `PlotDataSet` now supports binary operations including +, -, *, /, and **.
* New methods `FigWrapper.set_figure()` and `AxesWrapper.set_axes()` - use them instead of `*.set_plot()`.
* Simplified the usage of `AxesWrapper`.
* New plot settings: `subplots_adjust=`, `fontdict=` and `dpi=`.
* After this version, the required Python version is updated to >=3.11.9. Download and install v0.0.2 if the user is under lower Python version (>=3.8.13).

### v0.0.2
* Updated the meta-data.

### v0.0.1
* Initial release.
