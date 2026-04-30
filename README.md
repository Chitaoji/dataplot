# dataplot
Provides plotting tools useful in datascience.

A lightweight plotting library for data science that unifies data transformation and plotting in a single chainable API. `dataplot` is designed for fast exploration, teaching demos, and script-based analysis workflows.

## 🛠️ Installation
```sh
$ pip install dataplot
```

## 📦 Requirements
```txt
numpy, pandas, scipy, matplotlib, seaborn, validating, lazyr, loggings
```

## ✨ Features
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
It converts array-like input (for example `list`, `numpy.ndarray`, `pandas.Series`, or another `PlottableData`) into a unified `PlottableData` object.

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
- **Log / power family**: `log()` / `log10()` / `signedlog()` / `signedlog10()` / `pow()` / `root()` / `sqrt()` / ...
- **Statistical transforms**: `rolling()` / `demean()` / `zscore()` / `rank(pct=True)` / `cumsum()` / `abs()`
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

## 🕒 History
### v0.1.14
* Added `PlottableData.resample(...)` with rule-based resampling options for time-series style aggregation workflows.
* Added `copy=` support to data normalization/input handling so callers can control whether input values are copied.
* Optimized non-normal histogram fitting by adding a sampling path to improve performance on large datasets.
* Simplified histogram PDF fitting internals by removing legacy positive-parameter guard branches in `_fit_pdf`.

### v0.1.13
* Removed the legacy `label` alias from `Data`/`PlottableData` internals and completed the naming shift to `name` for a more consistent API.
* Renamed `normrank()` to `ranknorm()` for method-name consistency with existing rank-normalization terminology.
* Refined README presentation with updated section icons and clearer visual structure.

### v0.1.12
* Added rank-normalization support via `PlottableData.ranknorm(...)`, using normal-quantile mapping for percentile ranks.
* Optimized rank computation by vectorizing tie handling in `PlottableData.rank(...)`.
* Completed the naming migration from `PlotDataSet`/`PlottableDatas` to `PlottableData`/`PlottableDataSet` for clearer API consistency.
* Refactored `PlottableData` by moving its data-processing logic into a new parent class `Data`, keeping only plotting settings.

### v0.1.11
* Added arithmetic operator support to `MultiObject` (including reflected operators), enabling element-wise math workflows for grouped `PlottableData` objects with length checks.
* Improved numerical robustness in `utils.math` by consistently sanitizing `NaN`/`Inf` inputs and validating finite sample counts in 1D linear regression.
* Refined `PlottableData.plot()` / `PlottableData.scatter()` x-tick handling to accept broader array-like inputs and convert them consistently.
* Added and expanded unit tests for core/container behavior and utility modules.

### v0.1.10
* Renamed `dist_or_sample=` to `baseline=` in `PlottableData.qqplot()` for clearer baseline specification.
* Removed `edge_precision=` from `PlottableData.ppplot()` and `PlottableData.ksplot()`, and refined probability-range handling in the related diagnostic plotting flow.
* Improved QQ/PP plot readability by updating default axis labels and ensuring the rightmost x-axis tick label is preserved.

### v0.1.9
* New method `PlottableData.rank(pct=True)` for rank transformation, supporting `pct=False` to return raw ranks.
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
* `dp.data(...)` can accept `PlottableData` objects now.
* `FigWrapper.__enter__()` now returns a copy safely via `_entered_copy`.
* Internal maintenance and stability refinements.

### v0.1.6
* New method `PlottableData.scatter()` to draw true scatter charts while keeping `PlottableData.plot()` as line chart behavior.
* Improved automatic label inference for `dp.data(...)`, plotting labels, and x-axis labels in interactive contexts.
* Plot builders now use deferred drawing; removed `dp.show()` and improved axis/figure rendering in object representations.
* Refined rendering stability with fixes for empty-axis cleanup and figure re-rendering in `FigWrapper.__repr__`.
* Updated minimum required Python version to >=3.13.

### v0.1.5
* Fixed issue: unworking figure settings in the artist methods.

### v0.1.4
* Fixed issue: incorrectly displayed histogram statistics when the x-label had been modified by the user.

### v0.1.3
* Allowed users to set the plot-settings by kwargs in artist methods like `PlottableData.hist()`, `PlottableData.plot()`, etc.
* New operation methods `PlottableData.signedpow()` and `PlottableData.log10()`.
* Renamed `PlottableData.signlog()` to `.signedlog()`; renamed `PlottableData.opclear()` to `.undo_all()`; removed `PlottableData.opclear_records_only()`.
* New optional parameter `format_label=` for `PlottableData.set_plot()` to decide whether to format the label when painting on the axes.
* When defining the data classes, used *dataclasses* instead of *attrs* for a faster import.

### v0.1.2
* New methods `PlottableData.corrmap()`, `PlottableData.ppplot()`, and `PlottableData.resample()`.
* New optional parameter `fmt=` for `PlottableData.plot()`, `PlottableData.qqplot()`, `PlottableData.ppplot()`, and `PlottableData.ksplot()`.
* Bugfix.

### v0.1.1
* New module-level function `dp.show()`.
* New methods `PlottableData.qqplot()`, `PlottableData.ksplot()` and `PlottableData.abs()`.
* All the plotting method (e.g., `.hist()`) will now return an `Artist` object instead of None.
* New plot settings: `grid` and `grid_alpha`.
* Parameters of `FigWrapper.set_figure()`, `AxesWrapper.set_axes()` and `PlottableData.set_plot()` are keyword-only now.
* The returns of `.set_figure()` and `.set_axes()` will be None (instead of `self`) to avoid misunderstandings.
* New optional parameter `inplace=` for `PlottableData.set_plot()` to decide whether the changes will happen in-place (which is the only option before) or in a new copy.
* Parameter `ticks=` for `PlottableData.plot()` can be set to a `PlottableData` object now.

### v0.1.0
* `PlottableData` now supports binary operations including +, -, *, /, and **.
* New methods `FigWrapper.set_figure()` and `AxesWrapper.set_axes()` - use them instead of `*.set_plot()`.
* Simplified the usage of `AxesWrapper`.
* New plot settings: `subplots_adjust=`, `fontdict=` and `dpi=`.
* After this version, the required Python version is updated to >=3.11.9. Download and install v0.0.2 if the user is under lower Python version (>=3.8.13).

### v0.0.2
* Updated the meta-data.

### v0.0.1
* Initial release.