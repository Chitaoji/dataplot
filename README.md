# dataplot
Provides plotting tools useful in datascience.

A lightweight plotting library for data science that unifies data transformation and plotting in a single chainable API. `dataplot` is designed for fast exploration, teaching demos, and script-based analysis workflows.

## Installation
```sh
$ pip install dataplot
```

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

## Features
- Data-as-object workflow with `dp.data(...)` and `PlotDataSet`.
- Chainable transforms such as `zscore()`, `log10()`, `rank()`, `rolling()`, and more.
- Statistical plotting methods including histogram, line/scatter, QQ/PP/KS plots, and correlation map.
- Unified plotting settings across figure / axes / dataset scopes.
- Built on top of `matplotlib`, `numpy`, `scipy`, and `seaborn`.

## Quick Start
```python
import numpy as np
import dataplot as dp

x = np.random.normal(loc=0, scale=1, size=500)
d = dp.data(x)

artist1 = d.zscore().hist(bins=30, color="C0")
artist2 = d.zscore().qqplot(dist="normal")

fig = dp.figure(
    artist1,
    artist2,
    ncols=2,
    title="Distribution diagnostics",
    style="seaborn-v0_8-whitegrid",
)

with fig as f:
    f.axes[0].set_axes(title="Histogram", xlabel="value", ylabel="count")
    f.axes[1].set_axes(title="QQ Plot", reference_lines=["y=x"])
```

## Core Concepts
### `dp.data(...)`
Wrap raw input into objects that support both math operations and plotting.

### Data Operations
`PlotDataSet` supports arithmetic operators (`+ - * / **`) and transforms such as:
- `log()` / `log10()` / `signedlog()` / `signedlog10()`
- `pow()` / `signedpow()` / `root()` / `sqrt()` / `signedroot()` / `signedsqrt()`
- `rolling()` / `demean()` / `zscore()` / `rank(pct=True)` / `cumsum()` / `abs()`
- `resample()` / `reset()` / `undo_all()` / `copy()`

### Plot Methods
Each method returns an `Artist` object that can be passed to `dp.figure(...)`:
- `hist(...)`
- `plot(...)`
- `scatter(...)`
- `qqplot(...)`
- `ppplot(...)`
- `ksplot(...)`
- `corrmap(...)`

## Plot Settings
Common settings:
- `title`, `xlabel`, `ylabel`
- `alpha`, `grid`, `grid_alpha`
- `style`, `figsize`, `dpi`
- `fontdict`, `legend_loc`
- `subplots_adjust`
- `reference_lines` (for example: `"y=x"`, `"y=0"`, `"x=1"`)


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
