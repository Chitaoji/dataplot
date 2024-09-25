# dataplot
Provides plotting tools useful in datascience.

## Installation
```sh
$ pip install dataplot
```

## Requirements
```txt
attrs
hintwith
lazyr>=0.0.16
matplotlib
numpy
pandas
scipy
```

## See Also
### Github repository
* https://github.com/Chitaoji/dataplot/

### PyPI project
* https://pypi.org/project/dataplot/

## License
This project falls under the BSD 3-Clause License.

## History
### v0.1.3
* Allowed users to set the plot-settings by kwargs in artist methods like `PlotDataSet.hist()`, `PlotDataSet.plot()`, etc.
* New method `PlotDataSet.signedpow()`.
* Renamed `PlotDataSet.signlog()` to `.signedlog()`; renamed `PlotDataSet.opclear()` to `.undo_all()`; removed `PlotDataSet.opclear_records_only()`.
* When defining the data classes, used *dataclasses* instead of *attrs* for a faster import.

### v0.1.2
* New methods `PlotDataSet.corrmap()`, `PlotDataSet.ppplot()`, and `PlotDataSet.resample()`.
* New optional parameter `fmt=` for multiple methods including `PlotDataSet.plot()`, `PlotDataSet.qqplot()`, etc.
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
