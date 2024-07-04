# dataplot
Provides plotters useful in datascience.

## Installation
```sh
$ pip install dataplot
```

## Requirements
```txt
attrs
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

### v0.1.0
* `PlotDataSet` now supports binary operations including +, -, *, /, and **.
* Added `FigWrapper.set_figure()` and `AxesWrapper.set_axes()` - now use them instead of `.set_plot()`. `PlotDataSet.set_plot()` remains however.
* Simplified the usage of `AxesWrapper`.
* New plot settings: `subplots_adjust`, `fontdict` and `dpi`.


### v0.0.2
* Updated the meta-data.

### v0.0.1
* Initial release.