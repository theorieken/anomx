# Native time series

`anomx.TimeSeries` is Anomx's own numeric container for regular time series.
The full initial implementation is adapted from Unit8 Darts under Apache-2.0;
it lives in this repository and does not import or wrap the installed Darts
package. [Third-party notices](../THIRD_PARTY_NOTICES.md) record the exact
source revision, attribution, license, and modifications.

```python
import numpy as np
import pandas as pd
from anomx import TimeSeries, option_context

times = pd.date_range("2026-10-01", periods=120, freq="min")
series = TimeSeries.from_times_and_values(
    times,
    np.sin(np.arange(120) / 10),
    columns=["detuning"],
    metadata={"source": "local demonstration"},
)
train, test = series.split_before(0.8)

with option_context("plotting.use_anomx_style", True):
    ax = train.plot(label="training")
    test.plot(ax=ax, label="held out")
    ax.figure.savefig("detuning-split.png", bbox_inches="tight")
```

## Data model

Values have shape `(time, component, sample)`. A univariate series has one
component; a deterministic series has one sample. Multiple samples represent
probabilistic trajectories. `values()` returns a deterministic two-dimensional
view, and `all_values()` retains all three dimensions. These methods return
copies by default; `copy=False` exposes the underlying array deliberately.

The time axis is a monotonic, regular `pandas.DatetimeIndex` or `RangeIndex`.
Data must be numeric and component names unique. Missing timestamps can be
inserted with `fill_missing_dates=True` and an explicit `freq`. This inserts
missing values; interpolation and missing-value policies are separate
processing choices. Irregular observations belong in a Dataset or observation
view until a suitable regular grid has been chosen.

Native series store timezone-free timestamps. When converting timezone-aware
observations, normalize to UTC before removing the timezone and retain the
source timezone in metadata. This avoids introducing ambiguous local times
across daylight-saving transitions.

## Supported surface

The initial baseline includes the complete TimeSeries public surface from the
pinned source revision:

- Construction from arrays, pandas Series/DataFrames, grouped DataFrames,
  CSV, JSON, pickle, and xarray, plus Narwhals-compatible data backends.
- Conversion to arrays, Series, DataFrames, grouped DataFrames, CSV, JSON,
  pickle, and xarray.
- Positional and timestamp indexing; time/component selection; train/test
  splits; intersection; append, prepend, concatenate, and stack; shifts,
  resampling, gaps, differences, and window operations.
- Arithmetic, comparisons, mapping, aggregate statistics, probabilistic
  quantiles, and sample-wise statistics.
- Static covariates, component hierarchies, metadata, and explicit replacement
  methods that preserve series structure.
- Calendar/holiday attributes, Matplotlib plots, Plotly figures, plain-text
  display, and rich notebook display.

`concatenate()`, `slice_intersect()`, and `to_group_dataframe()` are exported
alongside `TimeSeries` from both `anomx` and `anomx.data`. Detailed parameter,
return-value, and example documentation lives in the source docstrings and can
be used by an API documentation generator.

## Processing and metrics

Native processing and metrics consume the same `TimeSeries` type. Fit
transformers on training observations and apply the fitted transformation to
held-out observations to keep evaluation meaningful.

```python
from anomx.data.processing.transformers import Scaler
from anomx.metrics import rmse

scaler = Scaler()
scaled_train = scaler.fit_transform(train)
scaled_test = scaler.transform(test)
restored = scaler.inverse_transform(scaled_test)
assert rmse(test, restored) < 1e-10
```

`Dataset` owns retrieval, batching, caching, source references, and tabular
materialization. `TimeSeries` owns the materialized regular numeric series.
Dataset conversion chooses a time grid explicitly when the source is
irregular. Generic sequences and unordered samples remain represented by the
existing observation-set and view contracts; this baseline does not impose a
regular time axis on those future data families.

## Optional backends and display

Matplotlib provides static plots; Plotly is optional for interactive figures.
Xarray and Polars are optional interoperability backends. The core container
does not require Darts or PyTorch. Global plotting defaults are unchanged until
`set_option("plotting.use_anomx_style", True)` or the scoped `option_context()`
is used. `display.max_rows` and `display.max_cols` control textual and notebook
output; `describe_option("all")` lists the available settings.

## Validation and maintenance

`tests/timeseries_baseline/` adapts the upstream TimeSeries regression suites
against native Anomx imports. It exercises deterministic and stochastic data,
integer and datetime indices, multivariate operations, metadata and hierarchy,
serialization, plotting, and display configuration. Additional native tests
verify operation when importing Darts is explicitly blocked. Regression tests
and attribution should be retained when this implementation evolves.
