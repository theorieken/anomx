# Processing and metrics

Anomx owns its time-series processing and metric implementations. They operate
on `anomx.TimeSeries`; installing or importing the optional Darts integration is
unnecessary. The baseline derives from Unit8 Darts under Apache 2.0, with its
API descriptions, mathematical definitions and references retained in docstrings.
See `NOTICE` and `licenses/DARTS-APACHE-2.0.txt` for provenance.

## Two explicit data contracts

Use `Pipeline` with native time series. Fit transformations on the training
partition, then reuse their fitted state for validation and prediction:

```python
from anomx import TimeSeries
from anomx.data.processing import Pipeline
from anomx.data.processing.transformers import MissingValuesFiller, Scaler
from sklearn.preprocessing import StandardScaler

series = TimeSeries.from_values([1., 2., 3., 4., 5., 6.])
train, validation = series.split_after(0.67)
processing = Pipeline([MissingValuesFiller(), Scaler(StandardScaler())])
train_scaled = processing.fit_transform(train)
validation_scaled = processing.transform(validation)
restored = processing.inverse_transform(validation_scaled, partial=True)
```

The existing `ProcessingPipeline`, `Transform`, and `Resample` APIs continue to
operate on pandas DataFrames, including dataset batch iterators. Their explicit
frame contract is useful before constructing a regular time series. A transform
that needs neighboring batches must implement `transform_batches()`; applying
an independent rolling calculation to each page would lose boundary context.

## Transformers

`anomx.data.processing.transformers` exports:

| Area | Implementations |
| --- | --- |
| Custom transforms | `BaseDataTransformer`, `FittableDataTransformer`, `InvertibleDataTransformer`, `Mapper`, `InvertibleMapper` |
| Numeric preprocessing | `Scaler`, `BoxCox`, `Diff`, `MissingValuesFiller` |
| Time windows and frequencies | `WindowTransformer`, `MIDAS` |
| Static features | `StaticCovariatesTransformer` |
| Hierarchical reconciliation | `BottomUpReconciliator`, `TopDownReconciliator`, `MinTReconciliator` |

Transformers support multivariate and sampled series, sequences of series,
component selection, parallel application, fitted state and inverse operations
where applicable. Use `verbose=True` for tqdm progress and `n_jobs` to control
parallelism. The detailed docstrings document each transform's limits, including
positive input for Box-Cox, missing values, hierarchical constraints and MIDAS
frequency ratios.

## Time encoders

`anomx.data.processing.encoders` includes past and future cyclic calendar,
datetime-attribute, integer-position and callable-index encoders. A
`SequentialEncoder` composes them, optionally fits a transformation, and
constructs matching training and forecast indices:

```python
from anomx.data.processing.encoders import SequentialEncoder

encoder = SequentialEncoder(
    add_encoders={"cyclic": {"future": ["hour", "dayofweek"]}},
    input_chunk_length=24,
    output_chunk_length=6,
    takes_future_covariates=True,
)
# `target` is a regularly sampled TimeSeries with a DatetimeIndex.
# _, train_covariates = encoder.encode_train(target=target)
# _, forecast_covariates = encoder.encode_inference(n=6, target=target)
```

Encoded component names start with `anomx_enc_`. Estimator integration is
explicit: use the generated covariates with an estimator that supports them.

## Dynamic time warping

`anomx.data.processing.dtw.dtw()` aligns univariate or multivariate series using
exact or multigrid dynamic time warping. It returns a `DTWAlignment` with the
path, distance, mean distance, warped series and Matplotlib visualizations.
Supported search windows include `NoWindow`, `SakoeChiba`, `Itakura`, and
`CRWindow`; custom distance functions are supported.

## Metrics

`anomx.metrics` provides the complete ported baseline:

| Family | Functions |
| --- | --- |
| Errors | `err`, `merr`, `ae`, `mae`, `se`, `mse`, `rmse`, `sle`, `rmsle` |
| Scaled errors | `ase`, `mase`, `sse`, `msse`, `rmsse` |
| Percentage/range errors | `ape`, `mape`, `wmape`, `sape`, `smape`, `ope`, `arre`, `marre` |
| Fit and tolerance | `r2_score`, `coefficient_of_variation`, `autc` |
| Probabilistic/quantile | `crps`, `mcrps`, `ql`, `mql`, `qr` |
| Prediction intervals | `iw`, `miw`, `iws`, `miws`, `ic`, `mic`, `incs_qr`, `mincs_qr` |
| Classification | `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix` |
| Alignment | `dtw_metric` |

Metrics align timestamps, support multiple components and series, and expose
reduction controls. Probabilistic metrics accept sampled or named-quantile
predictions; scaled metrics require training observations through `insample`.
Undefined metrics retain documented NaN behavior rather than reporting
misleading finite scores. Each function's docstring specifies its formula,
required inputs, output shape and examples.

```python
from anomx.metrics import mae, rmse

# `actual` and `prediction` are native TimeSeries on overlapping timestamps.
# print({"mae": mae(actual, prediction), "rmse": rmse(actual, prediction)})
```

`anomx.metrics.plot_tolerance_curve(actual, prediction)` returns Matplotlib axes
showing prediction coverage at each tolerated error and the matching AUTC score.
It supports multiple components, sampled predictions, named quantile predictions
and multiple series. Matplotlib is imported only when plotting; install
`anomx[visualization]` to enable the plotting dependencies.

## Regression coverage

`tests/native_processing_metrics` adapts the upstream processing and metrics
regression suites to Anomx imports. Encoder tests construct the standalone
encoder directly. Reconciliation and MIDAS tests supply explicit forecast and
residual series, so these tests verify processing independently of an external
forecasting-model implementation. Tests exercise all transformer families,
inverse transformations, encoders, DTW windows/plots, metric formulas,
probabilistic and classification outputs, index alignment and reduction shapes.
