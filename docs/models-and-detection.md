# Native models and anomaly detection

Anomx's native models and scorers use `anomx.TimeSeries`. They do not import
Darts. Optional Darts architectures remain available from
`anomx.integrations.darts.models`; use that namespace when a Darts algorithm or
Darts series is intended. The older `PyTorchModel` is a Darts adapter, while
`TorchForecastingModel` is native and needs only the optional Torch extra.

## Forecasting

Native forecasters share `fit(series) -> self` and
`predict(n, series=None) -> TimeSeries`. Predictions start after the supplied
history, preserve its frequency and component order, and contain exactly `n`
observations. These baselines are deterministic and accept finite, regular
series; clean missing values and fit transformations on training data first.

| Class | Fitted behavior |
| --- | --- |
| `NaiveMean` | Per-component mean of the training observations |
| `NaiveSeasonal(K=1)` | Repeats the most recent `K` context observations |
| `NaiveDrift` | Extends the line between first and last context observations |
| `LinearRegressionModel(lags=12, ridge=0)` | Multivariate autoregression fitted by least squares, optionally ridge regularized |
| `TorchForecastingModel` | Fits any Torch module on chronological input/output windows |

```python
from anomx import TimeSeries
from anomx.models import LinearRegressionModel
from anomx.metrics import mae

series = TimeSeries.from_dataframe(frame)  # regular DatetimeIndex or RangeIndex
train, test = series.split_before(0.8)
model = LinearRegressionModel(lags=24, ridge=0.01).fit(train)
forecast = model.predict(len(test))
print(mae(test, forecast))
series.plot(label="observed")
forecast.plot(label="forecast")
```

`historical_forecasts` provides expanding-window evaluation. Its `start` is
an integer forecast-origin position or a fraction in `(0, 1)`. With
`retrain=True`, each model copy sees only observations preceding its forecast.
With `retrain=False`, fit the model first on a separate training period.
`last_points_only=False` returns each complete forecast; otherwise the final
point of each horizon is assembled into a time series. `backtest` evaluates
these forecasts with MAE by default or an explicitly supplied metric.

## Generic models and wider families

`GenericModel` wraps estimators, callables, and Torch modules. Its `family`
describes forecasting, predictive, reconstruction, representation, or
classification work. It preserves the observation axis rather than inventing
time windows. Forecasting windows belong to `TorchForecastingModel`; tensors
for other sequence structures can already pass through `GenericModel`, but
dedicated sampled/sequence containers and algorithms are future extensions.

For estimators, `fit` and `predict` are delegated. Representation estimators
can expose `transform` instead of `predict`. Reconstruction can be specified
with an explicit prediction function:

```python
from sklearn.decomposition import PCA
from anomx.models import GenericModel

reconstruction = GenericModel(
    PCA(n_components=1),
    family="reconstruction",
    predict_fn=lambda estimator, x: estimator.inverse_transform(estimator.transform(x)),
).fit(train)
reconstructed = reconstruction.predict(test)

embedding = GenericModel(PCA(n_components=1), family="representation").fit(train)
features = embedding.predict(test)
```

Arrays retain their observation axis. DataFrame results retain the index;
TimeSeries results retain timestamps. Representation and classification
outputs always receive new component names and drop physical-channel static
covariates/hierarchies, even when their width matches the input. Changing the
width in other families also creates explicit output names. Nonfinite inputs/outputs, changing input
tensor shapes after fitting, and dropped output observations raise errors.

For Torch, the module consumes `float32` tensors. The default optimizer is
Adam, with MSE loss. Reconstruction can omit the target; otherwise pass
targets explicitly. Classification expects integer class targets and uses
cross-entropy, returning logits from prediction. Custom `fit_fn`,
`predict_fn`, `loss_fn`, and `optimizer_factory` cover other contracts.

```python
import torch
from anomx.models import GenericModel, TorchForecastingModel

regressor = GenericModel(torch.nn.Linear(3, 1), epochs=50)
regressor.fit(x_train, y_train, verbose=True)
predictions = regressor.predict(x_test)

forecaster = TorchForecastingModel(
    input_chunk_length=24,
    output_chunk_length=1,
    n_epochs=20,
    verbose=True,
).fit(train)
forecast = forecaster.predict(len(test))
```

`training_history` holds epoch/loss records for inspection and plotting.
Torch forecasting starts each `fit` from the initial module template; repeated
fits and historical refits do not retain weights learned from later data.
User-supplied Torch forecasting modules must return
`(batch, output_chunk_length, components)` for an input of
`(batch, input_chunk_length, components)`. Input normalization is explicit.
`save`/`load` use pickle and are intended for trusted local artifacts.

## Scoring

Scoring and decisions are separate. A scorer produces continuous values;
a detector turns those values into binary anomaly labels. Native series
scorers preserve timestamps and support one series or a sequence of series.
Actual and predicted component names must agree. Timestamp intersection is
explicitly aligned before residual calculation.

| Scorer | Interpretation |
| --- | --- |
| `DifferenceScorer` | Signed `actual - prediction` per component |
| `NormScorer` | Absolute component residuals or a vector norm |
| `KMeansScorer` | Distance to the closest fitted normal-window cluster |
| `WassersteinScorer` | 1D distance from a window's empirical distribution to training observations |
| `PyODScorer` | Adapts a provided estimator's `fit` and `decision_function` |
| `GaussianNLLScorer`, `LaplaceNLLScorer`, `CauchyNLLScorer` | Negative log likelihood of observations under prediction-sample distributions |
| `ExponentialNLLScorer`, `GammaNLLScorer`, `PoissonNLLScorer` | NLL under the corresponding nonnegative/count distributions |

```python
from anomx.detection import NormScorer, QuantileDetector

scorer = NormScorer(component_wise=True)
calibration_scores = scorer.score_from_prediction(calibration, calibration_forecast)
test_scores = scorer.score_from_prediction(test, forecast)
detector = QuantileDetector(high_quantile=0.99).fit(calibration_scores)
labels = detector.detect(test_scores)
```

Calibration data must be separate from the final evaluation period. The
forecast for calibration should itself be generated without fitting on those
calibration targets. Detection never recalculates thresholds from test data.

Trainable window scorers offer `fit`, `score`, `fit_from_prediction`, and
`score_from_prediction`. Each training series is windowed separately, so no
window crosses independent series boundaries. The default `window_agg=False`
returns one score at each trailing window's end; the first `window-1` times
have no score. `window_agg=True` averages every window covering each point;
that setting is for offline analysis because future windows influence a
point's score. Choose component-wise fitting for separate channel models.

NLL scorers require deterministic observations and at least two prediction
samples per timestamp. They estimate distribution parameters independently
at each time/component; they do not fit a global training distribution.
`min_scale` prevents zero-scale estimates. Impossible observations receive a
finite `-log(tiny)` score cap; infinite densities are clipped to its negative.
A `window` greater than one returns trailing
means of pointwise NLL values. Gaussian/Laplace estimates use maximum-
likelihood moments; Cauchy uses robust median/half-IQR; Gamma uses method of
moments. These choices are documented and can evolve independently of Darts.

## Decisions and custom components

`TimeSeriesThresholdDetector(low_threshold=..., high_threshold=...)` applies
fixed bounds. `QuantileDetector` learns quantile bounds, and `IQRDetector`
learns Tukey fences. Thresholds can be scalar or component-wise. A value equal
to a threshold remains normal; values outside the bounds become `1`.

The existing `ThresholdDetector`, `QuantileThresholdDetector`,
`AbsoluteErrorScorer`, and `ZScoreScorer` remain table-oriented components for
the earlier record pipeline. Their names and existing behavior are retained;
the explicit TimeSeries names avoid silently changing those contracts.

```python
from anomx.detection import GenericScorer, GenericDetector

scorer = GenericScorer(lambda values: abs(values[:, 0]))
detector = GenericDetector(lambda values: values[:, 0] > 3.0)
scores = scorer.score(residuals)
labels = detector.detect(scores)
```

`GenericScorer` also accepts estimators implementing `score_samples`,
`decision_function`, or a per-observation `score`. Set
`higher_is_anomaly=False` for inlier-oriented estimator outputs.
`GenericDetector` accepts `detect` or `predict`, with `{0,1}` labels by
default; `anomaly_label=-1` handles sklearn's `{-1,1}` convention.
Both adapters accept explicit fit and inference callbacks. They reject
misaligned rows, nonfinite outputs, and unsupported label values instead of
silently guessing. A scalar DataFrame result adds `score` or `is_anomaly` to
the original table; array and TimeSeries results retain their own containers.
Torch scoring/detection modules are invoked in evaluation mode with gradient
tracking disabled, on their existing parameter/buffer device.

Generic adapters and `PyODScorer` require an explicitly supplied backend, so
they set `component_discoverable=False`: they are importable library tools
but are omitted from the platform's default algorithm catalog. Concrete
adapter subclasses can deliberately opt into discovery with their own
configuration and backend choice.
