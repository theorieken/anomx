# Scientific APIs and architecture choice

## Native first, preserving the requested model

Use `anomx.TimeSeries` for regular scalar or multivariate series. It retains
indices/components; it is not a Darts TimeSeries. Native APIs do not require Darts:

| Task | Reusable API |
|---|---|
| Persistence, mean, drift | `anomx.models.NaiveSeasonal(K=1)`, `NaiveMean`, `NaiveDrift` |
| Lagged linear forecasting | `LinearRegressionModel(lags=24, ridge=0.01)` |
| Neural forecasting with a custom module | `TorchForecastingModel` with optional Torch |
| Estimator/callable/Torch reconstruction, representation, regression or classification | `GenericModel(..., family=...)` |
| Scaling | `anomx.data.processing.transformers.Scaler` fitted on training only |
| Forecast evaluation | `anomx.metrics.mae`, `rmse` on aligned native series |
| Signed or absolute residuals | `anomx.detection.DifferenceScorer`, `NormScorer` |
| Learned normal-window scoring | `KMeansScorer`, `WassersteinScorer`, `PyODScorer` |
| Score thresholds | `QuantileDetector`, `IQRDetector`, `TimeSeriesThresholdDetector` |

Native forecasters expose `fit(series)`, `predict(n, series=history)` and
`historical_forecasts(series, start=..., forecast_horizon=..., retrain=False)`.
Fit first on the training period. With `retrain=False`, rolling forecasts use
observed history available at each origin without changing fitted weights.
`last_points_only=True` evaluates the final point of each horizon, not every lead
time. Preserve that distinction in reports. Native baseline models require finite,
regular, deterministic series and matching component order/frequency.

Use `NormScorer(component_wise=True).score_from_prediction(actual, predicted)` for
absolute per-channel residuals. Trainable scorers need training-only fitting;
`window_agg=False` is causal trailing scoring, whereas `True` is offline and uses
later windows. Native `QuantileDetector` differs from the legacy table-oriented
`QuantileThresholdDetector`. Calibrate thresholds separately from final test data.
NLL scorers require prediction samples, not a single deterministic forecast.

`GenericModel` preserves observation axes; it does not automatically make time
windows. Supply an estimator or model matching the actual task. Array-valued
waveforms may need a declared reduction or tensor model; never flatten event and
intra-waveform axes together and call that a temporal sequence.

## A real LSTM through the native training API

`TorchForecastingModel` accepts a `torch.nn.Module` mapping
`(batch, input_chunk_length, channels)` to
`(batch, output_chunk_length, channels)`. The bundled `lstm-model.py` implements
that architecture with `torch.nn.LSTM` and a projection; the package handles
windowing, the optimizer/training loop and recursive forecasts. Configure seed,
chunk lengths, learning rate, batch size, epochs and a real available device.
Do not label an MLP, linear baseline or Transformer as the requested LSTM.

The native loop currently records `training_history` with `epoch` and `loss`.
It has no `val_series`, early-stopping or best-validation-epoch argument. Log that
history honestly after fitting; verbose progress is stdout, not a live metric
callback. For bounded experiments, fix epochs beforehand and select candidates on
validation. If best-epoch selection or covariates matter, use a supported optional
model/adapter or a justified custom loop and explain why. Do not invent kwargs.

Native forecasting `save`/`load` produce pickle-based trusted replay artifacts.
Keep the source module and transform state alongside them. Never load untrusted
pickle files and do not send them to `work.publish_model`, which requires ONNX.

## Optional Darts architectures

`anomx.integrations.darts.models` contains adapters such as `RNNModel`,
`BlockRNNModel`, `TiDEModel`, `TFTModel` and `TransformerModel`. Some names are also
lazily exported by `anomx.models`; that does not make their backend native.
`anomx.models.PyTorchModel` is the older Darts adapter; prefer
`TorchForecastingModel` for native series. Never interchange the two containers
or assume their fit/covariate/export signatures match.

For an LSTM through Darts, inspect the installed `RNNModel(model="LSTM", ...)` or
`BlockRNNModel(model="LSTM", ...)` contract and explicit native-to-Darts conversion
by constructing `darts.TimeSeries.from_dataframe(native_series.to_dataframe())`. Use `val_series` and callbacks only where
that installed API supports them. TiDE/TFT have additional covariate and export
contracts. Future covariates must be available at inference time; future target
values are not covariates. Read the documentation matching the selected runtime,
not every available architecture's manual.

Dependencies belong to the compute runtime. The package offers optional `torch`,
`darts` and `ml` extras; use the managed runtime's supported profile rather than
installing into the agent sandbox. On current Maxwell, profiles are `base`, `ml`
and `darts` (no `torch` profile). `configuration.training=true` defaults to `ml`;
explicit `runtime_profile="base"` can serve native non-neural work and `"darts"`
requests the Darts profile. Verify actual installed versions in Run provenance.

## Custom architectures and publication

Define custom model code only for a requirement the reusable APIs do not cover.
Keep shape, covariate availability, loss, optimization and checkpoint selection
inspectable. Use actual train/validation losses and finite gradient/output checks.
The existing `forecast.py` is such a custom MLP/Transformer example, including
ONNX export. It is not a general-purpose data adapter or an LSTM template.

Portable publication needs a self-contained, supported ONNX graph with parity
against the selected checkpoint. Read the actual named input schema; do not
assume one `history` tensor or a particular rank. Optional model export support is
architecture/version-specific. Unsupported ONNX does not invalidate a completed
research run with trusted checkpoints, transforms, predictions and metrics.
