# Turn predictions into findings

The goal of an operational system is a useful, inspectable finding. Keep stages
explicit: data selection → preprocessing → model inference → residual/score →
decision → finding. Training metrics are not automatically operational anomalies.

In Work Python, `work.finding(title, score=None, **evidence)` creates a persisted
finding through the host and returns its reference. `work.detection(...)` remains
a compatibility alias. On a standalone context without a host callback this only
collects a local event; verify the returned platform reference before claiming it
was saved. Findings start open as unlabeled novelties, pending reviewer classification. Work `kind` metadata describes the detector family; it is not a canonical event kind. Assigned reviewers can agree on Anomaly/Expected without a label (`classified`), then agree on a label (`approved`). Autonomous agent accounts with Events Management permission can submit their own reviews; delegated agents cannot cast a human vote.

Use native `NormScorer`/`DifferenceScorer` for aligned residuals, and an appropriate
native detector for calibrated score decisions. Training-only learned scorers and
separate threshold calibration prevent test leakage; see the scientific API
reference. Saved research checkpoints require their trusted source/transform
contract and are not passed to `work.load_model`.

Read the published input/output schema before registered-model inference. `work.load_model(...)`
returns `ONNXModel`: its `predict()` convenience method expects a three-dimensional
`(batch, input_window, channels)` tensor. For other shapes use named tensors with
`model.run(...)`; do not add a dimension that contradicts the exported graph.
For `assets/forecast.py`, the names are `history` and `forecast`, and the input is
two-dimensional `(batch, lookback)`:

```python
model = work.load_model(model_reference)
forecast = np.asarray(model.run({"history": histories})["forecast"]).reshape(-1)
```

Check finite outputs, batch length and forecast target alignment. Before using
grouped episodes, exercise the actual finding code with a local known event at a
nonzero position. Keep global scored positions, source-row indices and positions
within an episode distinct when selecting the peak timestamp. Synthetic checks
must not publish fabricated operational findings.

```python
error = float(abs(observed - predicted))
if error > threshold:
    finding = work.finding(
        "Forecast deviation exceeds the validation threshold",
        score=error,
        kind="forecast_deviation",
        message=f"Observed {observed:.4g}; expected {predicted:.4g}; threshold {threshold:.4g}.",
        detected_at=event_time_iso,
        model_reference=model_reference,
        dataset_reference=dataset_reference,
        dataset_version=dataset_version,
        observed=float(observed),
        forecast=float(predicted),
        threshold=float(threshold),
        units=target_units,
        threshold_basis="99th percentile of validation absolute residuals",
        sources=[
            {"kind": "object", "reference": model_reference, "label": "Forecast model"},
            {"kind": "object", "reference": dataset_reference, "label": "Evaluation dataset"},
            {"kind": "text", "text": "Threshold calibrated on validation residuals, independently of this event."},
        ],
    )
    work.log("Created forecast finding", finding=finding)
```

Use an independently calibrated or domain-defined threshold. Do not tune it on the
test event merely to force a finding. Include the actual model version/reference,
data version, event time or sequence position, observed/expected values, units,
threshold rule and a bounded evidence window. Explain that a deviation is a signal
for review, not proof of equipment failure. The host adds Job/Run provenance.
Use structured `sources` for evidence in the Sources tab, with verified object
references, HTTP(S) citations, or concise text. A source reference does not attach
an affected channel: supply `channels` or `relationships` separately. Supply
timezone-aware `started_at` and `ended_at` for a bounded episode. Do not author
the server's attribution fields `role` or `recorded_at`.

Avoid one finding per waveform element. Group a contiguous abnormal episode or
emit a bounded highest-priority summary. For scheduled replay of fixed historical
data, identify the run as a replay and avoid representing an old event as current.
If no deviations exceed the rule, log that result; don't manufacture a finding.
An actual model-quality problem can instead be an honest `model_evaluation`
finding explaining worse-than-baseline performance.

Retrieve `GET /events/<returned-id>` (`/findings` remains compatible) and check
title, score, description, `sources`, affected connections, producer/run and review
state. Open its object page when useful.
Use `investigate-findings` for follow-up evidence and user-requested feedback.
The current `run` relation refers to a unified Run, accessible through `/runs`.
Older evidence may also store `display_payload.work_run_id`; inspect the actual
returned references. A null or inaccessible relation does not establish that
the finding has no compute provenance. When adding sources later, read the
existing list first: PATCH replaces editable sources rather than appending them.
