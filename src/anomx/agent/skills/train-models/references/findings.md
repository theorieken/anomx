# Turn predictions into findings

The goal of an operational system is a useful, inspectable finding. Keep stages
explicit: data selection → preprocessing → model inference → residual/score →
decision → finding. Training metrics are not automatically operational anomalies.

In Work Python, `work.finding(title, score=None, **evidence)` creates a persisted
finding through the host and returns its reference. `work.detection(...)` remains
a compatibility alias. On a standalone context without a host callback this only
collects a local event; verify the returned platform reference before claiming it
was saved. Findings start open and pending human feedback. Do not mark them as
confirmed anomalies or resolved unless the user explicitly requests that action.

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
    )
    work.log("Created forecast finding", finding=finding)
```

Use an independently calibrated or domain-defined threshold. Do not tune it on the
test event merely to force a finding. Include the actual model version/reference,
data version, event time or sequence position, observed/expected values, units,
threshold rule and a bounded evidence window. Explain that a deviation is a signal
for review, not proof of equipment failure. The host adds Job/WorkRun provenance.

Avoid one finding per waveform element. Group a contiguous abnormal episode or
emit a bounded highest-priority summary. For scheduled replay of fixed historical
data, identify the run as a replay and avoid representing an old event as current.
If no deviations exceed the rule, log that result; don't manufacture a finding.
An actual model-quality problem can instead be an honest `model_evaluation`
finding explaining worse-than-baseline performance.

Retrieve `GET /findings/<returned-id>` and check title, score, description,
`display_payload` provenance and review state. Open its object page when useful.
Use `investigate-findings` for follow-up evidence and user-requested feedback.
The finding's legacy `run` relation refers to a DAQ JobRun; compute WorkRun identity
is recorded in `display_payload.work_run_id`. Do not claim a null legacy relation
means the finding has no compute provenance.
