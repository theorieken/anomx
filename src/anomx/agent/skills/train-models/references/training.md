# Training, logging and model publication

## Training output versus publication

Use the reusable APIs described in [scientific APIs](architectures.md). A
successful training result includes the requested architecture, frozen inputs,
resolved dataset selection/version, evaluation and replay artifacts. Persist
checkpoints, transforms, predictions and `procedure.json` with `work.artifact`.
The host accepts individual artifacts up to 64 MiB; split large result tables
into meaningful bounded parts rather than putting them in the Run result JSON.
A raw checkpoint is a trained artifact, not a registered portable Model. ONNX
publication is optional unless the user requested it or an inference workflow
requires it; export limitations must not trigger replacement of the requested
architecture or an endless retraining/install loop.

## Two launch paths

Prefer a shared Job workspace for editable/reusable training. When the task
specifically needs an individually configured immutable Run, create `/runs` with `name`,
`project`, optional draft/source `model`, `code` and `parameters`. Run with
`POST /runs/<id>/run`, choosing `compute_integration` or
`compute_service` and `timeout_seconds`. This executes the immutable Run once. It does not create a job component.
Never run the same Run twice or patch its procedure after creation.

For periodic or generic Work-driven training, set
`configuration={"training":true,"inputs":{...}}`. Optionally provide
`configuration.model_id` as the UUID of the selected internal model. Each component execution creates a new Run in the job's linked Project.
Do not put a previous run ID in recurring configuration. `parameters`/`inputs` should contain data references, mapping, architecture,
hyperparameters, seed and task definition rather than secrets or large data arrays.

Create an internal draft through `POST /models/models` if the user needs a named family
before first training. Otherwise publication can create it. Do not confuse imported
language models (e.g. DeepSeek, used by the agent) with internal forecasting models.

## Actual runtime and telemetry

Inspect package versions/device inside the selected compute process. Do not use
an agent-shell Torch import or local installation as evidence about a remote
worker. Persist the Run's actual service/integration and external scheduler ID
when present. A missing callback, failed dataset selection, missing dependency,
training exception and artifact upload failure are distinct stages; preserve the
failure and correct that stage before launching another Run.

The following per-epoch example applies to a loop that exposes both losses.
Native `TorchForecastingModel` records training loss in `training_history` after
fit; it does not provide validation callbacks or best-epoch selection. Log its
actual history and separately evaluate validation, or use a supported callback
adapter when live validation histories are required. Do not invent validation
losses or label a validation score as training loss.


```python
work.log("Starting training", dataset_version=version)
for epoch in range(1, epochs + 1):
    # Compute actual losses from this run.
    work.metric("train_loss", float(train_loss), step=epoch)
    work.metric("validation_loss", float(validation_loss), step=epoch)
    work.metric("epoch_seconds", float(elapsed), step=epoch)
    print(f"Epoch {epoch}: train={train_loss:.6f}, val={validation_loss:.6f}")
work.metric("test_rmse", float(test_rmse))
work.metric("persistence_rmse", float(baseline_rmse))
```

Metric names are nonempty, at most 128 characters; values must be finite numbers,
not booleans. Steps are nonnegative integers. Keep a consistent step interpretation
per metric. Scalar scores appear as numbers, repeated points as histories. Don't
emit fake zero metrics for missing evaluation. Plain stdout/stderr and `work.log`
are persisted in the Run activity trail; log tensors as small summaries, not
full datasets. Errors should fail the run with a useful message and preserved logs.

`GET /runs/<id>/telemetry` provides series, point counts, sampled flag,
latest log entries, resolved datasets, configuration identity and Run.
`GET /runs/<id>/audit-trail` is the shared Log/activity-sidebar source.
The Run UI's Overview shows live status/timing/latest numbers, Log shows the
full persisted activity, and Run shows curves, scalar metrics, code and provenance.

## Publish actual artifacts

Export a self-contained `.onnx`, with named inputs/outputs and a dynamic batch axis
where appropriate. Put required scaling in the model or publish an explicit,
inspectable preprocessing contract. Verify ONNX Runtime predictions against the
framework before publication; check shapes and a representative bounded batch.
Avoid external tensor data, network downloads in model loading, or arbitrary
pickled model objects. The Work publication and loading interfaces currently accept ONNX up to 64 MiB.
Larger architectures may require a smaller export or an explicit artifact-transport
extension; do not claim that a larger artifact was published successfully.

```python
published = work.publish_model(
    "forecast.onnx",
    name=inputs["model_name"],
    metrics={"test_rmse": test_rmse, "persistence_rmse": baseline_rmse},
    metadata={
        "framework": "pytorch",
        "task": "forecasting",
        "dataset_version": version,
        "parameters": dict(inputs),
        "normalization": normalization,
        "split": split_description,
        "packages": package_versions,
    },
)
result = {"model": published, "metrics": metrics, "preview": small_preview}
```

The platform stores and validates the artifact, computes SHA-256 and schemas,
records provenance and assigns model family/version. Publishing into a draft fills
v1; subsequent publications create new immutable versions. Keep a stable model
family/reference or name across retraining. Do not increment `version` yourself or
overwrite the previous file. Preserve the return value and inspect the resulting
Model; publication success alone does not prove the outer Work finished.

The model's training comparison groups by parameters, procedure hash and resolved
dataset versions. Completed runs contribute mean, standard deviation and count;
failed/running runs are not successful measurements. Different data/configurations
must remain separate. A single run has no empirical estimate of variability even
if its displayed standard deviation is zero.

Read the model at `GET /models/models/<id>` and its grouped training comparison
at `GET /models/models/<id>/training-summary`. `/models` is the legacy component
catalog, not the internal model registry.
