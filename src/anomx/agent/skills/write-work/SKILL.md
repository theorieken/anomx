---
name: write-work
description: Write executable Anomx multi-file Python Work using the package's datasets, models, scorers, detectors and host callbacks. Use for custom compute, forecasting, embeddings or anomaly pipelines.
metadata:
    title: Write Anomx Work
    keywords: Python, package, code, Work, pipeline, dataset, scorer, detector, embedding
---

# Python Work and the Anomx package

Anomx's package supplies reusable data/modeling interfaces; the platform owns
jobs, schedules, storage, authorization and distributed execution. Do not import
Django/platform models in a portable compute block. Read `manage-jobs` when saving
or scheduling Work, and `train-models` when fitting/evaluating/publishing a model.

The execution namespace contains `work`, `inputs`, `context` and `result`.
Use `edit-workspace` to mount and edit the Job's shared files and a configuration's Python entrypoint; set `result` to bounded outputs.
`work.compute(code, inputs=inputs)` sends that code to the selected compute target.
Outer variables/imports do not cross the boundary. Put heavy imports and fitting
inside the compute string; the outer procedure runs on the task worker even when
the selected destination is Maxwell.

| Import | Use |
| --- | --- |
| `from anomx import Dataset` | Local DataFrame/values or a lazy versioned `data_dataset-...` reference |
| `from anomx import DCacheDataset, MountedDataset` | Versioned external selections read natively on compute with an explicitly mapped read-only mount root |
| `from anomx import TimeSeries` | Native regular timestamp/component-aware series |
| `from anomx.models import LinearRegressionModel, NaiveSeasonal` | Native forecasting and persistence baseline |
| `from anomx.models import TorchForecastingModel, GenericModel` | Reusable optional Torch training or estimator/callable adapters |
| `from anomx.models import ONNXModel` | Portable artifact inference |
| `anomx.models` | Forecasting, reconstruction and representation components |
| `anomx.detection.scorers`, `anomx.detection.detectors` | Native scores and threshold/event decisions |
| `anomx.data.processing` | Reusable transforms; fit on training data and reuse for evaluation/inference |
| `anomx.base`, `anomx.models.base`, `anomx.detection.base` | Typed capabilities, input/output signatures and component contracts |
| `anomx.integrations.darts` | Optional Darts adapters; requires the matching extra/runtime |
| `from anomx.execution.work import WorkContext` | Host callbacks, logging, metrics, findings and dispatch |
| `from anomx.execution import current_run` | The active project-owned execution; `anomx.log` writes step history |

Use the documented public APIs first; inspect one installed signature only when
an unfamiliar argument or concrete version mismatch requires it. Do not repeatedly
dump package source or all of OpenAPI to discover a known workflow. Heavy imports
and dependency checks belong to the selected compute runtime. A missing Torch
import in the agent shell does not establish that remote compute lacks Torch.

`Dataset(data=frame, kind="time_series")` is local; `Dataset(reference)` resolves
through active Work callbacks. Use `.select(start=..., end=...)` for a half-open
time range, optionally `.downsample("10min", aggregation="mean")` before transfer.
`to_dataframe(max_rows=...)` gives a bounded prefix and resumes cached pages;
`iter_batches()` streams them. Do not clear the cache during routine recovery.
`to_timeseries(value_columns=[...])` creates native TimeSeries and materializes
the selection, so bound it first. Irregular sampling needs an explicit policy;
no automatic fill is assumed. See `train-models` for full data and runtime contracts.

Native models expose `fit(train)` and `predict(n, series=history)`. Native
`NormScorer().score_from_prediction(actual, predicted)` aligns residuals; a
separate detector applies calibrated thresholds. `TorchForecastingModel` supplies
windowing and training for a custom module, including an LSTM. `GenericModel`
preserves observation axes for wider tasks. Prefer these over handwritten loops.
Optional `anomx.integrations.darts.models` uses Darts series; the older
`PyTorchModel` is a Darts adapter, not the native Torch forecasting trainer.

Keep shape, timestamp, units and modality explicit. Forecasting predicts an ordered
future target, reconstruction reproduces observations, and representation produces
embeddings. A model's output does not become an anomaly by itself: apply a matching
scorer and calibrated decision rule. For multivariate or array-valued inputs,
justify reductions and alignment; never infer time units from array length.

`work.log` and `print` persist progress; `work.metric(name, value, step=epoch)`
persists finite scalar telemetry. `work.publish_model(path, name, metrics, metadata)`
publishes verified ONNX; `work.load_model(reference)` loads a registered artifact;
`work.finding(title, score=..., **evidence)` records a finding. Standalone
WorkContext collects local events; dispatch/publication need host callbacks.
Verify saved run results and artifacts before saying a platform action succeeded.

For training and comparisons, follow the `train-models` skill's job-synthesis
reference. Keep acquisition, processing, feature choices, optimization and
evaluation in editable modules with explicit inputs, and persist their actual
configuration plus dataset versions. Use `work.artifact(path, name=...,
metadata=...)` for tables, curves, predictions and checkpoints, with individual
files within the host's published limit. Source data remain read-only; scratch
and outputs belong to the managed Work directory.
