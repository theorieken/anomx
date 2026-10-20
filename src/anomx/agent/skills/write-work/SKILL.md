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
Use `edit-workspace` to mount and edit the component's files and Python entrypoint; set `result` to bounded outputs.
`work.compute(code, inputs=inputs)` sends that code to the selected compute target.
Outer variables/imports do not cross the boundary. Put heavy imports and fitting
inside the compute string; the outer procedure runs on the task worker even when
the selected destination is Maxwell.

| Import | Use |
| --- | --- |
| `from anomx import Dataset` | Local DataFrame/values or a lazy versioned `data_dataset-...` reference |
| `from anomx import DCacheDataset, MountedDataset` | Versioned external selections read natively on compute with an explicitly mapped read-only mount root |
| `from anomx.models import PyTorchModel, ONNXModel` | Custom tensor models and portable artifact inference |
| `anomx.components.models` | Forecasting, reconstruction and representation components |
| `anomx.scorers`, `anomx.detectors` | Scores and threshold/event decisions |
| `anomx.components.base` | Typed capabilities, input/output signatures and component contracts |
| `anomx.integrations.darts` | Optional Darts adapters; requires the matching extra/runtime |
| `from anomx.work import WorkContext` | Host callbacks, logging, metrics, findings and dispatch |
| `from anomx.runs import current_run` | The active project-owned execution; `anomx.log` writes step history |

Use installed public signatures (`inspect.signature` or relevant source) before
calling unfamiliar APIs; do not infer constructors from names. Heavy ML libraries
remain optional. `Dataset(data=frame, kind="time_series")` is a local dataset;
`Dataset("data_dataset-...")` resolves through the active Work host or configured
`PlatformClient`. It is not the Darts TimeSeries class. `describe()` establishes
schema/version; `iter_batches(batch_size=...)` bounds reads and pins that version.
`to_dataframe()` materializes the entire selection. In a connected agent shell,
the runtime supplies `ANOMX_URL`/`ANOMX_TOKEN`; never print them.

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
