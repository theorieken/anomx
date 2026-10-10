# Jobs and Work API

These contracts describe current Anomx. Confirm the needed contracts once against
the connected schema; older server versions may differ. Recheck a specific contract
after a server-version change or concrete mismatch; reuse the current request's
recorded confirmation after compression. API paths are relative to the API base,
without another `/api` prefix. Use canonical returned object references for relations.

## Create and edit

`POST /jobs` accepts `name`, `description`, `job_type`, `is_active`, `run_now` and
`works`. Current purposes include `data_acquisition`, `model_training` and
`advanced`. A single-purpose job needs exactly one matching Work in `works`.
Use an inactive job and `run_now=false` while preparing a procedure. An advanced
job can start with no Work. Standalone `anomaly_detection` creation is rejected;
use an advanced job with explicit Work for custom detection.

For a Python training job, replace the placeholders with actual references/code:

```json
{
  "name": "Forecast experiment",
  "job_type": "model_training",
  "is_active": false,
  "run_now": false,
  "works": [{
    "name": "Train forecast",
    "kind": "compute",
    "mode": "python",
    "python_code": "result = work.compute(TRAINING_CODE, inputs=inputs)",
    "compute_integration": "integrations_integration-<resolved-id>",
    "compute_service": null,
    "enabled": true,
    "schedule_mode": "once",
    "frequency_hz": 0.0033333333333333335,
    "timeout_seconds": 900,
    "configuration": {"training": true, "inputs": {}}
  }]
}
```

`TRAINING_CODE` must be defined as a real Python string in the saved procedure,
not left as the literal placeholder above. A JSON nested `works` array replaces
definitions; retain unrelated items or prefer individual updates.

- `GET/PATCH /job-works/<id>`: edit the saved procedure, inputs and schedule.
- `POST /job-works/<id>/run`: manual execution; response is a WorkRun (202).
- `POST /jobs/<id>/run`: manual single-purpose execution. **This endpoint exists.**
- `POST /jobs/<id>/start` or `/resume`: activate scheduled Work.
- `POST /jobs/<id>/pause` or `/stop`: deactivate; may cancel active work.
- `POST /jobs/<id>/add-work`: append a Work; converts a single-purpose job to
  advanced, preserving existing Work. On an active job the new Work starts disabled.
- `PATCH /jobs/<id>/work-settings`: settings-only update; cannot replace code.
- `GET /jobs/<id>/work-runs`, `/work-runs/<id>`, `/jobs/<id>/executions`: evidence.

Only the owner can define or launch executable Work. Running Work rejects changes
to executable configuration; changing its name, description or `enabled` is allowed.
Do not mutate an executed Training; create another Training or let recurring Work
create a fresh one. Never set read-only status, version or run fields yourself.

## The task/compute boundary

The main Python procedure runs on the task worker. Selecting Maxwell on a Work
does not move every statement there. Dispatch heavy code explicitly:

```python
training_code = '''
import torch
from anomx import Dataset
data = Dataset(inputs["dataset_reference"])
work.log("Reading the versioned training dataset")
# Actual preparation/training/export lives here.
result = {"dataset_version": data.describe()["version"]}
'''
result = work.compute(training_code, inputs=inputs)
```

The compute namespace contains `work`, `inputs`, `context`, `result`.
Set `result` explicitly. Remote code has its own namespace; outer imports and
variables do not cross automatically. Pass data through `inputs` or predecessor
results. NumPy arrays and DataFrames use the Work value codec; do not wrap them
in arbitrary pickle blobs. Prefer dataset/model references over transporting huge
arrays. Keep returned previews bounded and put artifacts in managed storage.

`work.compute(code, inputs=..., target=...)` uses the selected target when `target`
is omitted. Inspect actual target identity and capabilities; do not invent a GPU.
Maxwell uses a managed environment and a workspace beneath
`<user-home>/anomx/jobs/<job-uuid>/runs/<run-uuid>/`. Relative files are local to that
execution. Never write to the source dCache folder as a training output location.

## Graph representation

`mode=graph` stores `graph={nodes, edges, output_node}`. Use one start and one end;
all nodes must lie on a path from start to end, with no cycles. The package
`anomx.work.WorkGraph` validates this structure. Executable types are `python`
and `inference`; start/end are structural. A Python node has `config.code` and
`config.runtime`: `task` for coordination, `default` for the selected compute.
Other runtime values must be real authorized targets supported by the host.

```json
{
  "nodes": [
    {"id":"start","type":"start"},
    {"id":"prepare","type":"python","config":{"runtime":"task","code":"result = dict(inputs)"}},
    {"id":"train","type":"python","config":{"runtime":"default","code":"result = inputs['previous']"}},
    {"id":"end","type":"end"}
  ],
  "edges": [
    {"source":"start","target":"prepare"},
    {"source":"prepare","target":"train"},
    {"source":"train","target":"end"}
  ],
  "output_node":"train"
}
```

Replace the example train node with actual code. Each step receives original
inputs plus `previous` (one predecessor's result, or a mapping for a join) and
`results` (completed results by node ID). Steps run in deterministic topological
order; a graph does not promise parallel execution. Inference can load an artifact
with `work.load_model(reference)` and use its real input schema.

DAQ Work uses `kind=data_acquisition`, actual channels/DAQ service/store, validated
sampling frequency and retention/duration. Its frequency is acquisition-specific;
do not confuse `sampling_frequency_hz` with a compute Work's scheduling frequency.
