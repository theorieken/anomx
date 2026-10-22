# Jobs, shared workspace and remote execution

Confirm needed fields once against the connected schema and recorded server
version. API paths are relative to its base; do not add another `/api` prefix.
Use returned canonical references. `manage-jobs/references/jobs.md` is the shared
orchestration contract; this reference shows the training arrangement.

## One file tree, multiple configurations

A Job owns shared workspace files. A Work (`JobComponent`) selects one entrypoint,
arguments, inputs, target, timeout and timing. Several configurations can execute
the same `main.py` with different LSTM budgets or datasets. Do not create duplicate
file trees or jobs merely for parameter changes.

For a new empty training workspace, `POST /jobs` with actual name and:

```json
{
  "job_type": "advanced",
  "workspace_enabled": true,
  "acquisition_enabled": false,
  "is_active": false,
  "run_now": false,
  "works": []
}
```

Read `/jobs/<id>/workspace`, then use `edit-workspace` or its revision-checked
`PUT` with complete `files`, default `entrypoint` and current `revision`. Store
`main.py`, `training.py` and, for LSTM, `lstm_model.py` from the bundled assets.
The files contain the full editable implementation, not paths into the agent's
skill directory. Preserve unrelated files/configuration entrypoints. On HTTP 409,
read and reconcile the latest tree before saving; do not overwrite concurrent work.

Append a configuration with `/jobs/<id>/add-work` using actual values for:

```json
{
  "name": "LSTM forecast",
  "kind": "compute",
  "entrypoint": "main.py",
  "compute_integration": "<returned integration reference>",
  "compute_service": null,
  "enabled": true,
  "schedule_mode": "once",
  "timeout_seconds": 1800,
  "configuration": {
    "training": true,
    "runtime_profile": "ml",
    "arguments": [],
    "inputs": {
      "dataset_reference": "<returned dataset reference>",
      "target_columns": ["<verified target column>"],
      "start": "<verified UTC start>",
      "end": "<verified UTC end>",
      "architecture": "lstm",
      "lags": [24],
      "horizon": 1,
      "epochs": 10,
      "max_rows": 20000
    }
  }
}
```

These placeholders illustrate fields; resolve them before saving. `arguments` is
an ordered array of strings exposed as `sys.argv[1:]`; `inputs` is the Work data
mapping. They are separate mechanisms. Adding Work to an active Job starts that
configuration disabled; read it back before intended execution. Ordinary
`PATCH /job-works/<id>` changes configuration, not executable files. Nested `works`
replacement can remove unrelated definitions; append instead.

A single-template `model_training` job can also initialize files via its first
nested compute Work; `python_code` is a creation shortcut. Subsequent code edits
use the shared workspace endpoint. Job types are starting templates, not a limit
of one configuration per job. Use selected `/job-works/<id>/run` for a manual run;
`/jobs/<id>/run` is only unambiguous when there is one execution item. Preserve the
returned Run ID and inspect its detail/telemetry. `run_now` is the creation field;
`start_immediately` is not supported.

## The task/compute boundary

The entrypoint runs on the task worker even when a remote integration is selected.
Dispatch the scientific module explicitly:

```python
result = work.compute("from training import run\nresult = run(inputs, work)", inputs=inputs)
```

The frozen shared files are materialized in task and compute workspaces, enabling
module imports. Scientific imports and data loading belong inside `training.py`;
keep the entrypoint's module imports lightweight. Compute receives `work`, `inputs`,
`context`, and `result` in a separate namespace. Set `result`; outer variables and
imports do not transfer. `work.compute(..., target=None)` uses the selected target;
provide an explicit supported target only when the task needs another location.

Prefer Dataset/model references over large arrays. Return bounded summaries and
use `work.artifact` for predictions, checkpoints, transforms and provenance.
Relative outputs live in the managed execution directory. On Maxwell it is beneath
`<user-home>/anomx/jobs/<job-uuid>/runs/<run-uuid>/`; never put outputs in source
storage. Actual files/modules, argument order, configuration and workspace revision
are frozen per Run. Later edits do not alter a running execution.

The agent shell's Python environment, task-worker environment and selected compute
runtime are different. Test Torch and data access on the target, not by installing
into agent scratch space. Managed Maxwell profiles are `base`, `ml`, `darts`;
training defaults to `ml`. Check actual device/package versions in the Run. Do not
invent GPU capacity or an unsupported profile. One bounded dependency/data failure
is useful evidence; repeat only after a concrete correction, retaining failed Runs.

## Run and history evidence

`GET /jobs/<id>/work-runs?work=<id>&limit=25&offset=0` and
`/jobs/<id>/executions` return compact paginated history (`items`, `total`,
`has_more`, `next_offset`; maximum page size 100). Follow pagination only as needed.
Use `/runs/<id>` for frozen code/configuration/result and `/runs/<id>/telemetry`
for numeric histories/logs/resolved datasets. `/jobs/<id>/workspace-state` is UI
selection/editor state, never executable code or launch configuration.

Only the owner may define/launch executable Work. Running Work rejects executable
configuration edits; name, description and enabled can still be changed. Preserve
failed evidence and create another Run after correction. Never set status/version
fields yourself or rerun an immutable completed Run.
