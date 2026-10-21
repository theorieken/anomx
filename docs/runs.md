# Runs and step logging

A **Run** is one execution in a **Project**. It can represent training, a data
processing script, an experiment, or another procedure. The Python API works
without Django or an Anomx server. Numeric values become metrics on the platform;
all JSON values remain available in the event history.

## Offline first

```python
import anomx
from anomx.execution import current_run

with anomx.init(project="forecast-experiments", name="baseline", config={"window": 60}):
    for epoch in range(10):
        anomx.log({"loss": 1 / (epoch + 1), "epoch": epoch})
    anomx.log("evaluation", {"split": "held-out", "passed": True})
    print(current_run().summary)
```

By default each run writes a portable JSONL file under `.anomx/runs/`. Specify
`directory=` to change that location. Context-manager exit records success, or
failure with the exception message. `anomx.finish()` supports scripts without a
context manager. Logs may contain sensitive application values, so choose values
and the output directory deliberately.

## Steps

Each `log()` commits one step, starting at zero. Log a dictionary when several
metrics belong to the same observation. `commit=False` accumulates values until
the next committed call. An explicit `step=` may skip forward, but never backward:

```python
anomx.log("train/loss", 0.3, step=20, commit=False)
anomx.log("validation/loss", 0.4, step=20)
```

Commit pending values before changing the step. `finish()` flushes pending
values. Keys must be nonempty strings of at most 128 characters; values must be
JSON serializable and numbers finite. A run holds an in-memory latest-value
summary as well as its durable event history.

## Connected platform

```python
import anomx
from anomx.integrations.platform.tracking import PlatformBackend

# ANOMX_URL=https://your-instance/api/v1
# ANOMX_TOKEN=<your scoped token>
with anomx.init(project="forecast-experiments", backend=PlatformBackend()):
    anomx.log("loss", 0.15)
```

`PlatformBackend` accepts an explicit `PlatformClient` too. Project may be a
project name or a `runs_project-<uuid>` reference. A name creates or reuses a
project owned by the authenticated user. Creating and writing executions require
ownership; sharing a project gives its audience read access to its runs.

Events carry stable UUIDs. After a transport failure, `run.flush()` retries the
same event; the platform deduplicates it. The next `log()` also retries pending
delivery before accepting new values. Delivery is synchronous and errors are
reported to the caller. This is not a background upload service or a crash-safe
persistent retry queue. Offline JSONL files are not uploaded automatically.

## Inside an Anomx job

Every job component execution has a Run in the job's automatically linked Project.
The worker binds it before executing Python, so code can simply call:

```python
import anomx

anomx.log({"rows_processed": 2500, "quality": 0.98})
```

`anomx.init()` without a project reuses that hosted run. The worker controls its
start, completion, cancellation, and failure; `anomx.finish()` does not end the
worker. Omitted steps are allocated atomically by the platform, including across
worker processes. Explicit steps must advance the shared history. Prefer one
mapping per observation when multiple values need the same step.

`CallbackBackend` and the `RunBackend` protocol allow other execution systems to
reuse this API without importing platform code. `use_run(run)` binds a run to the
current context without controlling its lifecycle. Context bindings follow async
contexts; callers starting independent threads must bind the run in those threads.

## Editing a multi-file job workspace

The IDE and agent edit the same saved workspace. Text files and explicit empty
folders form a relative path mapping, with a selected Python entrypoint. A run
freezes the complete workspace so later edits do not change past executions.

```sh
python -m anomx.integrations.workspaces mount jobs_job_component-<uuid> ./procedure
# Edit procedure/main.py and add modules or folders using ordinary file tools.
python -m anomx.integrations.workspaces save ./procedure
```

This materializes a checkout, not an operating-system filesystem mount. The
`.anomx-workspace.json` manifest stores the component, entrypoint, and server
revision; credentials remain in the environment. Save replaces the entire remote
workspace only if the revision still matches. HTTP 409 means another editor
saved first: retrieve the latest workspace in a new directory and merge before
retrying. Never discard that conflict by blindly changing the revision.

Workspaces support 256 entries and 2 MiB of UTF-8 text. Absolute paths, traversal,
binary files, and symlinks are rejected. `.git`, `.venv`, and `__pycache__` are
excluded on save. Stop an active component before editing executable files.
Saving files does not start the job. The built-in `edit-workspace` skill teaches
the Anomx agent this workflow.
