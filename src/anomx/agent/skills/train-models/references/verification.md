# Evidence and recovery

Keep a compact evidence record with the actual references and values, not promises:

| Check | Evidence |
|---|---|
| Source | integration/file/channel, selected paths, mapping, read-only access |
| Dataset | kind/mode, version, observed rows/shapes, read coverage and limits |
| Procedure | Job/Work ID, saved Python/graph, inputs, compute and timeout |
| Runtime | WorkRun ID, trigger, timestamps, terminal status, Maxwell SLURM ID |
| Training | distinct Training ID, immutable code/config, train/val logs and points |
| Evaluation | held-out range/count, units, selected checkpoint, baseline comparison |
| Artifact | ready model, family/version, checksum, input/output schema |
| Inference | real input batch, returned output shape/finite values, framework agreement |
| Recurrence | two successful scheduled runs, observed interval, increasing versions |
| Final state | enabled/active flags and any next due time |

Use bounded polling with increasing delays while queued, and read only new/relevant
logs. Do not repeatedly reload whole datasets or training histories. If a poll says
completed, fetch final telemetry/artifact once. Mark stale data by its timestamp.
Use a cache-bypassing request for mutable status when a fresh read is required.

## Failure-specific next steps

- **401/403 or integration login action:** user reconnects through the integration
  form; never ask for credentials in chat or retrieve saved secrets.
- **404 API route:** inspect the schema once; changing search terms won't repair a
  nonexistent route. Object 404 may mean missing access or a stale reference.
- **Dataset source changed:** keep the failed fixed version; resolve/create a new
  version only within the user's intended data scope. Do not silently substitute.
- **HDF5 budget/shape failure:** narrow the selected arrays/files, inspect and read
  again. Do not copy the whole source or change permissions to evade a reader limit.
- **Duplicate timestamps:** verify event ordering; use sequence semantics when
  absolute timing is not established. Do not interpolate time without evidence.
- **Missing dependency:** identify the selected runtime and required optional
  dependency; use managed provisioning. A package in the task worker does not prove
  availability on Maxwell.
- **No metrics:** confirm the code reached the training loop, `work.metric` was
  called with finite values, and the callback belongs to this WorkRun. A process's
  stdout file alone is not the persisted training history.
- **Publication failed:** inspect the valid self-contained ONNX, file size, export
  opset and permissions. Keep logs and prior versions; never mark ready manually.
- **No second scheduled run:** read Work enabled/schedule/frequency/next due,
  Job active state, current queued/running executions, and scheduler health. Don't
  repeatedly call manual `/run` and claim recurrence.
- **Metrics worse than baseline:** report the actual result. Consider simpler
  models, more data, improved mapping or a revised task as a separate experiment.

Do not claim support for every possible format, model or runtime from one successful
example. The portable workflow is general; individual adapters, shapes, semantics,
dependencies and export paths still require validation.
