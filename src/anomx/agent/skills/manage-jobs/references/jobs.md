# Current job contracts

`GET /jobs/build-options` supplies `daq_services`, `timeseries_stores`, `sample_stores`, and the legacy protocol sampling-frequency limit. `/jobs/component-options` and `/models`, `/algorithms`, `/scorers`, `/detectors`, `/components` expose further catalogs. Inspect component `config_schema`, defaults, capabilities, signature/parameters and code version instead of selecting by name alone.

Creation supports `job_type=data_acquisition`, `model_training`, `number_crunching`, and `advanced` (shown as **Other**). These are starting templates, not limits on the number of configurations. Jobs independently enable `acquisition_enabled` and `workspace_enabled`; use `/jobs/<id>/features` to change them after creation. A blank Other job needs no Work items and opens the shared IDE. Compute configurations select entrypoints in the one job workspace; DAQ definitions hold acquisition settings.

- `work_mode=continuous` requires no duration; `fixed_duration` and `sliding_window` require a positive `duration_seconds`.
- Sampling frequency must be positive and supported by the chosen protocol/service; do not request a frequency beyond the published legacy limit.
- DAQ jobs require at least one recordable channel and a DAQ service. Current channel validation supports single/array bool/float/int channels.
- Time-series and independent-sample channels cannot be mixed in one job. Select the appropriate time-series or sample store from build-options. Forecasting is limited to time-series data.
- Detection's current configuration is manual and requires a positive detection period. Inspect the current schema/validation for required training intervals, models, scorers and detectors; do not invent a config from future product plans.
- `record_data`, `detect_anomalies`, and `forecast_data` are separate concerns. Describe which outputs the user requested and the storage/compute consequences.

Related endpoints: `/job-channel-bindings` (live channel roles, sampling, transport, read parameters), `/job-data-bindings` (recordings/files),  Use canonical references and serializer-supported relation fields. Read existing bindings before updates; a replacement array must retain unrelated bindings.

Compute definitions and evidence use `/job-works`, `/runs`, `/runs/<id>/telemetry`, `/models/models/<id>/training-summary`, and `/findings`. All component executions use `/runs`; numeric history is available via `/run-metric-points` and artifacts via `/models/artifacts`. A job's current configuration may differ from an old run's snapshot. For a failure, correlate run timestamps/errors with the actual DAQ/compute service and storage identity.

Maintainer sources: platform `modules/jobs/models.py`, `serializers.py`, `viewsets.py`, worker runtimes and job utilities. Expand this reference as execution contracts evolve; preserve the runtime-verified distinctions above.

Use `edit-workspace` before changing code. GET/PUT `/jobs/<id>/workspace` reads or replaces the shared tree using `files`, default `entrypoint`, and current `revision`. Save conflicts return HTTP 409. Configuration entrypoints must remain present. PATCH `/job-works/<id>` changes a configuration's `entrypoint`, `configuration.arguments`, timing and execution target; ordinary PATCH does not edit files. The legacy `/job-works/<id>/workspace` route aliases the shared workspace and revision. Runs freeze all files and the chosen entrypoint/arguments; later edits do not alter an active run.

Existing different Work repositories are migrated into separate folders under the shared workspace; identical trees are reused. Keep their migrated entrypoints and `configuration.workspace_root` until deliberately reorganizing imports. `/jobs/<id>/work-runs` lists all job runs; `?work=<id>` filters one configuration. `/jobs/<id>/workspace-state` persists the selected acquisition/runs/ide view and editor/run state. Activity and diagnostics remain in the object activity drawer.

## Compute submission and evidence

Resolve `compute_integration` or `compute_service` from real capabilities; do not
invent service IDs, GPU availability or runtime profiles. The selected target is
used by `work.compute`, not by every statement in the task entrypoint. A shared
`main.py` can dispatch `from training import run; result = run(inputs, work)`;
the scientific module and its dependencies load on compute. Relative imports and
resources must be part of the frozen workspace. `configuration.inputs` supplies
the data mapping; `configuration.arguments` supplies ordered command-line strings.

`POST /job-works/<id>/run` returns the selected Run. Capture its ID and inspect
`/runs/<id>` and `/runs/<id>/telemetry` for actual runtime identity, frozen
workspace/configuration, resolved Dataset versions, metrics, logs and artifacts.
An accepted request is not proof of execution or successful model training. A
trained checkpoint does not imply ONNX publication, a passing quality gate, an
active schedule, or permission to affect a control system.

Dataset time selection, aggregation and paging should use the package's host
callbacks in Work. Do not bypass an isolated runtime with a direct HTTP client
or reimplement unbounded downloads. Use bounded selection and the Dataset cache;
record errors and resume interrupted pages before discarding saved progress.
If worker capabilities differ from the agent shell, trust verified worker
capabilities. Report missing dependencies against that target's managed runtime.

For uncertain launch outcomes, inspect bounded history for the matching
configuration/time before resubmitting. Preserve failed Runs. Correct shared
code with revision checks while the configuration is idle, then create a fresh
Run. Recurrence is proven by scheduled trigger evidence, not repeated manual
submissions. Reuse already confirmed contracts and saved references across
context compression; refresh mutable status instead of restarting discovery.
