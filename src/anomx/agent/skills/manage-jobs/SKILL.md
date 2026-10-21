---
name: manage-jobs
description: Configure, start, pause, stop and diagnose Anomx acquisition and analysis jobs using actual worker capabilities and run evidence.
metadata:
    title: Manage Jobs
    hidden: true
    system: true
---

# Manage jobs

A Job owns one shared Python workspace and two independent features: data acquisition and workspace execution. Its compute Work objects (`JobComponent`) are **configurations**, not separate repositories. Each configuration selects an entrypoint in the shared files, command-line arguments, timing, and an execution target. Many configurations may call the same `main.py` with different arguments. Acquisition retains its channel/recording settings. Every execution creates a Run in the job’s linked Project with a frozen workspace revision and configuration snapshot. Use `edit-workspace` for file changes and `train-models` for modeling. Do not confuse user jobs with `/system/jobs`, which are platform maintenance tasks.

Read [job contracts](references/jobs.md) and `GET /jobs/build-options` before creating or restructuring a job. This resolves real DAQ services, stores and protocol limits. Retrieve an existing job before patching it and preserve unrelated configuration and bindings. Choose actual capabilities and validate units, timing and channel shape before enabling recording or analysis.

Use `start`/`resume` under `/jobs/<id>/...` to activate schedules, and `pause`/`stop` to deactivate. `POST /job-works/<id>/run` executes a selected configuration once. `/jobs/<id>/run` is only a convenience when exactly one execution item is available. Creation accepts `run_now`, not `start_immediately`. Prepare inactive jobs with `is_active=false` and `run_now=false` when only setup is requested. A request to execute authorizes the appropriate action under the active runtime policy.

Retrieve the job after an action, then inspect its run, errors and timestamps. An accepted action is not evidence that live acquisition has produced samples. Verify bounded history/storage and correlate run IDs. Keep component/code versions, configuration snapshots and model artifacts in explanations. Use `inspect-platform` when available for worker/bus problems, and `investigate-findings` for anomaly evidence.

For recording, read `manage-acquisition`; for compute destinations and external
storage, read `manage-integrations`; for package APIs and custom procedures, read
`write-work`. Keep these responsibilities distinct while composing one job:
continuous DAQ, occasional training, and frequent inference can run at different
times. A compute Work interval is a frequency in Hz (`1 / seconds`), not the DAQ
sample rate. Inspect `schedule_mode`, `enabled`, `next_run_at` and actual run state.
Do not patch the whole `works` array just to append one procedure to a running job.
Use `/jobs/<id>/add-work` or `POST /job-works` with the parent job reference, then
retrieve the saved Work and verify the existing acquisition remains unchanged.


The job UI is one card, with no per-Work tabs. Its menu lists **Configurations**
when Workspace is enabled, followed by **Data acquisition** and **Workspace**
switches. Configurations open as dialogs; Add is last in the nested menu. The
fourth creation option, **Other**, opens an empty IDE. DAQ-only shows acquisition
settings and the channel state table; workspace-only shows Runs and IDE; combined
jobs show acquisition, Runs, and IDE with DAQ settings inside acquisition.

Use `PATCH /jobs/<id>/features` with `acquisition_enabled` and/or
`workspace_enabled`. Turning a feature off cancels its active work and excludes
its configurations from scheduling, while preserving files and settings. Turning
DAQ on creates an empty, disabled acquisition definition if none exists; select
channels and validate settings before enabling it. Never interpret toggling a
feature as authorization to start a run.

Create a blank workspace with `job_type="advanced"`, `workspace_enabled=true`,
`acquisition_enabled=false`, `is_active=false`, and `works=[]`. Add compute
configurations through `/jobs/<id>/add-work` with `kind="compute"`, `entrypoint`,
`configuration.arguments` (an ordered string array, exposed as `sys.argv[1:]`),
`schedule_mode`, `frequency_hz`, compute target and `timeout_seconds`. Files live
at `/jobs/<id>/workspace`; do not create one copy for each configuration. A
configuration added to an active job starts disabled. Datasets, training logic,
and resource-specific code belong in the shared workspace, not a job settings
form. Preserve legacy configuration fields unless deliberately changing them.
