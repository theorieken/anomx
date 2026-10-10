---
name: manage-jobs
description: Configure, start, pause, stop and diagnose Anomx acquisition and analysis jobs using actual worker capabilities and run evidence.
metadata:
    title: Manage Jobs
    hidden: true
    system: true
---

# Manage jobs

A Job owns independently configured Work items. Work can acquire data or execute custom Python/graphs. WorkRun records a compute execution; acquisition also has a JobRun. Read `train-models` for datasets, custom architectures, live metrics, publication, findings and recurring retraining. Do not confuse user jobs with `/system/jobs`, which are platform maintenance tasks.

Read [job contracts](references/jobs.md) and `GET /jobs/build-options` before creating or restructuring a job. This resolves real DAQ services, stores, component definitions and protocol limits. Retrieve an existing job before patching it and preserve unrelated configuration and bindings. Choose actual capabilities and validate units, timing and channel shape before enabling recording or analysis.

Use `start`/`resume` under `/jobs/<id>/...` to activate schedules, and `pause`/`stop` to deactivate. `POST /jobs/<id>/run` executes a single-purpose job once; advanced jobs use `POST /job-works/<id>/run`. Creation accepts `run_now`, not `start_immediately`. Prepare inactive jobs with `is_active=false` and `run_now=false` when only setup is requested. A request to execute authorizes the appropriate action under the active runtime policy.

Retrieve the job after an action, then inspect its run, errors and timestamps. An accepted action is not evidence that live acquisition has produced samples. Verify bounded history/storage and correlate run IDs. Keep component/code versions, configuration snapshots and model artifacts in explanations. Use `inspect-platform` when available for worker/bus problems, and `investigate-findings` for anomaly evidence.
