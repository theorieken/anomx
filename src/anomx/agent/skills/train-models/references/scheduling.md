# Recurring training and custom Python

Work owns the schedule; the Job controls whether enabled Work may run.

| Intent | Work settings |
|---|---|
| Once | `schedule_mode=once` |
| Every x minutes | `schedule_mode=interval`, `frequency_hz=1/(60*x)` |
| Every five minutes | `frequency_hz=0.0033333333333333335` |
| Every ten minutes | `frequency_hz=0.0016666666666666668` |

Set `enabled=true`, a bounded `timeout_seconds`, and activate the Job with
`POST /jobs/<id>/start`. An inactive Job does not run recurring Work. A disabled
Work does not run on schedule. Manual `/run` is a different trigger and is not
evidence of periodic scheduling. The scheduler will not overlap queued/running
executions of the same Work. Intervals are measured from scheduling/start, not
guaranteed wall-clock cron times; resource queues can delay actual compute.

An API update of frequency, enabled state or schedule mode resets `next_run_at`.
Do not patch those repeatedly while observing a schedule, because that changes
what you are trying to verify. `next_run_at`, `last_started_at`, statuses and
external references are runtime-owned. Never simulate the scheduler by manually
inserting WorkRuns or by running an independent shell loop/cron.

## Two-cycle verification

1. Save a complete, inactive Job/Work. For recurring training use
   `configuration.training=true`, stable model identity and **no training_id**.
2. Run one manual execution if needed to validate expensive data loading or code.
   Fix failures while idle, preserving the failed run as evidence.
3. Enable the requested interval and activate the Job. Read back all settings.
4. Observe the first scheduled run through `/jobs/<id>/work-runs` or `/work-runs`.
   Check `trigger_type=schedule`, successful terminal state, completed Training,
   external compute reference and a ready artifact.
5. Wait for the next due interval without touching the schedule. Verify another
   distinct WorkRun with `trigger_type=schedule`, a new Training and a higher
   version in the same family. Compare start times and resolved dataset versions.
6. For a bounded test, `PATCH /job-works/<id>` with `enabled=false` prevents new
   schedules and does not cancel an already running execution. Let existing work
   finish. Then deactivate the Job if the user requested a stopped final state.
   Read back `enabled=false` and `is_active=false`; don't leave surprise compute.

If the user explicitly wants ongoing retraining, leave that requested schedule
enabled and report its interval, data window, compute and next due time. A fixed
dataset repeats the same data; an automatic channel window can advance on each run.
Model version growth proves publication, not an improvement in forecasting skill.

## Custom code stays inspectable

Persist user-specific preprocessing, training and post-processing directly in
`python_code` or graph node code, with parameters in `configuration.inputs`.
For example, calculate a chosen waveform RMS target in the compute block, or add
an orchestration-side quality gate after `work.compute` returns. Log the decision
and return a small structured result. A local file in the agent workspace is not
automatically available inside Maxwell; embed its contents in the Work procedure
or use a supported managed artifact/reference.

Do not implement a gate that hides a poor evaluation or silently skips requested
versioning. If promotion should require improvement, distinguish artifact creation
from deployment/promotion and implement only supported platform actions. Training
does not authorize production deployment or automatic control-system changes.
