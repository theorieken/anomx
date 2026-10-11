---
name: manage-acquisition
description: Start, inspect or adjust Anomx data recording using DAQ Work in a Job, and verify that real samples are stored for later analysis or training.
metadata:
    title: Manage Data Acquisition
    requires_platform: true
    keywords: record, recording, acquisition, DAQ, sampling, buffer, collect data
---

# Record data with DAQ Work

Read `manage-jobs` for the Job/Work contract and `retrieve-data` for channel
resolution. First inspect an existing recording for the requested signal. If it
already provides the required data, reuse it instead of starting another recorder.
A job can contain one DAQ Work and several independently scheduled compute Work
items; adding training must preserve the running DAQ item and its configuration.

Resolve channels and read `/jobs/build-options` for an actual DAQ service, store,
protocol limits and supported shapes. Choose the requested sampling frequency,
duration/retention and `record_data=true`. Use `work_mode=continuous` for an
ongoing recording, or a supported bounded mode for a defined capture. Distinguish
acquisition sampling frequency from how often a compute Work starts.

Create the job/Work and use its supported start/run action when recording was
requested. An accepted action, active flag or healthy service is insufficient:
inspect the DAQ Work, JobRun/Run, channel bindings and a bounded recorded
history window. Report real sample times, coverage and failures. Do not equate
`Job.is_active=false` with stopped DAQ: it controls scheduling; inspect the actual
Work and run state, including a manually started continuous acquisition.

For training from this recording, use `create-dataset` with explicit channel
references and an observed start/end window, resolve a version and verify rows.
Keep gaps and cadence visible. Use `train-models` for separate compute Work; DAQ
flags alone do not train a custom model. Stop/pause only the requested Work or Job,
and account for the other Work items a whole-job action affects.
