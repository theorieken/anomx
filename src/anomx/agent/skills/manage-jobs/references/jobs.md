# Current job contracts

`GET /jobs/build-options` supplies `daq_services`, `timeseries_stores`, `sample_stores`, `components`, and the legacy protocol sampling-frequency limit. `/jobs/component-options` and `/models`, `/algorithms`, `/scorers`, `/detectors`, `/components` expose further catalogs. Inspect component `config_schema`, defaults, capabilities, signature/parameters and code version instead of selecting by name alone.

The create serializer supports `job_type=data_acquisition`, `model_training` and `advanced`; standalone `anomaly_detection` creation is rejected. Single-purpose jobs require exactly one matching item in `works`. Advanced jobs can contain independently scheduled DAQ and compute Work. Follow `train-models` for the complete Python/graph, publication and recurring training contracts. Acquisition can also enable supported detection or forecasting through validated `runtime_config`; flags alone do not define a custom modeling pipeline.

- `work_mode=continuous` requires no duration; `fixed_duration` and `sliding_window` require a positive `duration_seconds`.
- Sampling frequency must be positive and supported by the chosen protocol/service; do not request a frequency beyond the published legacy limit.
- DAQ jobs require at least one recordable channel and a DAQ service. Current channel validation supports single/array bool/float/int channels.
- Time-series and independent-sample channels cannot be mixed in one job. Select the appropriate time-series or sample store from build-options. Forecasting is limited to time-series data.
- Detection's current configuration is manual and requires a positive detection period. Inspect the current schema/validation for required training intervals, models, scorers and detectors; do not invent a config from future product plans.
- `record_data`, `detect_anomalies`, and `forecast_data` are separate concerns. Describe which outputs the user requested and the storage/compute consequences.

Related endpoints: `/job-channel-bindings` (live channel roles, sampling, transport, read parameters), `/job-data-bindings` (recordings/files), `/job-component-bindings` (model/scorer/detector roles, selection and config). Use canonical references and serializer-supported relation fields. Read existing bindings before updates; a replacement array must retain unrelated bindings.

Compute definitions and evidence use `/job-works`, `/work-runs`, `/models/trainings`, `/models/trainings/<id>/telemetry`, `/models/models/<id>/training-summary`, and `/findings`. DAQ/legacy component evidence uses `/job-runs`, `/run-component-usages`, `/run-metric-points` and `/model-artifacts`. A job's current configuration may differ from an old run's snapshot. For a failure, correlate run timestamps/errors with the actual DAQ/compute service and storage identity.

Maintainer sources: platform `modules/jobs/models.py`, `serializers.py`, `viewsets.py`, worker runtimes and job utilities. Expand this reference as execution contracts evolve; preserve the runtime-verified distinctions above.
