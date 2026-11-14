# Data domain

Read [source guidance](../AGENTS.md). `base/` owns clients/connectors/transforms
and observation contracts; `sources/` implements local, mounted, DAQ and
platform-backed access; `views/` owns materialization contracts; `processing/`
owns transformations, pipelines, encoders and DTW.

- `dataset.py` is the Dataset facade; `timeseries.py` is the native regular
  TimeSeries container. Keep raw/irregular observations distinct from explicitly
  materialized regular series.
- Preserve timestamps, timezone, channel/column identity, shape, dtype, metadata
  and source/version references across reads and transformations.
- Bound reads, page journals, previews and batch iteration. Do not materialize
  an entire remote dataset for a small selection.
- Fit transformations on training observations only. Preserve temporal state
  across batches; reject unsupported batch semantics instead of producing
  incorrect partial-bin results.
- Use explicit client/source contracts. This domain must not depend on models,
  detection, execution, the agent or platform backend internals.
- Retain third-party attribution and inherited docs in native TimeSeries ports.
  Add optional storage/interop imports only where needed; update public stubs.
