# Portable execution

Read [source guidance](../AGENTS.md). `base/` owns execution/tracking contracts,
`jobs.py` portable node definitions, `runner.py` synchronous execution,
`local.py` local evaluation, `runs.py` lifecycle/tracking,
`backends/` event sinks and `work/` host callbacks, graphs and artifacts.

- Keep execution usable without Django, NATS, Celery or an agent. Distributed
  scheduling, authorization, workers and database persistence belong to the platform.
- WorkGraph and JobDefinition are distinct formats; do not silently convert
  their node types or execution semantics during a cleanup.
- Preserve cancellation, error propagation, active-run scope, timestamps,
  metrics, artifacts and callback ordering.
- Keep host access explicit through context/client contracts. Bound resource
  use and make materialization deliberate.
- Preserve serialization/public names and update exports, stubs and platform
  consumers together. Optional plotting/framework imports stay lazy.
