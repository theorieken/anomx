---
name: manage-integrations
description: Discover the user's configured Anomx integrations, inspect connector capabilities and health, and choose real compute or storage destinations for a job or dataset.
metadata:
    title: Manage Integrations
    requires_platform: true
    keywords: integration, connector, Maxwell, compute, filesystem, storage, credentials
---

# Integrations and capabilities

Read `use-anomx-api`. `GET /integrations/connector-catalog` describes available
connector types; `GET /integrations` lists connections accessible to this user.
A catalog entry is not a configured connection. Inspect the selected integration's
`connector.kind`, `connector.key`, `is_active`, `state`, and `health_details`.
Do not print its config wholesale: it can contain credentials. Use the existing
integration's canonical reference, not a connector UUID, as the destination.

- Compute: `connector.kind=compute_system`. Maxwell is a compute integration;
  it is not a local CPU/GPU NodeService. `GET /integrations/<id>/compute-usage`
  reports utilization. Select it as Work `compute_integration`, leaving
  `compute_service=null`; dispatch heavy code through `work.compute`.
- Filesystems: follow `browse-files` for the integration's external tree. That
  tree is not mounted in the agent workspace. `/integrations/storage-destinations`
  lists supported output destinations; it does not enumerate training files.
- DAQ: use `GET /jobs/build-options` for available acquisition services/stores,
  `retrieve-data` for channel discovery and `manage-acquisition` for recording.
- Language models: `/integrations/llm-options` and
  `/integrations/<id>/language-models` concern the agent's LLM. They do not select
  the compute destination for a forecasting model.

`POST /integrations/<id>/check-connection` refreshes connection health when needed;
it does not prove that a training workload has its dependencies. Reuse authenticated
connections. Credentials belong in the platform's integration settings UI; never
request passwords or tokens in conversation, store them in Work, or work around a
permission denial. Do not reset a working integration merely because a job failed.

For Maxwell, `configuration.runtime_profile` on Work selects `base`, `ml`, or
`darts`. Training Work defaults to `ml` (CPU PyTorch, ONNX, ONNX Runtime); use
`darts` for Darts models, and explicit `ml` for ONNX inference. The platform prepares
and caches a versioned environment before executing the compute block. First-time
setup needs extra time; allow a bounded 30-minute timeout for bootstrap and initial
training, then use a smaller timeout for warm runs. Runtime provisioning errors
are failures, not reasons to put repeated pip installs into job code.
These managed profiles target CPU partitions; verify a suitable runtime and GPU
allocation explicitly before promising GPU acceleration.

Follow `manage-jobs` for orchestration and `train-models` for model execution,
evaluation and publication. Report target identity, actual run state, and missing
capabilities separately from whether the integration is merely reachable.
