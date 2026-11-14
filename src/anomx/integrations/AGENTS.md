# Optional integrations

Read [source guidance](../AGENTS.md). `darts/` adapts Darts-native contracts;
`platform/` provides authenticated HTTP access and run/dataset adapters;
`workspaces.py` supports remote workspace operations.

Keep adapters optional and imports lazy. The platform client is a generic JSON
transport; dataset endpoint behavior belongs in the data source adapter, run
behavior with tracking, and workspace behavior with its consumer. Do not grow a
single client with all product domain logic.

Never import `anomx_platform`, Django models or tenant database objects.
Validate response shapes and propagate actionable errors without exposing
credentials. Preserve time/index and result metadata at interoperability
boundaries. Update optional extras, exports, stubs and compatibility examples
together when external dependencies change.
