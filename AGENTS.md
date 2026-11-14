# Anomx package guidance

This repository owns the open-source Python library and portable agent harness.
The scientific core must remain useful from a script without the Anomx platform,
an agent session, Django or a running service. The sibling platform owns product
orchestration, authorization, persistence, distributed workers and collaboration.

## Read the instructions for your area

Nested guides add detail to these rules. Paths below are repository-relative.

| Area | Guidance |
| --- | --- |
| Source ownership and dependency direction | [src/anomx/AGENTS.md](src/anomx/AGENTS.md) |
| Agent harness, runtime and CLI | [src/anomx/agent/AGENTS.md](src/anomx/agent/AGENTS.md) |
| Data containers, sources and processing | [src/anomx/data/AGENTS.md](src/anomx/data/AGENTS.md) |
| Models and inference | [src/anomx/models/AGENTS.md](src/anomx/models/AGENTS.md) |
| Scoring and detection | [src/anomx/detection/AGENTS.md](src/anomx/detection/AGENTS.md) |
| Portable execution and run evidence | [src/anomx/execution/AGENTS.md](src/anomx/execution/AGENTS.md) |
| External adapters | [src/anomx/integrations/AGENTS.md](src/anomx/integrations/AGENTS.md) |

[docs/architecture.md](docs/architecture.md) explains the current contracts.
Prefer actual code over historical sketches when they differ.

## Working rules

- Use Python 3.11+, four spaces, typed public interfaces and small composable
  abstractions. Group implementations by their owning domain.
- Keep `base` for class contracts/capabilities; helpers for supporting operations.
  Do not put unrelated functions in a catch-all file or import platform internals.
- Keep the default install lightweight. Heavy ML/storage dependencies belong
  behind the extras in `pyproject.toml`; imports must not load unused integrations.
- Preserve lazy public exports in `__init__.py`, matching `__init__.pyi` stubs
  and `py.typed`. Public names, component keys, serialized state and persisted
  Python paths are contracts, not incidental filenames.
- Batch and online semantics both matter. Preserve shape, time/index alignment,
  train/inference boundaries and inspectable score/threshold/label metadata.
- Preserve unrelated working-tree changes. Before removing code, inspect public
  imports, component discovery, stubs, runtime resources and platform consumers.
- Use concise reStructuredText docstrings in handwritten Python. Keep imports,
  constants and definitions in order; keep class attributes/configuration before
  decorated/private/public methods when dependencies permit.
- Retain attribution and inherited documentation in native TimeSeries,
  processing and metric ports. Do not mass reformat third-party-derived source.
- Keep agent reasoning and implementation notes out of user-facing CLI text.

## Documentation and verification

Keep READMEs, API docs and runtime skills beside their owner. `AGENTS.md` records
ownership/conventions, not a work log. Put investigation/design notes under
`brain/audits`, `brain/architecture` or `brain/reference` when needed; consolidate
existing notes and never copy credentials into them.

For documentation-only work, review paths and consistency. For behavior changes,
add or update focused tests and use the existing toolchain:

- `ruff check <changed paths>` and `mypy src/anomx` for static checks.
- `pytest <relevant test paths>` for the affected behavior.
- `python -m build` for explicitly requested packaging/release validation.

Use the configured environment and installed dependencies. Do not install every
ML extra just to check a small change. Platform work follows the platform's
separate restriction on builds/tests. Report exactly what was checked.

Publishing and deployment require task authorization. Keep package versions,
release inputs and consuming platform versions coordinated when a release is
requested; a local source cleanup does not publish anything.
