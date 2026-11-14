# Bundled agent skill resources

Read the [harness guide](../AGENTS.md). These files are runtime product inputs,
not developer instruction files or disposable notes.

- Each skill has `<command>/SKILL.md`, with references/assets beside it. Keep
  frontmatter, command identifiers and relative resource paths valid.
- Parsing/loading is in `__init__.py`; file validation/normalization is in
  `files.py`. Preserve bundled and workspace skill behavior.
- Keep reusable package workflows here. Platform-only skills belong in the
  platform's `backend/anomx_platform/agent_skills`.
- Update examples/templates with public API changes and verify their imports.
  Do not claim an action/tool exists unless the current harness exposes it.
- Preserve resource inclusion when changing packaging. Avoid adding developer
  notes inside a skill directory: its files may be offered to a running agent.
