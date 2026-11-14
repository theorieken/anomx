# Agent tools

Read the [harness guide](../AGENTS.md). Each file owns a coherent tool or small
tool family. `__init__.py` assembles the tool sets used by agent roles.

- Reuse `BaseTool` and its context/result contracts. Keep names, descriptions,
  argument schema, validation and execution semantics consistent.
- Tool methods adapt arguments to runtime/helper operations; they do not own
  provider transports, terminal rendering or duplicate platform domain logic.
- Use existing approval, mode and trusted-root checks for shell/file/API work.
  Pending approval is distinct from denial, success and failure.
- Keep errors actionable and output bounded. Large results use existing result
  storage. Preserve object/file references for subsequent tools.
- Preserve cancellation, asynchronous-process IDs, question/approval callbacks
  and subagent lifecycle. Do not mark work complete before its effects complete.
- Register tools in the appropriate role sets and inspect schemas/callers before
  renaming. Update relevant skill resources when the agent-facing contract changes.
