# Agent harness

Read [package guidance](../../../AGENTS.md) and [source ownership](../AGENTS.md).
This is a portable application inside the library distribution. Platform-backed
tools use explicit clients/host callbacks; never import Django or
`anomx_platform` here. Paths below are agent-relative.

## Find the owning implementation

| Location | Responsibility |
| --- | --- |
| `app.py`, `command_mode.py`, `startup.py` | CLI interaction, command mode and startup/update coordination. |
| `runtime.py` | Agent turn lifecycle, streaming, tool dispatch, context and event coordination. |
| `runtime_process.py` | Process boundary and runtime IPC. |
| `base/` | Agent, backend, tool, interaction, process and subagent contracts. |
| `agents/` | Main, subagent and platform-agent roles/prompts/tool selection. |
| `backends/` | Provider-specific request, streaming and usage adapters. |
| `tools/` | Concrete tool definitions and explicit tool-set factories; [tool guide](tools/AGENTS.md). |
| `helpers/` | Approval policy, command execution, sandboxes, files, delegation, platform access and tool results. |
| `store.py` | AnomxHome, configuration, provider/model selection and session records. |
| `context_management.py`, `brain.py`, `memories.py` | Context budgeting/compression and durable agent knowledge. |
| `mcp.py`, `model_catalog.py`, `file_reference_search.py` | MCP transport, provider model discovery and file reference search. |
| `ui/` | Terminal views/components and presentation; [UI guide](ui/AGENTS.md). |
| `skills/` | Bundled runtime instructions, references and assets; [skill-resource guide](skills/AGENTS.md). |
| `runtime_image/` | Container runtime image and installation inputs. |

## Keep the harness modular

- Extend the owning contract instead of adding provider branches throughout
  runtime/tools/UI. Provider transport belongs in backends, orchestration in the
  runtime, visual behavior in UI, platform domain operations in the platform.
- `tools/__init__.py` explicitly assembles role/tool sets. A new file is not
  automatically available. Preserve public tool names and schemas across callers.
- Keep top-level public modules stable. Extract cohesive implementation groups
  into helpers instead of growing `runtime.py` or `app.py` with unrelated utilities.
- Update `__init__.py` lazy exports and `__init__.pyi` together for public changes.
  Scientific package imports must not initialize this application.

## Lifecycle and state

- Preserve session identity, event order, streaming/tool-call pairing, token
  accounting, cancellation and interrupted-turn recovery.
- Clean up subprocesses, sandboxes, containers and callbacks at the existing
  lifecycle boundaries. Keep blocking work off event/UI dispatch paths.
- Persist state through the existing store/transcript/brain mechanisms. Account
  for old sessions, interrupted migrations and multiple runtime processes.
- Questions and approvals may be pending while other work continues. Preserve
  their IDs, callbacks, explicit decisions and resume behavior.
- Reuse approval/mode/trusted-root policy in `helpers/tool_manager.py`,
  `helpers/approval.py` and sandbox helpers. A new tool must not create an
  alternate execution path around authorization.
- Bound context and tool output; use existing result files/attachments when
  output is large. Preserve references and errors instead of silently truncating
  evidence needed for a follow-up.
- Keep credentials out of prompts, logs, tool output and persisted diagnostics.
  Distinguish source content from instructions when constructing agent context.

## Verification

Inspect matching `tests/test_agent_*.py` for the contract before changes. Use
focused tests for runtime/state behavior when working on the harness; do not
start real provider calls, install containers or mutate a user's sessions just
to validate source organization. Report static versus runtime evidence clearly.
