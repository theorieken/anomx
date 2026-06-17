"""Remove-subagent tool."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property
from anomx.agent.tools._time import utc_now_iso


class RemoveSubagentTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="remove_subagent",
            description="Remove a subagent from prompt context and UI.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "agent_id": {"type": "string", "description": "Subagent id."},
                },
                ["statement", "agent_id"],
            ),
            aliases=("remove_agent", "interrupt_agent"),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(self.name, arguments)
        if not context.runtime.agent_spec.can_spawn_subagents:
            return context.json_result({"error": "Only the build agent can remove subagents."})
        agent_id = str(arguments.get("agent_id", "")).strip()
        if not agent_id:
            return context.json_result({"error": "remove_subagent requires an agent_id."})
        with context.runtime._subagent_lock:
            state = context.runtime._subagents.get(agent_id)
            if state is None:
                return context.json_result(
                    {"removed": False, "error": "Unknown subagent id."}
                )
            state.status = "removed"
            state.statement = str(arguments.get("statement", "")).strip() or "Removed subagent"
            state.finished_at = utc_now_iso()
            state.cancel_event.set()
            runtime = state.runtime
        if runtime is not None:
            with suppress(Exception):
                runtime.abort_current_turn(state.session_path)
            with suppress(Exception):
                runtime.shutdown(state.session_path)
        context.runtime._publish_subagent_state(
            state,
            context.session_path,
            message=state.statement,
        )
        with context.runtime._subagent_lock:
            context.runtime._subagents.pop(agent_id, None)
        return context.json_result({"removed": True, "agent_id": agent_id})
