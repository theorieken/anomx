"""Remove-subagent tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property


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
        context.runtime._emit_operator_tool_statement(
            self.name, arguments, context.callbacks
        )
        return context.runtime._remove_subagent_tool(arguments, context.session_path)
