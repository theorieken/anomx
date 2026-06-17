"""Start-subagent tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property


class StartSubagentTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="start_subagent",
            description="Start an asynchronous subagent.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "agent_kind": {
                        "type": "string",
                        "enum": ["general", "explore"],
                        "description": "Kind of subagent to start.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Short display name for the subagent.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Complete task prompt for the subagent.",
                    },
                },
                ["statement", "agent_kind", "name", "prompt"],
            ),
            aliases=("start_agent",),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.runtime._emit_operator_tool_statement(
            self.name, arguments, context.callbacks
        )
        return context.runtime._start_subagent_tool(
            arguments, context.session_path, context.callbacks
        )
