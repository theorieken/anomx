"""Check-command-status tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema


class CheckCommandStatusTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="check_command_status",
            description=(
                "Check a currently running long-running command tool call and read "
                "its current CLI output."
            ),
            parameters=object_schema(
                {
                    "command_id": {
                        "type": "string",
                        "description": "Long-running command id to inspect.",
                    },
                },
                ["command_id"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.runtime._emit_operator_tool_statement(
            self.name, arguments, context.callbacks
        )
        return context.runtime._check_command_status_tool(arguments)
