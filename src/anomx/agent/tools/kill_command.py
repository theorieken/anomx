"""Kill-command tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema


class KillCommandTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="kill_command",
            description="Kill a currently running long-running command tool call.",
            parameters=object_schema(
                {
                    "command_id": {
                        "type": "string",
                        "description": "Long-running command id to kill.",
                    },
                },
                ["command_id"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.runtime._emit_operator_tool_statement(
            self.name, arguments, context.callbacks
        )
        return context.runtime._kill_command_tool(
            arguments, context.session_path, context.callbacks
        )
