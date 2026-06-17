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
        context.emit_operator_statement(self.name, arguments)
        command_id = str(
            arguments.get("command_id") or arguments.get("process_id") or ""
        ).strip()
        if not command_id:
            return context.json_result({"error": "kill_command requires a command_id."})
        return context.runtime._end_process(
            command_id,
            context.session_path,
            context.callbacks,
            allowed_sources=context.runtime._command_tool_sources(),
        )
