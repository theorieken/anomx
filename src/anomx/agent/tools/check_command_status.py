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
        context.emit_operator_statement(self.name, arguments)
        command_id = str(
            arguments.get("command_id") or arguments.get("process_id") or ""
        ).strip()
        if not command_id:
            return context.json_result(
                {"error": "check_command_status requires a command_id."}
            )
        process_state = context.runtime._command_state(command_id)
        if process_state is None:
            return context.json_result({"error": "Unknown command id."})
        payload = context.runtime._command_state_payload(process_state)
        context.runtime._append_command_event_snapshot(process_state)
        return context.json_result(payload)
