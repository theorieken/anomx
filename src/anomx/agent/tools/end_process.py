"""End-process tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property


class EndProcessTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="end_process",
            description="End a running async CLI process.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "process_id": {
                        "type": "string",
                        "description": "Async process id to end.",
                    },
                },
                ["statement", "process_id"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(self.name, arguments)
        process_id = str(arguments.get("process_id") or "").strip()
        if not process_id:
            return context.json_result({"error": "end_process requires a process_id."})
        return context.runtime.end_process(process_id, context.session_path)
