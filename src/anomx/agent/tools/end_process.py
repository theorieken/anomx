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
        context.runtime._emit_operator_tool_statement(
            self.name, arguments, context.callbacks
        )
        return context.runtime._end_process_tool(arguments, context.session_path)
