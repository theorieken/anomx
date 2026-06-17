"""Read-file tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property


class ReadFileTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="read",
            description="Read a file inside the trusted workspace.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "path": {"type": "string", "description": "File path to read."},
                    "start_line": {
                        "type": "integer",
                        "description": "One-based start line.",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum lines to return.",
                    },
                },
                ["statement", "path", "start_line", "max_lines"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.runtime._emit_operator_tool_statement(
            self.name, arguments, context.callbacks
        )
        return context.runtime._read_file_tool(arguments)
