"""Grep tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property


class GrepTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="grep",
            description="Search file text inside the trusted workspace.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "pattern": {
                        "type": "string",
                        "description": "Regex or literal search text.",
                    },
                    "path": {"type": "string", "description": "File or directory path."},
                    "include": {"type": "string", "description": "File glob filter."},
                    "limit": {"type": "integer", "description": "Maximum matches."},
                },
                ["statement", "pattern", "path", "include", "limit"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.runtime._emit_operator_tool_statement(
            self.name, arguments, context.callbacks
        )
        return context.runtime._grep_tool(arguments)
