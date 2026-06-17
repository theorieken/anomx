"""List-directory tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property


class ListDirectoryTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="list",
            description="List a directory inside the trusted workspace.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "path": {"type": "string", "description": "Directory path to list."},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum entries to return.",
                    },
                },
                ["statement", "path", "limit"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.runtime._emit_operator_tool_statement(
            self.name, arguments, context.callbacks
        )
        return context.runtime._list_path_tool(arguments)
