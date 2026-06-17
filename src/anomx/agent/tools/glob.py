"""Glob tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property


class GlobTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="glob",
            description="Find files by glob pattern inside the trusted workspace.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "pattern": {"type": "string", "description": "Glob pattern."},
                    "path": {"type": "string", "description": "Root path for the glob."},
                    "limit": {"type": "integer", "description": "Maximum matches."},
                },
                ["statement", "pattern", "path", "limit"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.runtime._emit_operator_tool_statement(
            self.name, arguments, context.callbacks
        )
        return context.runtime._glob_tool(arguments)
