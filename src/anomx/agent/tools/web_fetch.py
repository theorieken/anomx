"""Web-fetch tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property


class WebFetchTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="web_fetch",
            description="Fetch a web page by URL.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "url": {"type": "string", "description": "HTTP or HTTPS URL."},
                },
                ["statement", "url"],
            ),
            aliases=("webfetch",),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.runtime._emit_operator_tool_statement(
            self.name, arguments, context.callbacks
        )
        return context.runtime._web_fetch_tool(arguments)
