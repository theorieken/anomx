"""Web-search tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property


class WebSearchTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="web_search",
            description="Search the web for relevant pages.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "query": {"type": "string", "description": "Search query."},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results.",
                    },
                },
                ["statement", "query", "limit"],
            ),
            aliases=("websearch",),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.runtime._emit_operator_tool_statement(
            self.name, arguments, context.callbacks
        )
        return context.runtime._web_search_tool(arguments)
