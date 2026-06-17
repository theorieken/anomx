"""Web research tools."""

from __future__ import annotations

from anomx.agent.base.tools import BaseTool, object_schema, statement_property


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
        )


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
        )
