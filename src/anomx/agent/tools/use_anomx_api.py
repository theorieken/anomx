"""Anomx Platform API tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property
from anomx.agent.helpers.anomx_api import AnomxApiError, call_anomx_api, connection_from_home


class UseAnomxApiTool(BaseTool):
    """Call the connected Anomx Platform API and store the raw response."""

    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="use_anomx_api",
            description=(
                "Call the connected Anomx Platform REST API. The tool returns metadata "
                "and writes the response body to ~/.anomx/responses as JSON."
            ),
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                        "description": "HTTP method.",
                    },
                    "path": {
                        "type": "string",
                        "description": "API path such as /objects or /data/channels.",
                    },
                    "query": {
                        "type": "object",
                        "description": "Optional query parameters.",
                        "additionalProperties": True,
                    },
                    "body": {
                        "type": "object",
                        "description": "Optional JSON request body for non-GET requests.",
                        "additionalProperties": True,
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional extra HTTP headers.",
                        "additionalProperties": {"type": "string"},
                    },
                    "output_name": {
                        "type": "string",
                        "description": "Optional response filename stem.",
                    },
                },
                ["statement", "method", "path"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(self.name, arguments)
        connection = connection_from_home(context.runtime.home)
        if connection is None:
            return context.json_result(
                {
                    "error": "No Anomx Platform connection is configured.",
                    "connected": False,
                }
            )
        try:
            result = call_anomx_api(
                connection,
                method=str(arguments.get("method") or "GET"),
                path=str(arguments.get("path") or ""),
                query=_object_or_none(arguments.get("query")),
                body=arguments.get("body") if isinstance(arguments.get("body"), dict) else None,
                headers=_string_object_or_none(arguments.get("headers")),
                output_name=str(arguments.get("output_name") or ""),
            )
        except AnomxApiError as error:
            return context.json_result({"error": str(error), "connected": True})
        return context.json_result(result)


def _object_or_none(value: object) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _string_object_or_none(value: object) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    return {str(key): str(item) for key, item in value.items()}
