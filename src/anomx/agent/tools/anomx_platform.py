"""Focused read tools for a connected Anomx Platform."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema
from anomx.agent.helpers.anomx_api import AnomxApiError, call_anomx_api, connection_from_home


def _clamped_limit(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(10, min(100, parsed))


def _read_call_payload(result: dict[str, object]) -> object:
    response_path = str(result.get("response_path") or "").strip()
    if not response_path:
        return {}
    try:
        return json.loads(Path(response_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _get_payload(
    context: ToolExecutionContext,
    *,
    path: str,
    query: dict[str, object] | None = None,
) -> tuple[dict[str, object], object] | str:
    connection = connection_from_home(context.runtime.home)
    if connection is None:
        return context.json_result(
            {"connected": False, "error": "No Anomx Platform connection is configured."}
        )
    try:
        result = call_anomx_api(connection, method="GET", path=path, query=query)
    except AnomxApiError as error:
        return context.json_result({"connected": True, "error": str(error)})
    return result, _read_call_payload(result)


class GetAnomxObjectDetailsTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="get_anomx_object_details",
            description=(
                "Get the complete serialized details of an Anomx object by object reference."
            ),
            parameters=object_schema(
                {
                    "object_reference": {
                        "type": "string",
                        "description": "Canonical Anomx object reference.",
                    }
                },
                ["object_reference"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(
            self.name, arguments, default_statement="Loading Anomx object"
        )
        object_reference = str(arguments.get("object_reference") or "").strip()
        response = _get_payload(context, path=f"/objects/{object_reference}")
        if isinstance(response, str):
            return response
        result, payload = response
        return context.json_result(
            {"object_reference": object_reference, "object": payload, "request": result}
        )


class SearchAnomxObjectsTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="search_anomx_objects",
            aliases=("serach_anomx_objects",),
            description=(
                "Search Anomx objects through the unified objects endpoint. Results include "
                "canonical "
                "object references for follow-up tool calls. limit is clamped to 10-100."
            ),
            parameters=object_schema(
                {
                    "query": {"type": "string", "description": "Object name or search terms."},
                    "limit": {
                        "type": "integer",
                        "minimum": 10,
                        "maximum": 100,
                        "description": "Maximum results (default 10).",
                    },
                },
                ["query"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(
            self.name, arguments, default_statement="Searching Anomx objects"
        )
        query = str(arguments.get("query") or "").strip()
        limit = _clamped_limit(arguments.get("limit"), 10)
        response = _get_payload(context, path="/objects", query={"query": query, "limit": limit})
        if isinstance(response, str):
            return response
        result, payload = response
        return context.json_result(
            {"query": query, "limit": limit, "results": payload, "request": result}
        )


class SearchAnomxDataChannelsTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="search_anomx_data_channels",
            description=(
                "Search live Anomx data channels. Build the query from the beginnings of known "
                "identifier segments (for example successive path/device prefixes), not arbitrary "
                "middle substrings. Returns up to limit concrete channels with object references "
                "plus non-channel hints that show how to continue the identifier. limit is clamped "
                "to 10-100."
            ),
            parameters=object_schema(
                {
                    "query": {
                        "type": "string",
                        "description": "Known leading identifier parts or prefixes.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 10,
                        "maximum": 100,
                        "description": "Maximum concrete channels and hints (default 10).",
                    },
                },
                ["query"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(
            self.name, arguments, default_statement="Searching Anomx data channels"
        )
        query = str(arguments.get("query") or "").strip()
        limit = _clamped_limit(arguments.get("limit"), 10)
        channel_response = _get_payload(
            context, path="/data/channels/live-search", query={"query": query, "limit": limit}
        )
        if isinstance(channel_response, str):
            return channel_response
        hints_response = _get_payload(
            context, path="/data/channels/live-hints", query={"query": query, "limit": limit}
        )
        if isinstance(hints_response, str):
            return hints_response
        channel_request, channel_payload = channel_response
        hints_request, hints_payload = hints_response
        channels = channel_payload.get("items", []) if isinstance(channel_payload, dict) else []
        hints = (
            [item for item in hints_payload if isinstance(item, dict) and not item.get("channel")]
            if isinstance(hints_payload, list)
            else []
        )
        return context.json_result(
            {
                "channels": channels[:limit] if isinstance(channels, list) else [],
                "hints": hints[:limit],
                "limit": limit,
                "query": query,
                "requests": [channel_request, hints_request],
            }
        )


class GetAnomxDataChannelHistoryTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="get_anomx_data_channel_history",
            description=(
                "Get historical samples for a concrete Anomx data channel object reference."
            ),
            parameters=object_schema(
                {
                    "object_reference": {
                        "type": "string",
                        "description": "Canonical data_channel object reference.",
                    },
                    "range": {
                        "type": "string",
                        "description": "Relative history window such as 15m, 1h, or 7d.",
                    },
                    "max_data_points": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Maximum returned data points (default 100).",
                    },
                },
                ["object_reference", "range"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(
            self.name, arguments, default_statement="Loading Anomx channel history"
        )
        object_reference = str(arguments.get("object_reference") or "").strip()
        if not object_reference.startswith("data_channel-"):
            return context.json_result({"error": "object_reference must identify a data_channel."})
        history_range = str(arguments.get("range") or "1h").strip()
        max_points = max(1, min(100, context.positive_int(arguments.get("max_data_points"), 100)))
        response = _get_payload(
            context,
            path=f"/data/channels/{object_reference}/history",
            query={"range": history_range, "max_points": max_points},
        )
        if isinstance(response, str):
            return response
        result, payload = response
        return context.json_result(
            {
                "history": payload,
                "max_data_points": max_points,
                "object_reference": object_reference,
                "range": history_range,
                "request": result,
            }
        )
