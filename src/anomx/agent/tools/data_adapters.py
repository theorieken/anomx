"""Manage declarative file-to-channel adapters through a connected platform."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property
from anomx.agent.tools.use_anomx_api import UseAnomxApiTool


class ManageDataAdaptersTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="manage_data_adapters",
            description=(
                "Inspect and manage folder-level data packages and channel mappings. "
                "List candidates and inspect "
                "file schemas/previews, or propose a declarative mapping. Readers support Parquet, "
                "Arrow, CSV, JSON/JSONL, NumPy (no pickle), Excel XLSX and HDF5. Recipes use "
                "time_column, value_column, time_unit "
                "(seconds/milliseconds/microseconds/nanoseconds), "
                "optional sheet, delimiter, datasets or HDF5 sample_axis (-1 for the last axis). "
                "Use one folder path_pattern for compatible files. Sources are read-only: mapping "
                "changes Anomx metadata only. Arrays without timestamps require a known "
                "start_at and sample_interval_seconds. Native Anomx Parquet uses native=true "
                "and source_recorded_channel_id. Never invent timing or identity. Proposals "
                "remain inactive until activated. Approval and storage access policies apply; "
                "background runs may inspect and propose, but cannot activate or change access."
            ),
            parameters=object_schema(
                {
                    "statement": statement_property("Describe the adapter work."),
                    "integration_id": {
                        "type": "string",
                        "description": "Storage integration UUID.",
                    },
                    "action": {
                        "type": "string",
                        "enum": [
                            "list",
                            "inspect",
                            "discover",
                            "propose",
                            "activate",
                            "validate",
                            "disable",
                        ],
                    },
                    "id": {
                        "type": "string",
                        "description": "Adapter UUID for an existing adapter.",
                    },
                    "state": {
                        "type": "string",
                        "description": "Filter: needs_review, ready, error, disabled or queued.",
                    },
                    "offset": {"type": "integer", "minimum": 0},
                    "package_id": {
                        "type": "string",
                        "description": "Limit candidates to this logical data package.",
                    },
                    "view": {
                        "type": "string",
                        "enum": ["packages", "adapters"],
                        "description": "List packages or individual signal mappings.",
                    },
                    "reader_options": {
                        "type": "object",
                        "additionalProperties": True,
                        "description": "HDF5 datasets selects arrays or subgroups to inspect.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Discovered file path relative to the integration root.",
                    },
                    "path_pattern": {
                        "type": "string",
                        "description": "Glob identifying files with the same schema.",
                    },
                    "name": {"type": "string"},
                    "mapping": {
                        "type": "object",
                        "additionalProperties": True,
                        "description": "Declarative reader recipe; never executable code.",
                    },
                    "channel": {
                        "type": "string",
                        "description": "Verified existing channel reference, if known.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Evidence supporting this mapping and channel identity.",
                    },
                    "create_channel": {
                        "type": "boolean",
                        "description": "Create a channel when activating an unmatched candidate.",
                    },
                },
                ["statement", "integration_id", "action"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        try:
            integration_id = str(UUID(str(arguments.get("integration_id", ""))))
        except ValueError:
            return context.json_result(
                {"ok": False, "error": "A storage integration UUID is required."}
            )
        action = arguments.get("action")
        if action not in {
            "list",
            "inspect",
            "discover",
            "propose",
            "activate",
            "validate",
            "disable",
        }:
            return context.json_result({"ok": False, "error": "Unknown adapter action."})
        request: dict[str, Any] = {
            "statement": arguments.get("statement") or "Managing data adapters",
            "method": "GET" if action in {"list", "inspect"} else "POST",
            "path": f"/integrations/{integration_id}/data-adapters",
        }
        if action == "list":
            request["query"] = {
                key: arguments[key]
                for key in ("state", "offset", "package_id", "view")
                if key in arguments
            }
        elif action == "inspect":
            request["query"] = {"path": arguments.get("path", "")}
            if arguments.get("reader_options"):
                request["query"]["reader_options"] = json.dumps(arguments["reader_options"])
        else:
            request["body"] = {
                key: value
                for key, value in arguments.items()
                if key not in {"statement", "integration_id", "state", "offset"}
            }
        return UseAnomxApiTool(statement_description="Managing data adapters").execute(
            request, context
        )
