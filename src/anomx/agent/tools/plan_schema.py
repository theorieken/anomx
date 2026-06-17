"""Shared planning tool schemas."""

from __future__ import annotations

from anomx.agent.base.tools import object_schema, statement_property


def plan_schema(*, require_position: bool) -> dict[str, object]:
    """Return the strict schema for create/update plan tools."""

    properties: dict[str, dict[str, str]] = {
        "title": {
            "type": "string",
            "description": "Short user-visible plan item title.",
        },
        "description": {
            "type": "string",
            "description": "Private operator-facing detail for this step.",
        },
        "is_done": {
            "type": "boolean",
            "description": "Whether this step is complete.",
        },
    }
    required = ["title", "description", "is_done"]
    if require_position:
        properties = {
            "position": {
                "type": "integer",
                "description": "One-based plan position.",
            },
            **properties,
        }
        required = ["position", *required]
    return object_schema(
        {
            "statement": statement_property(
                "Persistent user-visible working message for this tool call."
            ),
            "steps": {
                "type": "array",
                "description": "Ordered plan steps.",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        },
        ["statement", "steps"],
    )
