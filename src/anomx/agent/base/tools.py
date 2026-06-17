"""Object-oriented tool primitives for Anomx agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

JsonSchema = dict[str, Any]
ToolDefinition = dict[str, Any]


@dataclass(frozen=True)
class BaseTool:
    """Base class for model-callable tools.

    Tool objects own the public schema exposed to models. Execution remains
    runtime-specific because most tools operate on mutable session state.
    """

    name: str
    description: str
    parameters: JsonSchema

    def definition(self) -> ToolDefinition:
        """Return the OpenAI/Anthropic-compatible function definition."""

        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def object_schema(
    properties: dict[str, JsonSchema],
    required: list[str],
) -> JsonSchema:
    """Build a strict object schema for a tool parameter payload."""

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def statement_property(description: str) -> JsonSchema:
    """Return the common statement property schema."""

    return {"type": "string", "description": description}
