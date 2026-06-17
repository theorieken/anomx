"""Object-oriented tool primitives for Anomx agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anomx.agent.exceptions import ToolExecutionError

JsonSchema = dict[str, Any]
ToolDefinition = dict[str, Any]


@dataclass(frozen=True)
class ToolExecutionContext:
    """Runtime resources available to a tool execution."""

    runtime: Any
    callbacks: Any
    session_path: Path | None = None

    def json_result(self, payload: dict[str, Any]) -> str:
        """Serialize a tool result payload with the active runtime."""

        return self.runtime._json_tool_result(payload)


@dataclass(frozen=True)
class BaseTool:
    """Base class for model-callable tools.

    Tool objects own both the public schema exposed to models and the execution
    behavior for model tool calls. Runtime-specific resources are supplied
    through :class:`ToolExecutionContext`.
    """

    name: str
    description: str
    parameters: JsonSchema
    aliases: tuple[str, ...] = ()

    def definition(self) -> ToolDefinition:
        """Return the OpenAI/Anthropic-compatible function definition."""

        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def handles(self, name: str) -> bool:
        """Return whether this tool should execute a requested tool name."""

        normalized = name.strip()
        return normalized == self.name or normalized in self.aliases

    def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> str:
        """Execute the tool and return a serialized model-facing result."""

        raise ToolExecutionError(f"{self.name} does not implement execute().")


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
