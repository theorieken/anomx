"""Wait tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema


class WaitTool(BaseTool):
    def __init__(self, *, target_description: str) -> None:
        super().__init__(
            name="wait",
            description=f"Wait up to 60 seconds for {target_description}.",
            parameters=object_schema({}, []),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        return context.runtime._wait_tool(arguments, context.callbacks)
