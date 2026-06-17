"""Remove-plan tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property


class RemovePlanTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="remove_plan",
            description="Clear the current user-visible plan.",
            parameters=object_schema(
                {"statement": statement_property(statement_description)},
                ["statement"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        return context.runtime._remove_plan_tool(
            arguments, context.session_path, context.callbacks
        )
