"""Finish-anyways tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property


class FinishAnywaysTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="finish_anyways",
            description=(
                "Clear the current user-visible plan and allow final delivery after "
                "the plan-finish checker asks for an explicit override."
            ),
            parameters=object_schema(
                {"statement": statement_property(statement_description)},
                ["statement"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        return context.runtime._finish_anyways_tool(
            arguments, context.session_path, context.callbacks
        )
