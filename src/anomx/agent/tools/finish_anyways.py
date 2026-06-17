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
        if context.session_path is None:
            return context.json_result({"error": "finish_anyways requires a session."})
        context.runtime.home.append_session_event(
            context.session_path,
            "plan_update",
            {"steps": []},
        )
        context.emit_operator_statement(
            self.name,
            arguments,
            default_statement="Finishing anyway",
        )
        return context.json_result({"finish_anyways": True, "removed_plan": True})
