"""Create-plan tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext
from anomx.agent.helpers.state import build_plan_steps, serialize_plan_steps
from anomx.agent.tools.plan_schema import plan_schema


class CreatePlanTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="create_plan",
            description="Create a user-visible ordered plan.",
            parameters=plan_schema(require_position=False),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(self.name, arguments)
        if not context.runtime.agent_spec.can_use_plans:
            return context.json_result({"error": "This agent kind cannot create plans."})
        if context.session_path is None:
            return context.json_result({"error": "create_plan requires a session."})

        steps = build_plan_steps(arguments.get("steps"))
        if not steps:
            return context.json_result({"error": "create_plan requires at least one step."})

        payload = {"steps": serialize_plan_steps(steps), "action": "create"}
        context.runtime.home.append_session_event(
            context.session_path,
            "plan_update",
            payload,
        )
        return context.json_result({"created": True, **payload})
