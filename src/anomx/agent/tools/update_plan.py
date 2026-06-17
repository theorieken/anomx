"""Update-plan tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext
from anomx.agent.helpers.state import latest_plan_steps, merge_plan_steps, serialize_plan_steps
from anomx.agent.tools.plan_schema import plan_schema


class UpdatePlanTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="update_plan",
            description="Update the user-visible ordered plan.",
            parameters=plan_schema(require_position=True),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(self.name, arguments)
        if not context.runtime.agent_spec.can_use_plans:
            return context.json_result({"error": "This agent kind cannot update plans."})
        if context.session_path is None:
            return context.json_result({"error": "update_plan requires a session."})

        current = latest_plan_steps(
            context.runtime.home.read_session_events(context.session_path)
        )
        steps = merge_plan_steps(current, arguments.get("steps"))
        if not steps:
            return context.json_result({"error": "update_plan requires plan steps."})

        payload = {"steps": serialize_plan_steps(steps), "action": "update"}
        context.runtime.home.append_session_event(
            context.session_path,
            "plan_update",
            payload,
        )
        return context.json_result({"updated": True, **payload})
