"""Update-plan tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext
from anomx.agent.tools.plan_schema import plan_schema


class UpdatePlanTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="update_plan",
            description="Update the user-visible ordered plan.",
            parameters=plan_schema(require_position=True),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.runtime._emit_operator_tool_statement(
            self.name, arguments, context.callbacks
        )
        if not context.runtime.agent_spec.can_use_plans:
            return context.json_result({"error": "This agent kind cannot update plans."})
        return context.runtime._update_plan_tool(arguments, context.session_path)
