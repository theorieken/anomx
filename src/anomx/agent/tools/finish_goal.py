"""Explicit completion of a user-enabled persistent goal."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema


class FinishGoalTool(BaseTool):
    """Close goal mode only after recording its outcome in GOAL.md."""

    def __init__(self) -> None:
        super().__init__(
            name="finish_goal",
            description=(
                "End goal mode after reviewing GOAL.md and checking the outcome. "
                "Use completed only when the goal is achieved; use stopped for a user "
                "cancellation or a blocker that prevents further progress. Explain what "
                "was verified and any unresolved work in summary. Then deliver the final answer."
            ),
            parameters=object_schema(
                {
                    "status": {"type": "string", "enum": ["completed", "stopped"]},
                    "summary": {
                        "type": "string",
                        "description": "Verified outcome or precise reason for stopping.",
                    },
                },
                ["status", "summary"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        try:
            result = context.runtime.finish_goal(
                context.session_path,
                status=str(arguments.get("status", "")),
                summary=str(arguments.get("summary", "")),
            )
        except (OSError, ValueError) as error:
            result = {"error": f"Cannot record goal completion: {error}"}
        return context.json_result(result)
