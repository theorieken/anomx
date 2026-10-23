"""Collect an asynchronous approval and execute its authorized action exactly once."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema


class CheckApprovalTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="check_approval",
            description=(
                "Check a pending approval by approval_id. Once approved, this executes the "
                "original action exactly once and returns its result. Continue independent "
                "work while pending; use wait_seconds up to 60 for a critical dependency. "
                "Never resubmit the original action while its approval is open."
            ),
            parameters=object_schema(
                {
                    "approval_id": {"type": "string"},
                    "wait_seconds": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 60,
                        "description": "Seconds to wait; default 0 returns immediately.",
                    },
                },
                ["approval_id"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        try:
            wait = float(arguments.get("wait_seconds") or 0)
        except (ValueError, TypeError):
            return context.json_result({"error": "wait_seconds must be a number."})
        if wait > 0 and context.callbacks.status is not None:
            context.callbacks.status(f"Waiting:{min(60.0, max(0.0, wait))}")
        return context.json_result(
            context.runtime.check_approval(
                str(arguments.get("approval_id") or ""),
                wait,
            )
        )
