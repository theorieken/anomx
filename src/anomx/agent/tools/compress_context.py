"""Let agents checkpoint completed work before the hard context safeguard."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext


class CompressContextTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="compress_context",
            description=(
                "Request a substantial context checkpoint at a completed milestone. "
                "The runtime summarizes older history after the entire tool batch settles, "
                "preserving user instructions, recent complete exchanges, and original events. "
                "Continue the same task afterward. Skip this if you can finish directly."
            ),
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        if context.session_path is None:
            return context.json_result({"error": "Context compression requires a session."})
        requested = context.runtime.request_context_compression(context.session_path)
        return context.json_result(
            {
                "scheduled": requested,
                "message": (
                    "Checkpoint scheduled after this batch. Continue the current task afterward."
                    if requested
                    else "Compression is unavailable. Continue useful work."
                ),
            }
        )
