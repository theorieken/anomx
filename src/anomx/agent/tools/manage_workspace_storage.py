"""Host-controlled inspection and explicit removal of disposable workspace files."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext


class ManageWorkspaceStorageTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="manage_workspace_storage",
            description=(
                "Inspect storage usage and bounded batches of disposable temporary files/cache. "
                "Use inspect first, then delete only candidate IDs whose files are no longer "
                "needed, "
                "with a reason. Candidates exclude durable work and other active chats. Changed "
                "files "
                "are skipped. Repeat inspection if more cleanup is needed. No arbitrary paths or "
                "shell deletion."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["status", "inspect", "delete"]},
                    "candidate_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
                    "reason": {"type": "string"},
                },
                "required": ["action", "candidate_ids", "reason"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        handler = context.runtime.workspace_storage
        if handler is None:
            return context.json_result({"error": "Managed workspace storage is unavailable."})
        if context.runtime._running_command_states() or context.runtime._running_subagent_states():
            return context.json_result(
                {"error": "Wait for active commands and subagents before managing storage."}
            )
        result = handler(arguments)
        if isinstance(result.get("usage"), dict):
            context.runtime._workspace_usage = result["usage"]
        return context.json_result(result)
