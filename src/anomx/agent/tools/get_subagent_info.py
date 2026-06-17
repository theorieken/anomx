"""Get-subagent-info tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema
from anomx.agent.helpers.state import subagent_snapshots


class GetSubagentInfoTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="get_subagent_info",
            description="Inspect the latest outputs from a subagent.",
            parameters=object_schema(
                {"agent_id": {"type": "string", "description": "Subagent id."}},
                ["agent_id"],
            ),
            aliases=("check_agent",),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        if context.session_path is None:
            return context.json_result({"error": "get_subagent_info requires a session."})
        agent_id = str(arguments.get("agent_id", "")).strip()
        if not agent_id:
            return context.json_result({"error": "get_subagent_info requires an agent_id."})

        for snapshot in subagent_snapshots(
            context.runtime.home.read_session_events(context.session_path),
            include_removed=True,
        ):
            if snapshot.agent_id != agent_id:
                continue
            return context.json_result(
                {
                    "agent_id": snapshot.agent_id,
                    "name": snapshot.name,
                    "agent_kind": snapshot.kind,
                    "status": snapshot.status,
                    "statement": snapshot.statement,
                    "prompt": snapshot.prompt,
                    "response": snapshot.response,
                    "error": snapshot.error,
                    "session_path": snapshot.session_path,
                    "context_tokens": snapshot.context_tokens,
                    "context_percent": snapshot.context_percent,
                    "latest_outputs": [
                        {
                            "timestamp": entry.timestamp,
                            "kind": entry.kind,
                            "text": entry.text,
                        }
                        for entry in snapshot.history[-5:]
                    ],
                    "commands": list(snapshot.command_history),
                }
            )
        return context.json_result({"error": "Unknown subagent id."})
