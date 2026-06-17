"""Get-subagent-info tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema


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
        return context.runtime._get_subagent_info_tool(arguments, context.session_path)
