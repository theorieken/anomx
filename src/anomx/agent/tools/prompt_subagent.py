"""Prompt-subagent tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property


class PromptSubagentTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="prompt_subagent",
            description="Send another prompt to an idle subagent.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "agent_id": {"type": "string", "description": "Subagent id."},
                    "prompt": {"type": "string", "description": "Follow-up prompt."},
                },
                ["statement", "agent_id", "prompt"],
            ),
            aliases=("prompt_agent",),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.runtime._emit_operator_tool_statement(
            self.name, arguments, context.callbacks
        )
        return context.runtime._prompt_subagent_tool(
            arguments, context.session_path, context.callbacks
        )
