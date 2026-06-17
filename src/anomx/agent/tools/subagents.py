"""Subagent orchestration tools."""

from __future__ import annotations

from anomx.agent.base.tools import BaseTool, object_schema, statement_property


class StartSubagentTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="start_subagent",
            description="Start an asynchronous subagent.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "agent_kind": {
                        "type": "string",
                        "enum": ["general", "explore"],
                        "description": "Kind of subagent to start.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Short display name for the subagent.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Complete task prompt for the subagent.",
                    },
                },
                ["statement", "agent_kind", "name", "prompt"],
            ),
        )


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
        )


class RemoveSubagentTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="remove_subagent",
            description="Remove a subagent from prompt context and UI.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "agent_id": {"type": "string", "description": "Subagent id."},
                },
                ["statement", "agent_id"],
            ),
        )


class GetSubagentInfoTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="get_subagent_info",
            description="Inspect the latest outputs from a subagent.",
            parameters=object_schema(
                {"agent_id": {"type": "string", "description": "Subagent id."}},
                ["agent_id"],
            ),
        )
