"""Start-process tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property


class StartProcessTool(BaseTool):
    def __init__(self, *, statement_description: str, build_agent: bool = False) -> None:
        super().__init__(
            name="start_process",
            description="Start a long-running async CLI process.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "command": {
                        "type": "string",
                        "description": (
                            "Long-running CLI command, for example 'npm run dev'. "
                            "It continues after the agent turn until ended."
                            if build_agent
                            else "Long-running CLI command."
                        ),
                    },
                },
                ["statement", "command"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.runtime._emit_operator_tool_statement(
            self.name, arguments, context.callbacks
        )
        if not context.runtime.agent_spec.can_start_processes:
            return context.json_result(
                {"error": "This agent kind cannot start async processes."}
            )
        return context.runtime._start_process_tool(
            arguments, context.session_path, context.callbacks
        )
