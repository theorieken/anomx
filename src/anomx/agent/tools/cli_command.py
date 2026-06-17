"""CLI command tool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property

CliCommandAccess = Literal["read", "write"]


@dataclass(frozen=True)
class CliCommandTool(BaseTool):
    """Run a CLI command through the runtime command policy engine."""

    access: CliCommandAccess = "write"

    def __init__(
        self,
        *,
        statement_description: str,
        name: str = "run_command",
        description: str | None = None,
        access: CliCommandAccess = "write",
        aliases: tuple[str, ...] = (),
        build_agent: bool = False,
    ) -> None:
        command_description = (
            "A read-only shell command inside the trusted workspace."
            if access == "read"
            else (
                "A single CLI command, for example 'ls -la'. Shell operators and "
                "redirection may be used when necessary; paths must resolve inside "
                "the trusted workspace root."
                if build_agent
                else "A single CLI command inside the trusted workspace."
            )
        )
        object.__setattr__(self, "access", access)
        super().__init__(
            name=name,
            description=description
            or (
                "Run a read-only CLI command inside the trusted workspace."
                if access == "read"
                else "Run a CLI command inside the trusted workspace."
            ),
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "command": {
                        "type": "string",
                        "description": command_description,
                    },
                },
                ["statement", "command"],
            ),
            aliases=aliases,
        )

    def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> str:
        """Execute a CLI command using the configured access policy."""

        return context.runtime._execute_cli_command_tool(
            arguments,
            context.callbacks,
            context.session_path,
            read_only=self.access == "read",
            tool_name=self.name,
        )
