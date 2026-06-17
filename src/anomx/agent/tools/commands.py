"""Command and process tools."""

from __future__ import annotations

from anomx.agent.base.tools import BaseTool, object_schema, statement_property


class RunCommandTool(BaseTool):
    def __init__(self, *, statement_description: str, build_agent: bool = False) -> None:
        command_description = (
            "A single CLI command, for example 'ls -la'. Shell operators and "
            "redirection may be used when necessary; paths must resolve inside "
            "the trusted workspace root."
            if build_agent
            else "A single CLI command inside the trusted workspace."
        )
        super().__init__(
            name="run_command",
            description=(
                "Run a CLI command for operator inspection or validation."
                if build_agent
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
        )


class BashTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="bash",
            description="Run a read-only shell command inside the trusted workspace.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "command": {
                        "type": "string",
                        "description": "A command that must be classified read-only.",
                    },
                },
                ["statement", "command"],
            ),
        )


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


class EndProcessTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="end_process",
            description="End a running async CLI process.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "process_id": {
                        "type": "string",
                        "description": "Async process id to end.",
                    },
                },
                ["statement", "process_id"],
            ),
        )


class CheckCommandStatusTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="check_command_status",
            description=(
                "Check a currently running long-running command tool call and read "
                "its current CLI output."
            ),
            parameters=object_schema(
                {
                    "command_id": {
                        "type": "string",
                        "description": "Long-running command id to inspect.",
                    },
                },
                ["command_id"],
            ),
        )


class KillCommandTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="kill_command",
            description="Kill a currently running long-running command tool call.",
            parameters=object_schema(
                {
                    "command_id": {
                        "type": "string",
                        "description": "Long-running command id to kill.",
                    },
                },
                ["command_id"],
            ),
        )


class WaitTool(BaseTool):
    def __init__(self, *, target_description: str) -> None:
        super().__init__(
            name="wait",
            description=f"Wait up to 60 seconds for {target_description}.",
            parameters=object_schema({}, []),
        )
