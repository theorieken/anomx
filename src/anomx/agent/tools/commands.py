"""Compatibility exports for command tools."""

from anomx.agent.tools.check_command_status import CheckCommandStatusTool
from anomx.agent.tools.cli_command import CliCommandTool
from anomx.agent.tools.end_process import EndProcessTool
from anomx.agent.tools.kill_command import KillCommandTool
from anomx.agent.tools.start_process import StartProcessTool
from anomx.agent.tools.wait import WaitTool


class RunCommandTool(CliCommandTool):
    """Compatibility wrapper for the write-capable CLI command tool."""

    def __init__(self, *, statement_description: str, build_agent: bool = False) -> None:
        super().__init__(
            statement_description=statement_description,
            aliases=("run_cli_command",),
            build_agent=build_agent,
        )


class BashTool(CliCommandTool):
    """Compatibility wrapper for the read-only bash tool."""

    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            statement_description=statement_description,
            name="bash",
            description="Run a read-only shell command inside the trusted workspace.",
            access="read",
        )

__all__ = [
    "BashTool",
    "CheckCommandStatusTool",
    "CliCommandTool",
    "EndProcessTool",
    "KillCommandTool",
    "RunCommandTool",
    "StartProcessTool",
    "WaitTool",
]
