"""Concrete tool classes and tool-set factories for Anomx agents."""

from __future__ import annotations

from anomx.agent.base.tools import BaseTool
from anomx.agent.tools.commands import (
    BashTool,
    CheckCommandStatusTool,
    EndProcessTool,
    KillCommandTool,
    RunCommandTool,
    StartProcessTool,
    WaitTool,
)
from anomx.agent.tools.filesystem import (
    GlobTool,
    GrepTool,
    ListDirectoryTool,
    ReadFileTool,
)
from anomx.agent.tools.interaction import AskQuestionTool
from anomx.agent.tools.planning import (
    CreatePlanTool,
    FinishAnywaysTool,
    RemovePlanTool,
    UpdatePlanTool,
)
from anomx.agent.tools.subagents import (
    GetSubagentInfoTool,
    PromptSubagentTool,
    RemoveSubagentTool,
    StartSubagentTool,
)
from anomx.agent.tools.web import WebFetchTool, WebSearchTool

BUILD_STATEMENT_DESCRIPTION = "Persistent user-visible working message for this tool call."
SUBAGENT_STATEMENT_DESCRIPTION = "Persistent working message for this tool call."


def build_agent_tools() -> tuple[BaseTool, ...]:
    """Return tools available to the primary build-style agents."""

    statement = BUILD_STATEMENT_DESCRIPTION
    return (
        RunCommandTool(statement_description=statement, build_agent=True),
        StartProcessTool(statement_description=statement, build_agent=True),
        EndProcessTool(statement_description=statement),
        AskQuestionTool(statement_description=statement),
        CreatePlanTool(),
        UpdatePlanTool(),
        RemovePlanTool(statement_description=statement),
        FinishAnywaysTool(statement_description=statement),
        StartSubagentTool(statement_description=statement),
        PromptSubagentTool(statement_description=statement),
        RemoveSubagentTool(statement_description=statement),
        GetSubagentInfoTool(),
    )


def general_agent_tools() -> tuple[BaseTool, ...]:
    """Return tools available to general implementation subagents."""

    statement = SUBAGENT_STATEMENT_DESCRIPTION
    return (
        RunCommandTool(statement_description=statement),
        StartProcessTool(statement_description=statement),
        EndProcessTool(statement_description=statement),
        WebSearchTool(statement_description=statement),
        WebFetchTool(statement_description=statement),
    )


def explore_agent_tools() -> tuple[BaseTool, ...]:
    """Return tools available to read-only exploration subagents."""

    statement = SUBAGENT_STATEMENT_DESCRIPTION
    return (
        BashTool(statement_description=statement),
        ReadFileTool(statement_description=statement),
        ListDirectoryTool(statement_description=statement),
        GlobTool(statement_description=statement),
        GrepTool(statement_description=statement),
        WebSearchTool(statement_description=statement),
        WebFetchTool(statement_description=statement),
    )


def command_control_tools() -> tuple[BaseTool, ...]:
    """Return tools for active long-running command calls."""

    return (CheckCommandStatusTool(), KillCommandTool())


def wait_tool(target_description: str) -> BaseTool:
    """Return a wait tool for the provided target description."""

    return WaitTool(target_description=target_description)


__all__ = [
    "BaseTool",
    "BashTool",
    "CheckCommandStatusTool",
    "CreatePlanTool",
    "EndProcessTool",
    "FinishAnywaysTool",
    "GetSubagentInfoTool",
    "GlobTool",
    "GrepTool",
    "KillCommandTool",
    "ListDirectoryTool",
    "PromptSubagentTool",
    "ReadFileTool",
    "RemovePlanTool",
    "RemoveSubagentTool",
    "RunCommandTool",
    "StartProcessTool",
    "StartSubagentTool",
    "UpdatePlanTool",
    "WaitTool",
    "WebFetchTool",
    "WebSearchTool",
    "build_agent_tools",
    "command_control_tools",
    "explore_agent_tools",
    "general_agent_tools",
    "wait_tool",
]
