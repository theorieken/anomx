"""Agent kind registry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from anomx.agent.agents.build import BUILD_AGENT_PROMPT
from anomx.agent.agents.explore import EXPLORE_AGENT_PROMPT
from anomx.agent.agents.general import GENERAL_AGENT_PROMPT
from anomx.agent.agents.scout import SCOUT_AGENT_PROMPT


class AgentKind(StrEnum):
    """Supported Anomx agent kinds."""

    BUILD = "build"
    GENERAL = "general"
    EXPLORE = "explore"
    SCOUT = "scout"


@dataclass(frozen=True)
class AgentSpec:
    """Static behavior for one agent kind."""

    kind: AgentKind
    prompt: str
    can_spawn_subagents: bool = False
    can_ask_questions: bool = False
    can_use_plans: bool = False
    read_only: bool = False
    can_start_processes: bool = False
    can_use_web: bool = True


AGENT_SPECS: dict[AgentKind, AgentSpec] = {
    AgentKind.BUILD: AgentSpec(
        AgentKind.BUILD,
        BUILD_AGENT_PROMPT,
        can_spawn_subagents=True,
        can_ask_questions=True,
        can_use_plans=True,
        read_only=False,
        can_start_processes=True,
        can_use_web=True,
    ),
    AgentKind.GENERAL: AgentSpec(
        AgentKind.GENERAL,
        GENERAL_AGENT_PROMPT,
        read_only=False,
        can_start_processes=True,
        can_use_web=True,
    ),
    AgentKind.EXPLORE: AgentSpec(
        AgentKind.EXPLORE,
        EXPLORE_AGENT_PROMPT,
        read_only=True,
        can_start_processes=False,
        can_use_web=True,
    ),
    AgentKind.SCOUT: AgentSpec(
        AgentKind.SCOUT,
        SCOUT_AGENT_PROMPT,
        read_only=True,
        can_start_processes=False,
        can_use_web=True,
    ),
}


def agent_spec(kind: AgentKind | str) -> AgentSpec:
    """Return the spec for a kind, defaulting unknown aliases conservatively."""

    normalized = str(kind or AgentKind.BUILD).strip().lower()
    if normalized == "operator":
        normalized = AgentKind.BUILD.value
    if normalized == "worker":
        normalized = AgentKind.GENERAL.value
    try:
        parsed = AgentKind(normalized)
    except ValueError:
        parsed = AgentKind.BUILD
    return AGENT_SPECS[parsed]
