"""Class-based agent definitions for the Anomx agent runtime."""

from anomx.agent.agents.main_agents import (
    AUTO_AGENT_PROMPT,
    PLAN_AGENT_PROMPT,
    AutoAgent,
    BuildAgent,
    PlanAgent,
)
from anomx.agent.agents.main_agents.build import BUILD_AGENT_PROMPT
from anomx.agent.agents.sub_agents import ExploreAgent, GeneralAgent
from anomx.agent.agents.sub_agents.explore import EXPLORE_AGENT_PROMPT
from anomx.agent.agents.sub_agents.general import GENERAL_AGENT_PROMPT
from anomx.agent.helpers.utils import (
    AgentKind,
    AgentSpec,
    agent_spec,
    main_agent_kinds,
    next_main_agent_kind,
    parse_agent_kind,
)

__all__ = [
    "AUTO_AGENT_PROMPT",
    "BUILD_AGENT_PROMPT",
    "EXPLORE_AGENT_PROMPT",
    "GENERAL_AGENT_PROMPT",
    "PLAN_AGENT_PROMPT",
    "AgentKind",
    "AgentSpec",
    "AutoAgent",
    "BuildAgent",
    "ExploreAgent",
    "GeneralAgent",
    "PlanAgent",
    "agent_spec",
    "main_agent_kinds",
    "next_main_agent_kind",
    "parse_agent_kind",
]
