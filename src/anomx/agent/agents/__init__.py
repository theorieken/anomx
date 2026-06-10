"""Agent kind definitions for the Anomx agent runtime."""

from anomx.agent.agents.build import BUILD_AGENT_PROMPT
from anomx.agent.agents.explore import EXPLORE_AGENT_PROMPT
from anomx.agent.agents.general import GENERAL_AGENT_PROMPT
from anomx.agent.agents.kinds import AgentKind, AgentSpec, agent_spec
from anomx.agent.agents.scout import SCOUT_AGENT_PROMPT

__all__ = [
    "BUILD_AGENT_PROMPT",
    "EXPLORE_AGENT_PROMPT",
    "GENERAL_AGENT_PROMPT",
    "SCOUT_AGENT_PROMPT",
    "AgentKind",
    "AgentSpec",
    "agent_spec",
]
