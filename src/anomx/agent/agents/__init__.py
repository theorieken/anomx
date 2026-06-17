"""Agent kind definitions for the Anomx agent runtime."""

from anomx.agent.agents.build import BUILD_AGENT_PROMPT
from anomx.agent.agents.explore import EXPLORE_AGENT_PROMPT
from anomx.agent.agents.general import GENERAL_AGENT_PROMPT
from anomx.agent.agents.kinds import AgentKind, AgentSpec, agent_spec

__all__ = [
    "BUILD_AGENT_PROMPT",
    "EXPLORE_AGENT_PROMPT",
    "GENERAL_AGENT_PROMPT",
    "AgentKind",
    "AgentSpec",
    "agent_spec",
]
