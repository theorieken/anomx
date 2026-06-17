"""Automatic build agent."""

from __future__ import annotations

from anomx.agent.agents.main_agents.build import BUILD_AGENT_PROMPT
from anomx.agent.base.agents import AgentKind, BaseAgent
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.tools import build_agent_tools

AUTO_AGENT_PROMPT = BUILD_AGENT_PROMPT.replace(
    "# Anomx Build Agent",
    "# Anomx Auto Agent",
    1,
).replace(
    "- Confirm Mode does not mean \"ask the user in prose before doing work.\" If a command needs\n"
    "  approval, call run_command anyway; the command approval UI will ask the user at the\n"
    "  moment approval is required.",
    "- Automatic Mode may run known read, compute, execute, install, and file-modifying\n"
    "  commands inside the trusted workspace without first asking the user. Unknown,\n"
    "  structurally ambiguous, or serious host-control commands still go through the\n"
    "  command approval UI.",
)


class AutoAgent(BaseAgent):
    """Primary build-capable agent using automatic command approval policy."""

    def __init__(self) -> None:
        super().__init__(
            kind=AgentKind.AUTO,
            name="Auto Agent",
            system_prompt=AUTO_AGENT_PROMPT,
            tools=build_agent_tools(),
            approval_mode=AgentMode.AUTO,
            color="accent",
            symbol="Λ",
            can_spawn_subagents=True,
            can_ask_questions=True,
            can_use_plans=True,
            read_only=False,
            can_start_processes=True,
            can_use_web=True,
        )
