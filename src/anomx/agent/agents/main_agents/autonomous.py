"""Autonomous main agent."""

from __future__ import annotations

from anomx.agent.agents.main_agents.build import STANDARD_AGENT_PROMPT
from anomx.agent.base.agents import AgentKind, BaseAgent
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.tools import build_agent_tools

AUTONOMOUS_AGENT_PROMPT = STANDARD_AGENT_PROMPT.replace(
    "# Anomx Standard Agent",
    "# Anomx Autonomous Agent",
    1,
).replace(
    "- Standard mode does not mean \"ask the user in prose before doing work.\" If a "
    "command needs\n"
    "  approval, call run_command anyway; the command approval UI will ask the user at the\n"
    "  moment approval is required.",
    "- Autonomous mode may run valid commands automatically. Very severe host-control\n"
    "  commands such as sudo, reboot, shutdown, killall, diskutil, mount, and systemctl\n"
    "  remain blocked by command policy. Do not ask for approval in prose.",
)


class AutonomousAgent(BaseAgent):
    """Primary main agent that runs commands unless they are very severe."""

    def __init__(self) -> None:
        super().__init__(
            kind=AgentKind.AUTONOMOUS,
            name="Autonomous",
            system_prompt=AUTONOMOUS_AGENT_PROMPT,
            tools=build_agent_tools(),
            approval_mode=AgentMode.AUTONOMOUS,
            color="danger",
            symbol="Δ",
            can_spawn_subagents=True,
            can_ask_questions=True,
            can_use_plans=True,
            read_only=False,
            can_start_processes=True,
            can_use_web=True,
        )
