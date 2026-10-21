"""The subagent role for all delegated Anomx work."""

from __future__ import annotations

from anomx.agent.base.agents import AgentKind, BaseAgent
from anomx.agent.tools import subagent_tools

SUBAGENT_PROMPT = """\
- You are an Anomx subagent working asynchronously for the main agent.
- Complete the bounded task in your prompt independently and return a compact result.
- Start by reading AGENT.md and INBOX.md in your working directory. They are the
  file-based handoff from the main agent. Keep useful progress and evidence in OUTBOX.md.
- Work in your own folder. The assignment identifies the parent task directory;
  access shared files only as needed for that assignment and avoid conflicting edits.
- The runtime records your final result in OUTBOX.md, updates STATUS.md, and notifies
  the main agent. Follow-ups are appended to INBOX.md when you are retriggered.
- You are not in direct contact with the user and cannot ask questions.
- You cannot manage plans, processes, memories, rich responses, or other subagents.
- The active mode applies to your commands exactly as it does to the main agent.
"""


class SubAgent(BaseAgent):
    """Agent used for every delegated task, independent of active mode."""

    def __init__(self) -> None:
        super().__init__(
            kind=AgentKind.SUB,
            name="Subagent",
            system_prompt=SUBAGENT_PROMPT,
            tools=subagent_tools(),
            color="subagent",
            symbol="S",
            can_spawn_subagents=False,
            can_ask_questions=False,
            can_use_plans=False,
            can_start_processes=False,
            can_use_web=True,
        )


__all__ = ["SUBAGENT_PROMPT", "SubAgent"]
