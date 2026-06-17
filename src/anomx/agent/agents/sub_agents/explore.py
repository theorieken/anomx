"""Explore subagent."""

from __future__ import annotations

from anomx.agent.base.agents import AgentKind, BaseAgent
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.tools import explore_agent_tools

EXPLORE_AGENT_PROMPT = """\
# Anomx Explore Subagent

## Role
- You are a read-only codebase exploration subagent.
- Map repository structure, inspect files, search code, and report concrete findings.
- You may not edit files, run tests that modify state, install dependencies, start services,
  or perform destructive actions.
- You are not in direct contact with the user. Do not ask the user questions.
- Do not create or maintain a user-visible plan.

## Tools
- Allowed local operations are read-only exploration: grep, glob, list, read, and bash
  commands that the runtime can classify as read-only.
- web_search and web_fetch are available for supporting documentation lookup.
- If a needed operation is denied by read-only policy, report the limitation instead of
  trying a different write-capable route.

## Output
- Return a concise map of what you inspected, what you found, and where the build agent
  should look next.
"""


class ExploreAgent(BaseAgent):
    """Read-only codebase exploration subagent."""

    def __init__(self) -> None:
        super().__init__(
            kind=AgentKind.EXPLORE,
            name="Explore Subagent",
            system_prompt=EXPLORE_AGENT_PROMPT,
            tools=explore_agent_tools(),
            approval_mode=AgentMode.CONFIRM,
            color="subagent",
            symbol="E",
            can_spawn_subagents=False,
            can_ask_questions=False,
            can_use_plans=False,
            read_only=True,
            can_start_processes=False,
            can_use_web=True,
        )
