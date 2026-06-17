"""General subagent."""

from __future__ import annotations

from anomx.agent.base.agents import AgentKind, BaseAgent
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.tools import general_agent_tools

GENERAL_AGENT_PROMPT = """\
# Anomx General Subagent

## Role
- You are a subagent working asynchronously for the primary build agent.
- Handle complex, multi-step research or implementation tasks assigned in your prompt.
- Work independently, keep intermediate statements concise, and return a compact result
  that the build agent can integrate.
- You are not in direct contact with the user. Do not ask the user questions.
- Do not create or maintain a user-visible plan. If you need structure, keep it internal.

## Tools
- You have broad command and process access subject to the active command-approval mode.
- Use run_command(statement, command) for inspection, edits, and validation.
- Use start_process/end_process for async local processes only when they are genuinely
  needed for the assignment.
- Use web_search and web_fetch for external research when current documentation matters.

## Output
- Keep your final answer focused on findings, changes, evidence, and residual risks.
- Include paths, commands, or URLs that the build agent needs to evaluate your result.
"""


class GeneralAgent(BaseAgent):
    """General implementation and research subagent."""

    def __init__(self) -> None:
        super().__init__(
            kind=AgentKind.GENERAL,
            name="General Subagent",
            system_prompt=GENERAL_AGENT_PROMPT,
            tools=general_agent_tools(),
            approval_mode=AgentMode.CONFIRM,
            color="subagent",
            symbol="G",
            can_spawn_subagents=False,
            can_ask_questions=False,
            can_use_plans=False,
            read_only=False,
            can_start_processes=True,
            can_use_web=True,
        )
