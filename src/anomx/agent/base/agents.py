"""Object-oriented agent primitives for Anomx."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from anomx.agent.base.tools import BaseTool
from anomx.agent.helpers.mode import AgentMode


class AgentKind(StrEnum):
    """Supported Anomx agent kinds."""

    BUILD = "build"
    AUTO = "auto"
    PLAN = "plan"
    GENERAL = "general"
    EXPLORE = "explore"


@dataclass(frozen=True)
class BaseAgent:
    """Base class for class-based Anomx agents."""

    kind: AgentKind
    name: str
    system_prompt: str
    tools: tuple[BaseTool, ...]
    approval_mode: AgentMode = AgentMode.CONFIRM
    color: str = "accent"
    symbol: str = "Ω"
    can_spawn_subagents: bool = False
    can_ask_questions: bool = False
    can_use_plans: bool = False
    read_only: bool = False
    can_start_processes: bool = False
    can_use_web: bool = True

    @property
    def prompt(self) -> str:
        """Compatibility alias for older runtime code."""

        return self.system_prompt

    @property
    def prompt_hint(self) -> str:
        """Return compact prompt-bar text for this agent."""

        return f"{self.symbol}  {self.name} ({self.approval_mode.label}, shift+tab agent)"

    def tool_definitions(self) -> list[dict[str, object]]:
        """Return function definitions for this agent's assigned tools."""

        return [tool.definition() for tool in self.tools]

    def tool_for(self, name: str) -> BaseTool | None:
        """Return the assigned tool that handles a requested tool name."""

        for tool in self.tools:
            if tool.handles(name):
                return tool
        return None

    def with_approval_mode(self, approval_mode: AgentMode) -> BaseAgent:
        """Return a copy of this agent using a different approval policy."""

        return BaseAgent(
            kind=self.kind,
            name=self.name,
            system_prompt=self.system_prompt,
            tools=self.tools,
            approval_mode=approval_mode,
            color=self.color,
            symbol=self.symbol,
            can_spawn_subagents=self.can_spawn_subagents,
            can_ask_questions=self.can_ask_questions,
            can_use_plans=self.can_use_plans,
            read_only=self.read_only,
            can_start_processes=self.can_start_processes,
            can_use_web=self.can_use_web,
        )
