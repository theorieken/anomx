"""Shared utility helpers for the Anomx agent package."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from anomx.agent.base.agents import AgentKind, BaseAgent

AgentSpec = BaseAgent


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp suitable for JSONL events."""

    return datetime.now(tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def session_id_from_path(session_path: Path) -> str:
    """Extract a session identifier from a session transcript path."""

    stem = session_path.stem
    if stem.startswith("rollout-"):
        return stem.rsplit("-", 1)[-1]
    parts = stem.split("-", 2)
    if len(parts) >= 3:
        return parts[-1]
    return stem


def _new_agents() -> dict[AgentKind, BaseAgent]:
    from anomx.agent.agents.main_agents import (
        AutomaticAgent,
        AutonomousAgent,
        PlanAgent,
        StandardAgent,
    )
    from anomx.agent.agents.sub_agents import ExploreAgent, GeneralAgent

    standard = StandardAgent()
    automatic = AutomaticAgent()
    autonomous = AutonomousAgent()
    return {
        AgentKind.STANDARD: standard,
        AgentKind.AUTOMATIC: automatic,
        AgentKind.AUTONOMOUS: autonomous,
        AgentKind.BUILD: standard,
        AgentKind.AUTO: automatic,
        AgentKind.PLAN: PlanAgent(),
        AgentKind.GENERAL: GeneralAgent(),
        AgentKind.EXPLORE: ExploreAgent(),
    }


def parse_agent_kind(value: object, default: AgentKind = AgentKind.STANDARD) -> AgentKind:
    """Parse stored config/session values into an agent kind."""

    aliases = {
        "operator": AgentKind.STANDARD,
        "build": AgentKind.STANDARD,
        "standard": AgentKind.STANDARD,
        "worker": AgentKind.GENERAL,
        "auto": AgentKind.AUTOMATIC,
        "automatic": AgentKind.AUTOMATIC,
        "autonomous": AgentKind.AUTONOMOUS,
        "full_control": AgentKind.AUTONOMOUS,
        "fullcontrol": AgentKind.AUTONOMOUS,
        "planning": AgentKind.PLAN,
    }
    normalized = (
        value.value if isinstance(value, AgentKind) else str(value or "")
    ).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in aliases:
        return aliases[normalized]
    try:
        return AgentKind(normalized)
    except ValueError:
        return default


def agent_spec(kind: AgentKind | str | object) -> BaseAgent:
    """Return a fresh agent object for a kind."""

    return _new_agents()[parse_agent_kind(kind)]


def main_agent_kinds() -> tuple[AgentKind, ...]:
    """Return the Shift+Tab cycle for user-facing main agents."""

    return (AgentKind.STANDARD, AgentKind.AUTOMATIC, AgentKind.AUTONOMOUS)


def next_main_agent_kind(kind: AgentKind | str | object) -> AgentKind:
    """Return the next main agent kind."""

    current = parse_agent_kind(kind)
    order = main_agent_kinds()
    if current not in order:
        current = AgentKind.STANDARD
    return order[(order.index(current) + 1) % len(order)]


__all__ = [
    "AgentKind",
    "AgentSpec",
    "BaseAgent",
    "agent_spec",
    "main_agent_kinds",
    "next_main_agent_kind",
    "parse_agent_kind",
    "session_id_from_path",
    "utc_now_iso",
]
