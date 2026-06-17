"""Runtime subagent state value objects."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anomx.agent.base.agents import AgentKind


@dataclass
class SubagentRuntimeState:
    """Mutable in-process state for one asynchronous subagent."""

    agent_id: str
    kind: AgentKind
    name: str
    prompt: str
    status: str
    statement: str
    started_at: str
    runtime: Any = None
    session_path: Path | None = None
    worker: threading.Thread | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    response: str = ""
    error: str = ""
    finished_at: str = ""
    context_tokens: int = 0
    context_percent: int = 0
    command_history: list[dict[str, str]] = field(default_factory=list)
