"""Time-based presentation of live activity, independent of runtime execution."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

# Runtime phases that describe the model loop itself rather than what the agent is doing.
TRANSIENT_STATUSES = frozenset({
    "Thinking",
    "Writing",
    "Executing",
    "Context compression",
    "Context optimization",
    "Context Optimization",
    "Waiting",
    "Loading model",
    "Starting Sandbox",
    "Pulling sandbox image",
    "Starting sandbox container",
    "Sandbox startup completed",
    "Evaluating project size",
    "Reconnecting",
})


def is_activity_statement(status_text: str) -> bool:
    """Return whether a runtime status names concrete work worth showing and keeping."""

    normalized = status_text.strip()
    if not normalized or normalized.startswith("Reconnecting "):
        return False
    return normalized not in TRANSIENT_STATUSES


@dataclass
class ActivityPresentation:
    """Show every new thought briefly, retaining specific tool statements between calls."""

    phase: str = "Thinking"
    latest: tuple[str, bool] | None = None
    current: tuple[str, bool] | None = None
    since: float = 0.0
    pending: deque[tuple[str, bool]] = field(default_factory=deque)
    running: dict[str, str] = field(default_factory=dict)

    def push(self, text: str, *, thought: bool = False) -> None:
        label = (text, thought)
        if not thought and self.latest == label:
            return
        self.latest = label
        self.pending.append(label)

    def display(self, now: float) -> str:
        # Runtime events are never delayed: only their visible labels dwell.
        if self.current is not None and now - self.since < 1.0:
            return self.current[0]
        if self.pending:
            self.current = self.pending.popleft()
            self.since = now
            return self.current[0]
        self.current = None
        if self.running:
            return next(reversed(self.running.values()))
        if self.latest is not None and not self.latest[1]:
            return self.latest[0]
        return self.phase
