"""Remind an interactive agent to communicate at the next model boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic


@dataclass
class UserProgress:
    """Track visible text separately from thoughts and individual tool labels."""

    clock: Callable[[], float] = monotonic
    last_text_at: float = field(init=False)
    tools_since_text: int = 0

    def __post_init__(self) -> None:
        self.last_text_at = self.clock()

    def text_received(self, text: str) -> None:
        if text.strip():
            self.last_text_at = self.clock()
            self.tools_since_text = 0

    def tool_completed(self) -> None:
        self.tools_since_text += 1

    def reminder(self) -> str:
        if self.tools_since_text < 5 and self.clock() - self.last_text_at < 45:
            return ""
        return (
            "## User update due\n"
            "The user has not received a recent text update. If the request is answered, "
            "deliver the final result now. Otherwise, before your next useful tool calls, "
            "give one or two sentences with the result so far and the specific remaining "
            "check or blocker. Tool labels and private thoughts do not count as updates. "
            "Do not add investigation or a separate tool call merely to send an update."
        )
