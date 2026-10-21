"""Explicit chat feature commands and terminal response pacing."""

from __future__ import annotations

import re

FEATURE_COMMANDS = {
    "subagents": ("subagents_enabled", "Delegate work to agents in their own folders"),
    "goal": ("goal_enabled", "Keep working from GOAL.md until the goal is complete"),
}

_COMMAND = re.compile(r"(?<!\S)/(subagents|goal)(:off)?(?=\s|$)", re.IGNORECASE)
_CODE = re.compile(r"```[\s\S]*?(?:```|$)|`[^`\n]*`")


def parse_feature_commands(message: str) -> tuple[str, dict[str, bool]]:
    """Consume standalone mode commands, leaving quoted code and file paths intact."""
    code_ranges = tuple((match.start(), match.end()) for match in _CODE.finditer(message))
    features: dict[str, bool] = {}

    def consume(match: re.Match[str]) -> str:
        if any(start <= match.start() < end for start, end in code_ranges):
            return match.group()
        features[FEATURE_COMMANDS[match.group(1).lower()][0]] = not bool(match.group(2))
        return ""

    return _COMMAND.sub(consume, message).strip(), features


def response_reveal_frames(message: str, width: int) -> tuple[str, ...]:
    """Return complete visual-line prefixes without changing the Markdown source."""
    if not message:
        return ()
    width = max(1, width)
    boundaries: list[int] = []
    column = 0
    for index, character in enumerate(message, 1):
        column += 1
        if character == "\n" or column >= width:
            boundaries.append(index)
            column = 0
    if not boundaries or boundaries[-1] != len(message):
        boundaries.append(len(message))
    # Bound repaint work for large tables, logs, and generated documents.
    stride = max(1, (len(boundaries) + 59) // 60)
    selected = boundaries[stride - 1 :: stride]
    if not selected or selected[-1] != len(message):
        selected.append(len(message))
    return tuple(message[:boundary] for boundary in selected)


def completed_response_lines(message: str, width: int) -> str:
    """Buffer the unfinished visual line while a provider is still streaming."""
    width = max(1, width)
    boundary = 0
    column = 0
    for index, character in enumerate(message, 1):
        column += 1
        if character == "\n" or column >= width:
            boundary = index
            column = 0
    return message[:boundary]
