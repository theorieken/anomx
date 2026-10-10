"""Bounded, secret-redacted action details for approval evaluation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


def approval_user_context(messages: Sequence[Mapping[str, Any]]) -> str:
    """Keep every original request and steering message verbatim for approval assessment."""

    user_messages = [
        str(message.get("content") or "").strip()
        for message in messages
        if message.get("role") == "user" and str(message.get("content") or "").strip()
    ]
    if not user_messages:
        return ""

    # Dropping an earlier message could hide a restriction or an authorization.
    # If the evaluator cannot fit the exact prompts, it must fail closed instead.
    return "\n\nLater user message:\n".join(user_messages)



def approval_action_details(value: object) -> str:
    """Retain targets and changes without forwarding credential fields."""

    def redact(item: object) -> object:
        if isinstance(item, dict):
            return {
                str(key): "[redacted]"
                if re.search(
                    r"password|secret|token|authorization|credential|api[_-]?key", str(key), re.I
                )
                else redact(entry)
                for key, entry in item.items()
            }
        if isinstance(item, list):
            return [redact(entry) for entry in item]
        return item

    serialized = json.dumps(redact(value), ensure_ascii=False, default=str)
    return (
        serialized
        if len(serialized) <= 16000
        else serialized[:16000]
        + "\n[Action details truncated; do not assume omitted effects are authorized.]"
    )
