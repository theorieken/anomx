"""Bounded, secret-redacted action details for approval evaluation."""

from __future__ import annotations

import json
import re


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
