"""Helpers for extracting JSON objects from model text."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first JSON object found in free-form model text."""

    stripped = text.strip()
    if not stripped:
        return None

    direct = _loads_object(stripped)
    if direct is not None:
        return direct

    fenced = _fenced_json_candidates(stripped)
    for candidate in fenced:
        parsed = _loads_object(candidate)
        if parsed is not None:
            return parsed

    for candidate in _balanced_object_candidates(stripped):
        parsed = _loads_object(candidate)
        if parsed is not None:
            return parsed
    return None


def _loads_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _fenced_json_candidates(text: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    ]


def _balanced_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if escaped:
            escaped = False
            continue
        if character == "\\" and in_string:
            escaped = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : index + 1])
                start = None
    return candidates
