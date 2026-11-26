"""Select JSON response fields without evaluating model-supplied expressions."""

from __future__ import annotations

from typing import Any


def response_field_paths(value: object) -> tuple[str, ...]:
    """Validate a bounded set of literal dotted field paths."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 32:
        raise ValueError("response_fields must contain between 1 and 32 dotted field paths.")
    paths = []
    for field in value:
        if not isinstance(field, str):
            raise ValueError("Each response_fields entry must be a string.")
        field = field.strip()
        parts = field.split(".")
        if (
            not field or len(field) > 128 or len(parts) > 8
            or any(not part or any(char in part for char in "*[]") for part in parts)
        ):
            raise ValueError("Use literal dotted fields without wildcards or array indexes.")
        if field not in paths:
            paths.append(field)
    return tuple(paths)


def project_response_fields(payload: object, paths: tuple[str, ...]) -> dict[str, object]:
    """Preserve containers, nulls and order; report fields absent from every item."""
    tree: dict[str, Any] = {}
    for path in paths:
        node = tree
        parts = path.split(".")
        for part in parts[:-1]:
            if part in node and node[part] is None:
                break  # A selected parent already includes this child.
            node = node.setdefault(part, {})
        else:
            node[parts[-1]] = None

    def select(value: object, fields: dict[str, Any]) -> object:
        if isinstance(value, list):
            return [select(item, fields) for item in value]
        if isinstance(value, dict):
            return {
                key: value[key] if children is None else select(value[key], children)
                for key, children in fields.items() if key in value
            }
        return value

    def contains(value: object, parts: list[str]) -> bool:
        if not parts:
            return True
        if isinstance(value, list):
            return any(contains(item, parts) for item in value)
        return (
            isinstance(value, dict) and parts[0] in value
            and contains(value[parts[0]], parts[1:])
        )

    return {
        "response": select(payload, tree),
        "response_fields": list(paths),
        "missing_response_fields": [
            path for path in paths if not contains(payload, path.split("."))
        ] if payload else [],
    }
