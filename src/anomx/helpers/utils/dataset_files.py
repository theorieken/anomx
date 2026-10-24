"""Validation for portable, declarative file dataset selections."""

from __future__ import annotations

from typing import Any


def file_selection(current: dict[str, Any], **arguments: Any) -> dict[str, Any]:  # noqa: ANN401
    """Validate half-open file/row/array bounds without reading observations."""
    if set(current) & {"start", "end", "frequency", "aggregation"}:
        raise ValueError("File selections cannot be combined with time aggregation.")
    result = dict(current)
    for name, value in arguments.items():
        if value is None:
            continue
        if name in {"variables", "columns"}:
            if (
                isinstance(value, str)
                or not value
                or len(value) > 100
                or any(not isinstance(item, str) or not item or len(item) > 4096 for item in value)
            ):
                raise ValueError(f"{name} must contain between 1 and 100 names.")
            result[name] = list(dict.fromkeys(value))
        elif name.startswith("path_"):
            if not isinstance(value, str) or not value.startswith("/") or ".." in value.split("/"):
                raise ValueError("Path bounds must be absolute integration paths.")
            result[name] = value
        else:
            if (
                type(value) is not int
                or not (1 if name.endswith("step") else 0) <= value <= 2**53 - 1
            ):
                raise ValueError(f"{name} must be a nonnegative integer; steps must be positive.")
            result[name] = value
    for prefix in ("row", "sample"):
        if prefix + "_end" in result and result[prefix + "_end"] <= result.get(
            prefix + "_start", 0
        ):
            raise ValueError(f"{prefix}_end must be greater than {prefix}_start.")
    if (
        result.get("path_start")
        and result.get("path_end")
        and result["path_start"] >= result["path_end"]
    ):
        raise ValueError("path_end must be greater than path_start.")
    return result
