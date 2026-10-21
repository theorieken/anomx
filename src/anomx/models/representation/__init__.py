"""Representation models."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {"IsolationForestModel": "anomx.models.representation.isolation_forest"}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:  # noqa: ANN401 - lazy public API
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
