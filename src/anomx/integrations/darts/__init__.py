"""Optional Darts-native types and explicit adapters to Anomx contracts."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DartsForecastingModel": "anomx.integrations.darts.adapters",
    "is_darts_available": "anomx.integrations.darts.adapters",
    "Dataset": "anomx.integrations.darts.datasets",
    "DartsDataset": "anomx.integrations.darts.datasets",
    "DartsModel": "anomx.integrations.darts.models",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:  # noqa: ANN401 - lazy public API
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
