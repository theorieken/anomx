"""Observation and sequence views."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ObservationSet": "anomx.data.views.observations",
    "BaseObservationView": "anomx.data.views.core",
    "TabularView": "anomx.data.views.core",
    "SequenceView": "anomx.data.views.core",
    "ObservationWindow": "anomx.data.views.core",
    "WindowView": "anomx.data.views.core",
    "TimeSeriesView": "anomx.data.views.core",
    "GraphView": "anomx.data.views.core",
    "TimeSeriesBatch": "anomx.data.views.timeseries",
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
