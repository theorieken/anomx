"""Data source implementations."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DCacheDataset": "anomx.data.sources.mounted",
    "DaqReadError": "anomx.data.sources.daq",
    "DatasetSourceChanged": "anomx.data.sources.mounted",
    "LocalFSConnector": "anomx.data.sources.local_fs",
    "MountedDataset": "anomx.data.sources.mounted",
    "MountedSource": "anomx.data.sources.mounted",
    "PlatformDatasetClient": "anomx.data.sources.platform",
    "iter_daq_records": "anomx.data.sources.daq",
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
