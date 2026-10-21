"""Dataset facades, data contracts, and optional sources."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AnomxConnectionError": "anomx.integrations.platform.datasets",
    "AnomxDataset": "anomx.integrations.platform.datasets",
    "BaseConnector": "anomx.data.base.connector",
    "BaseObservationView": "anomx.data.base",
    "ChannelMetadata": "anomx.data.datasets",
    "DCacheDataset": "anomx.data.sources.mounted",
    "DataCharacteristics": "anomx.data.base",
    "Dataset": "anomx.data.dataset",
    "DatasetAdapter": "anomx.data.base",
    "DatasetClient": "anomx.data.dataset",
    "DatasetKind": "anomx.data.dataset",
    "DatasetMode": "anomx.data.dataset",
    "DatasetSourceChanged": "anomx.data.sources.mounted",
    "GraphView": "anomx.data.base",
    "Grouping": "anomx.data.base",
    "Hierarchy": "anomx.data.base",
    "LocalFSConnector": "anomx.data.sources",
    "Modality": "anomx.data.base",
    "MountedDataset": "anomx.data.sources.mounted",
    "MountedSource": "anomx.data.sources.mounted",
    "ObservationSet": "anomx.data.base",
    "ObservationUnit": "anomx.data.base",
    "ObservationWindow": "anomx.data.base",
    "Ordering": "anomx.data.base",
    "ProcessingPipeline": "anomx.data.processing",
    "RecordsAdapter": "anomx.data.base",
    "RelationSet": "anomx.data.base",
    "Resample": "anomx.data.processing",
    "SequenceView": "anomx.data.base",
    "Structure": "anomx.data.base",
    "TabularView": "anomx.data.base",
    "TemporalSemantics": "anomx.data.base",
    "TimeSeriesBatch": "anomx.data.views",
    "TimeSeriesBatchAdapter": "anomx.data.base",
    "TimeSeriesDataset": "anomx.data.datasets",
    "TimeSeriesView": "anomx.data.base",
    "Topology": "anomx.data.base",
    "Transform": "anomx.data.base.transform",
    "WindowView": "anomx.data.base",
    "make_sine_anomaly_dataset": "anomx.data.loaders",
    "read_platform_connection": "anomx.integrations.platform.datasets",
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
