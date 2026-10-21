"""Native anomaly scoring and decision contracts."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AdaptiveThresholdDetector": "anomx.detection.base.detectors",
    "BaseDetector": "anomx.detection.base.detectors",
    "BaseScorer": "anomx.detection.base.scorers",
    "BoundaryScorer": "anomx.detection.base.scorers",
    "ChangePointDetector": "anomx.detection.base.detectors",
    "CompositeScorer": "anomx.detection.base.scorers",
    "Detector": "anomx.detection.base.detectors",
    "DirectDataScorer": "anomx.detection.base.scorers",
    "EventAggregationDetector": "anomx.detection.base.detectors",
    "LikelihoodScorer": "anomx.detection.base.scorers",
    "ReconstructionScorer": "anomx.detection.base.scorers",
    "RepresentationScorer": "anomx.detection.base.scorers",
    "ResidualScorer": "anomx.detection.base.scorers",
    "Scorer": "anomx.detection.base.scorers",
    "StaticThresholdDetector": "anomx.detection.base.detectors",
    "StatisticalDetector": "anomx.detection.base.detectors",
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
