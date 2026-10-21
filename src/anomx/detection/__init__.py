"""Scores, decisions, and composable anomaly detection pipelines."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AbsoluteErrorScorer": "anomx.detection.scorers.absolute_error",
    "AdaptiveThresholdDetector": "anomx.detection.base.detectors",
    "AnomalyEvent": "anomx.detection.results",
    "AnomalyEvents": "anomx.detection.results",
    "AnomalyPipeline": "anomx.detection.pipeline",
    "AnomalyScores": "anomx.detection.results",
    "BaseDetector": "anomx.detection.base",
    "BaseScorer": "anomx.detection.base",
    "BoundaryScorer": "anomx.detection.base.scorers",
    "ChangePointDetector": "anomx.detection.base.detectors",
    "CompositeScorer": "anomx.detection.base.scorers",
    "Detector": "anomx.detection.base.detectors",
    "DirectDataScorer": "anomx.detection.base.scorers",
    "EventAggregationDetector": "anomx.detection.base.detectors",
    "LikelihoodScorer": "anomx.detection.base.scorers",
    "PipelineValidationError": "anomx.detection.pipeline",
    "QuantileThresholdDetector": "anomx.detection.detectors.quantile",
    "ReconstructionScorer": "anomx.detection.base.scorers",
    "RepresentationScorer": "anomx.detection.base.scorers",
    "ResidualScorer": "anomx.detection.base.scorers",
    "ScoreLevel": "anomx.detection.results",
    "Scorer": "anomx.detection.base.scorers",
    "StaticThresholdDetector": "anomx.detection.base.detectors",
    "StatisticalDetector": "anomx.detection.base.detectors",
    "ThresholdDetector": "anomx.detection.detectors.threshold",
    "ZScoreScorer": "anomx.detection.scorers.zscore",
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
