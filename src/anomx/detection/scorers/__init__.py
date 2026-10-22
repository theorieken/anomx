"""Native scorers; Darts adapters live in anomx.integrations.darts."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AbsoluteErrorScorer": "anomx.detection.scorers.absolute_error",
    "AnomalyScorer": "anomx.detection.scorers.timeseries",
    "BaseScorer": "anomx.detection.base",
    "CauchyNLLScorer": "anomx.detection.scorers.timeseries",
    "DifferenceScorer": "anomx.detection.scorers.timeseries",
    "ExponentialNLLScorer": "anomx.detection.scorers.timeseries",
    "GammaNLLScorer": "anomx.detection.scorers.timeseries",
    "GaussianNLLScorer": "anomx.detection.scorers.timeseries",
    "GenericScorer": "anomx.detection.generic",
    "KMeansScorer": "anomx.detection.scorers.timeseries",
    "LaplaceNLLScorer": "anomx.detection.scorers.timeseries",
    "NLLScorer": "anomx.detection.scorers.timeseries",
    "NormScorer": "anomx.detection.scorers.timeseries",
    "PoissonNLLScorer": "anomx.detection.scorers.timeseries",
    "PyODScorer": "anomx.detection.scorers.timeseries",
    "Scorer": "anomx.detection.base.scorers",
    "WassersteinScorer": "anomx.detection.scorers.timeseries",
    "WindowedAnomalyScorer": "anomx.detection.scorers.timeseries",
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
