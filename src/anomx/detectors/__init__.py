"""Anomaly detector abstractions and implementations."""

from anomx.detectors.base import AnomalyDetector, DetectionResult
from anomx.detectors.statistical import MovingAverageDetector

__all__ = [
    "AnomalyDetector",
    "DetectionResult",
    "MovingAverageDetector",
]
