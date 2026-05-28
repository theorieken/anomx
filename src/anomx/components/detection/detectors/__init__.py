"""Detector components."""

from anomx.components.detection.detectors.base import BaseDetector
from anomx.components.detection.detectors.threshold import ThresholdDetector

__all__ = [
    "BaseDetector",
    "ThresholdDetector",
]
