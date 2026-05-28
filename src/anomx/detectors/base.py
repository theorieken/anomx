"""Base detector contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from anomx.datasets import TimeSeriesDataset


@dataclass(frozen=True)
class DetectionResult:
    """Output produced by an anomaly detector."""

    scores: pd.Series
    labels: pd.Series
    threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Return scores and labels in a DataFrame."""
        return pd.DataFrame(
            {
                "score": self.scores.astype(float),
                "is_anomaly": self.labels.astype(bool),
            }
        )


class AnomalyDetector(ABC):
    """Base class for batch and online anomaly detectors."""

    @abstractmethod
    def fit(self, dataset: TimeSeriesDataset) -> AnomalyDetector:
        """Fit the detector on a dataset."""

    @abstractmethod
    def score(self, dataset: TimeSeriesDataset) -> pd.Series:
        """Return anomaly scores for a dataset."""

    @abstractmethod
    def predict(self, dataset: TimeSeriesDataset) -> DetectionResult:
        """Return anomaly labels and scores for a dataset."""

    def fit_predict(self, dataset: TimeSeriesDataset) -> DetectionResult:
        """Fit the detector and return predictions for the same dataset."""
        return self.fit(dataset).predict(dataset)
