"""Statistical anomaly detectors."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from anomx.datasets import TimeSeriesDataset
from anomx.detectors.base import AnomalyDetector, DetectionResult
from anomx.scorers import ZScoreScorer


@dataclass
class MovingAverageDetector(AnomalyDetector):
    """Detect deviations from a rolling moving average.

    This reference implementation is deliberately simple. It gives the package
    a usable baseline while leaving room for Darts-backed and learned detectors.
    """

    window: int = 24
    threshold: float = 3.0
    min_periods: int | None = None

    def __post_init__(self) -> None:
        if self.window <= 1:
            msg = "window must be greater than 1."
            raise ValueError(msg)

    def fit(self, dataset: TimeSeriesDataset) -> MovingAverageDetector:
        """Fit is stateless for this rolling baseline."""
        self._target_columns = dataset.target_columns
        return self

    def score(self, dataset: TimeSeriesDataset) -> pd.Series:
        """Score observations by robust z-score of rolling residuals."""
        target = dataset.target
        frame = target.to_frame() if isinstance(target, pd.Series) else target
        rolling = frame.rolling(
            window=self.window,
            min_periods=self.min_periods or max(2, self.window // 3),
        ).mean()
        residuals = (frame - rolling).dropna(how="all")
        scores = ZScoreScorer(robust=True).score(residuals)
        return scores.reindex(dataset.data.index, fill_value=0.0)

    def predict(self, dataset: TimeSeriesDataset) -> DetectionResult:
        """Return anomaly scores and labels."""
        scores = self.score(dataset)
        labels = scores >= self.threshold
        return DetectionResult(
            scores=scores,
            labels=labels.rename("is_anomaly"),
            threshold=self.threshold,
            metadata={
                "detector": self.__class__.__name__,
                "window": self.window,
                "target_columns": dataset.target_columns,
            },
        )
