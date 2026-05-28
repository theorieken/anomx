"""Simple statistical anomaly scorers."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from anomx.scorers.base import AnomalyScorer


class ZScoreScorer(AnomalyScorer):
    """Score observations by absolute robust z-score."""

    def __init__(self, *, robust: bool = True, epsilon: float = 1e-9) -> None:
        self.robust = robust
        self.epsilon = epsilon

    def score(self, values: pd.Series | pd.DataFrame) -> pd.Series:
        """Return absolute z-scores for a series or max-column score for a frame."""
        frame = values.to_frame() if isinstance(values, pd.Series) else values
        if self.robust:
            center = frame.median()
            scale = (frame - center).abs().median() * 1.4826
        else:
            center = frame.mean()
            scale = frame.std(ddof=0)
        z_scores = (frame - center).abs() / (scale.replace(0, np.nan) + self.epsilon)
        return cast(pd.Series, z_scores.max(axis=1).fillna(0.0).rename("score"))


class ThresholdScorer(AnomalyScorer):
    """Score values by distance above a fixed threshold."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def score(self, values: pd.Series | pd.DataFrame) -> pd.Series:
        """Return positive distance above the threshold."""
        frame = values.to_frame() if isinstance(values, pd.Series) else values
        distances = (frame - self.threshold).clip(lower=0)
        return cast(pd.Series, distances.max(axis=1).rename("score"))
