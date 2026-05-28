"""Base scorer contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class AnomalyScorer(ABC):
    """Base class for components that convert observations into anomaly scores."""

    @abstractmethod
    def score(self, values: pd.Series | pd.DataFrame) -> pd.Series:
        """Return one anomaly score per timestamp."""
