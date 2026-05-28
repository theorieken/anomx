"""Base model contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd

from anomx.datasets import TimeSeriesDataset


@dataclass(frozen=True)
class Forecast:
    """Forecast values with optional model metadata."""

    values: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Return forecast values as a DataFrame."""
        return cast(pd.DataFrame, self.values.copy())


class ForecastingModel(ABC):
    """Base class for forecasting models."""

    @abstractmethod
    def fit(self, dataset: TimeSeriesDataset) -> ForecastingModel:
        """Fit the model."""

    @abstractmethod
    def predict(self, horizon: int) -> Forecast:
        """Predict a future horizon."""
