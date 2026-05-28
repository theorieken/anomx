"""Naive forecasting baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd

from anomx.datasets import TimeSeriesDataset
from anomx.models.base import Forecast, ForecastingModel


@dataclass
class NaiveSeasonalModel(ForecastingModel):
    """Forecast by repeating the latest observed seasonal window."""

    season_length: int = 1

    def __post_init__(self) -> None:
        if self.season_length < 1:
            msg = "season_length must be at least 1."
            raise ValueError(msg)

    def fit(self, dataset: TimeSeriesDataset) -> NaiveSeasonalModel:
        """Store the last seasonal window from the training dataset."""
        target = dataset.target
        frame = target.to_frame() if isinstance(target, pd.Series) else target
        if len(frame) < self.season_length:
            msg = "Dataset is shorter than season_length."
            raise ValueError(msg)
        self._columns = tuple(frame.columns)
        self._last_window = frame.tail(self.season_length).copy()
        self._last_timestamp = frame.index[-1]
        index = cast(pd.DatetimeIndex, frame.index)
        self._frequency = dataset.frequency or pd.infer_freq(index)
        return self

    def predict(self, horizon: int) -> Forecast:
        """Predict by repeating the stored seasonal window."""
        if horizon <= 0:
            msg = "horizon must be positive."
            raise ValueError(msg)
        if not hasattr(self, "_last_window"):
            msg = "Model must be fit before predict."
            raise RuntimeError(msg)
        values = []
        for step in range(horizon):
            values.append(self._last_window.iloc[step % self.season_length])
        frame = pd.DataFrame(values, columns=self._columns)

        if self._frequency is not None:
            index = pd.date_range(
                self._last_timestamp,
                periods=horizon + 1,
                freq=self._frequency,
                tz=self._last_timestamp.tz,
            )[1:]
            frame.index = index

        return Forecast(
            values=frame,
            metadata={"model": self.__class__.__name__, "season_length": self.season_length},
        )
