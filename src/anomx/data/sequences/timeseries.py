"""Simple typed time-series batch container."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class TimeSeriesBatch:
    """A batch of aligned time-series values backed by a DataFrame."""

    frame: pd.DataFrame
    time_column: str
    value_columns: list[str] = field(default_factory=list)

    def sorted_frame(self) -> pd.DataFrame:
        if self.time_column not in self.frame.columns:
            raise KeyError(f"Time column '{self.time_column}' does not exist in the batch.")
        return self.frame.sort_values(self.time_column).reset_index(drop=True)

    def value_frame(self) -> pd.DataFrame:
        frame = self.sorted_frame()
        columns = self.value_columns or frame.select_dtypes(include=["number"]).columns.tolist()
        if not columns:
            raise ValueError("The time-series batch does not contain any numeric value columns.")
        return frame[columns].copy()
