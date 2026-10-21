"""Typed model outputs, independent of anomaly decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from anomx._shared import dataframe_to_records, ensure_dataframe


class InferenceKind(str, Enum):
    """What kind of output a normality model produces."""

    FORECAST = "forecast"
    RECONSTRUCTION = "reconstruction"
    EMBEDDING = "embedding"
    DENSITY = "density"
    DISTANCE = "distance"
    CLASSIFICATION = "classification"


@dataclass(slots=True)
class InferenceOutput:
    """Model output aligned with the observations it was computed for."""

    kind: InferenceKind
    frame: pd.DataFrame
    output_columns: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frame = ensure_dataframe(self.frame)

    def to_records(self) -> list[dict[str, Any]]:
        return dataframe_to_records(self.frame)
