"""Shared helpers for package-internal normalization and dataframe handling."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def normalize_text(value: object) -> str:
    """Return a trimmed single-line string representation."""
    return str(value or "").strip()


def normalise_component_key(value: object) -> str:
    """Normalize component identifiers into a stable snake-case-ish key."""
    normalized = normalize_text(value).lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def ensure_dataframe(data: Any) -> pd.DataFrame:
    """Convert a supported tabular payload into a DataFrame copy."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, pd.Series):
        return data.to_frame()
    return pd.DataFrame(data)


def dataframe_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize a DataFrame into JSON-friendly records."""
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].astype(str)
    if isinstance(normalized.index, pd.DatetimeIndex):
        normalized.index = normalized.index.astype(str)
    return list(normalized.reset_index(drop=False).to_dict(orient="records"))
