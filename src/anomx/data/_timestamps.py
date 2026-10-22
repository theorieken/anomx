"""Timestamp normalization shared by dataset constructors and remote readers."""

from typing import Any, Literal, cast

import pandas as pd


def parse_timestamps(values: Any, *, unit: str | None = None) -> Any:  # noqa: ANN401
    """Parse varying ISO precision and offsets as UTC, preserving numeric units.

    Pandas otherwise infers a single fractional-second format from the first
    record. Numeric epoch sources must supply their unit in the definition.
    """
    if unit:
        return pd.to_datetime(
            values, utc=True, unit=cast("Literal['D', 's', 'ms', 'us', 'ns']", unit)
        )
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values, utc=True)
    return pd.to_datetime(values, utc=True, format="mixed")
