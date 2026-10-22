"""Canonical time selections shared by local tables and remote dataset views."""

from __future__ import annotations

from datetime import datetime
from typing import TypeAlias

import pandas as pd

TimeBound: TypeAlias = str | datetime | pd.Timestamp
AGGREGATIONS = frozenset({"mean", "last", "first", "min", "max", "sum", "count"})


def time_selection(
    current: dict[str, str], *, start: TimeBound | None, end: TimeBound | None
) -> dict[str, str]:
    """Normalize explicit bounds to UTC; timezone-free inputs mean UTC."""
    selection = dict(current)
    for key, value in (("start", start), ("end", end)):
        if value is None:
            continue
        if not isinstance(value, (str, datetime, pd.Timestamp)):
            raise ValueError(f"{key} must be an ISO timestamp or datetime.")
        try:
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp):
                raise ValueError
            timestamp = (
                timestamp.tz_localize("UTC")
                if timestamp.tzinfo is None
                else timestamp.tz_convert("UTC")
            )
        except (ValueError, TypeError, OverflowError):
            raise ValueError(f"{key} must be a valid timestamp.") from None
        selection[key] = timestamp.isoformat()
    if (
        "start" in selection
        and "end" in selection
        and pd.Timestamp(selection["start"]) >= pd.Timestamp(selection["end"])
    ):
        raise ValueError("start must be earlier than end (the end is exclusive).")
    return selection


def sampling_selection(current: dict[str, str], frequency: str, aggregation: str) -> dict[str, str]:
    """Use fixed UTC-epoch bins, with explicit aggregation and no gap filling."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {', '.join(sorted(AGGREGATIONS))}.")
    try:
        if not isinstance(frequency, str) or len(frequency) > 64:
            raise ValueError
        interval = pd.tseries.frequencies.to_offset(frequency).nanos
        if interval < 10_000_000:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError("frequency must be a fixed interval of at least 10 ms.") from None
    return {**current, "frequency": f"{interval}ns", "aggregation": aggregation}


def select_frame(frame: pd.DataFrame, selection: dict[str, str]) -> pd.DataFrame:
    """Filter observations before aggregating; omit bins with no observations."""
    if not selection:
        return frame.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Time selection and downsampling require a DatetimeIndex.")
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    if "start" in selection:
        frame = frame.loc[frame.index >= pd.Timestamp(selection["start"])]
    if "end" in selection:
        frame = frame.loc[frame.index < pd.Timestamp(selection["end"])]
    if "frequency" in selection and not frame.empty:
        bins = frame.resample(selection["frequency"], origin="epoch", closed="left", label="left")
        frame = bins.agg(selection["aggregation"]).loc[bins.size() > 0]
    return frame
