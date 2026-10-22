"""Atomic, credential-scoped JSON caches for materialized remote datasets.

Caches never deserialize executable Python objects. A platform resolution must
succeed before reuse; a cached observation table is not an offline auth bypass.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.io.json import build_table_schema


class DatasetCache:
    """Store materialized tables using a validated source token or an explicit TTL.

    ``namespace`` must identify both the server and authorization scope. The
    built-in platform adapter hashes the API origin and token; custom clients
    can expose ``cache_namespace`` to opt into persistence. A token is never
    written to disk. Entries are atomic and user-readable only on POSIX.
    """

    def __init__(self, directory: str | Path, *, namespace: str, reference: str) -> None:
        identity = json.dumps([namespace, reference], separators=(",", ":"))
        self.path = Path(directory).expanduser() / (
            hashlib.sha256(identity.encode()).hexdigest() + ".json"
        )

    def read(self, definition: dict[str, Any], *, max_age: float | None = 0) -> pd.DataFrame | None:
        """Return a current entry, or a cache miss for stale/corrupt/absent data."""
        try:
            entry = json.loads(self.path.read_text())
            if entry["format"] != 1 or entry["version"] != definition["version"]:
                return None
            interpretation = {
                key: definition.get(key) for key in ("kind", "time_column", "time_unit")
            }
            if entry.get("interpretation") != interpretation:
                return None
            token = definition.get("cache_token")
            if token:
                if entry["cache_token"] != token:
                    return None
            elif max_age is not None and not (
                max_age > 0 and 0 <= time.time() - entry["created"] <= max_age
            ):
                return None
            payload = entry["table"]
            if (
                not isinstance(payload, str)
                or hashlib.sha256(payload.encode()).hexdigest() != entry["sha256"]
            ):
                return None
            frame = pd.read_json(io.StringIO(payload), orient="table", precise_float=True)
            if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
                frame.index = frame.index.tz_convert("UTC")
            return frame
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            return None

    def write(self, frame: pd.DataFrame, definition: dict[str, Any]) -> None:
        """Atomically publish one complete table, never a partial download."""
        table = {
            "schema": build_table_schema(frame, index=True),
            "data": _json_value(frame.reset_index().to_dict(orient="records")),
        }
        payload = json.dumps(table, allow_nan=False, separators=(",", ":"))
        entry = {
            "format": 1,
            "version": definition["version"],
            "cache_token": definition.get("cache_token"),
            "interpretation": {
                key: definition.get(key) for key in ("kind", "time_column", "time_unit")
            },
            "created": time.time(),
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "table": payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary = tempfile.mkstemp(prefix=".dataset-", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(entry, stream, separators=(",", ":"))
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def clear(self) -> None:
        """Remove only this dataset's entry in this authorization scope."""
        self.path.unlink(missing_ok=True)


def _json_value(value: Any) -> Any:  # noqa: ANN401 - JSON/scientific scalar boundary
    """Keep Python's lossless float representation instead of pandas' 15 digits."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, float) and math.isinf(value):
        raise ValueError("Infinite observations cannot be represented in the JSON table cache.")
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (timedelta, pd.Timedelta)):
        return pd.Timedelta(value).isoformat()
    return value
