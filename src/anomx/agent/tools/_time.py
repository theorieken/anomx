"""Small time helpers for tools."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp suitable for JSONL events."""

    return datetime.now(tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
