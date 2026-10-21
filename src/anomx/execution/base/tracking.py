"""Persistence contract for run lifecycle and step events."""

from collections.abc import Mapping
from typing import Any, Protocol


class RunBackend(Protocol):
    """Persist events synchronously; raise on failure so callers can retry.

    Events contain stable UUIDs. Remote implementations must deduplicate by
    ``event_id`` because a successful write may lose its acknowledgement.
    """

    def write(self, event: Mapping[str, Any]) -> None:
        """Durably accept one lifecycle or history event."""
        ...
