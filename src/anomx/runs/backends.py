"""Interchangeable persistence for run lifecycle and step events."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from anomx.integrations.platform import PlatformClient


class RunBackend(Protocol):
    """Persist events synchronously; raise on failure so callers can retry.

    Events contain stable UUIDs. Remote implementations must deduplicate by
    ``event_id`` because a successful write may lose its acknowledgement.
    """

    def write(self, event: Mapping[str, Any]) -> None:
        """Durably accept one lifecycle or history event."""
        ...


class JsonlBackend:
    """Append portable JSON events to a local file (offline by default)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = RLock()

    def write(self, event: Mapping[str, Any]) -> None:
        line = json.dumps(dict(event), allow_nan=False, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()


class CallbackBackend:
    """Adapt an existing worker transport or application-specific event sink."""

    def __init__(self, callback: Callable[[dict[str, Any]], Any]) -> None:
        self.callback = callback

    def write(self, event: Mapping[str, Any]) -> None:
        self.callback(dict(event))


class PlatformBackend:
    """Use the lightweight authenticated platform client to persist runs.

    ``client`` may be a :class:`anomx.PlatformClient`; when omitted its explicit
    ``ANOMX_URL`` and ``ANOMX_TOKEN`` environment configuration is used.
    The API URL includes ``/api`` (or ``/api/v1``).
    """

    def __init__(self, client: PlatformClient | None = None) -> None:
        if client is None:
            from anomx.integrations.platform import PlatformClient

            client = PlatformClient.from_env()
        self.client = client

    def write(self, event: Mapping[str, Any]) -> None:
        from uuid import UUID

        run_id = str(UUID(str(event["run_id"])))
        path = "/runs/start" if event["kind"] == "start" else f"/runs/{run_id}/events"
        self.client.request(path, dict(event))
