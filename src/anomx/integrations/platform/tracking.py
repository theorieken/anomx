"""Optional HTTP persistence for run events."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anomx.integrations.platform import PlatformClient


class PlatformBackend:
    """Use the lightweight authenticated platform client to persist runs.

    ``client`` may be a :class:`anomx.PlatformClient`; when omitted its explicit
    ``ANOMX_URL`` and ``ANOMX_TOKEN`` environment configuration is used.
    Site-origin URLs default to ``/api``; explicit API prefixes are preserved.
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
        self.client.post(path, dict(event))
