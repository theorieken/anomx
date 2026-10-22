"""Dataset API semantics, kept separate from the generic HTTP transport."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any
from urllib.parse import quote, urlsplit

from anomx.integrations.platform import PlatformClient, PlatformError


class PlatformDatasetClient:
    """Adapt a generic PlatformClient to the data domain's DatasetClient contract.

    Dataset resolution, versioned batches, channel loading, and response shapes
    belong here. The HTTP client only sends requests. Both unversioned ``/api``
    and grouped ``/api/v1`` routes are supported.
    """

    def __init__(self, client: PlatformClient, *, channels_path: str | None = None) -> None:
        self.client = client
        grouped = urlsplit(client.url).path.endswith("/v1")
        self.channels_path = channels_path or (
            "/jobs/jobs/channel-data" if grouped else "/jobs/channel-data"
        )

    @property
    def cache_namespace(self) -> str:
        """Partition cached data by server and authorization, without storing a token."""
        return hashlib.sha256(f"{self.client.url}\0{self.client.token}".encode()).hexdigest()

    def _dataset_path(self, reference: str, operation: str) -> str:
        prefix = "/data" if urlsplit(self.client.url).path.endswith("/v1") else ""
        return f"{prefix}/datasets/{quote(reference, safe='')}/{operation}"

    def describe_dataset(self, reference: str) -> dict[str, Any]:
        """Resolve a dataset definition without loading its observations."""
        result = self.client.get(self._dataset_path(reference, "resolve"))
        if not isinstance(result, dict):
            raise PlatformError("The dataset API returned an invalid definition.")
        return result

    def describe_dataset_selection(
        self, reference: str, *, selection: dict[str, str]
    ) -> dict[str, Any]:
        """Resolve a server-filtered view and require explicit host support.

        Older servers may ignore unknown query arguments. Never accept such a
        response as a filtered dataset: the capability and effective selection
        must be present before any observation request is sent.
        """
        result = self.client.get(self._dataset_path(reference, "resolve"), params=selection)
        if not isinstance(result, dict):
            raise PlatformError("The dataset API returned an invalid definition.")
        capabilities = result.get("capabilities")
        if (
            not isinstance(capabilities, dict)
            or capabilities.get("selection") is not True
            or not isinstance(result.get("selection"), dict)
            or not result["selection"]
        ):
            raise PlatformError(
                "This dataset host does not support server-side time selection/downsampling. "
                "Update the platform API before using select() or downsample() remotely."
            )
        if "frequency" in selection:
            from anomx.data._selection import sampling_selection

            effective = result["selection"]
            try:
                requested_sampling = sampling_selection(
                    {}, selection["frequency"], selection.get("aggregation", "mean")
                )
                effective_sampling = sampling_selection(
                    {}, effective["frequency"], effective["aggregation"]
                )
            except (KeyError, TypeError, ValueError):
                raise PlatformError(
                    "The dataset host did not confirm the selected downsampling."
                ) from None
            if requested_sampling != effective_sampling:
                raise PlatformError("The dataset host returned different downsampling settings.")
        return result

    def read_dataset(
        self, reference: str, *, version: str, cursor: str | None = None, limit: int = 5000
    ) -> dict[str, Any]:
        """Read a bounded batch pinned to an explicitly resolved source version."""
        result = self.client.post(
            self._dataset_path(reference, "data"),
            {"version": version, "cursor": cursor, "limit": limit},
        )
        if not isinstance(result, dict):
            raise PlatformError("The dataset API returned an invalid batch.")
        return result

    def load_channels(
        self,
        channels: Sequence[str],
        *,
        start: str | None = None,
        end: str | None = None,
        frequency: str | None = None,
    ) -> object:
        """Load aligned channel records using the data domain's channel endpoint."""
        result = self.client.post(
            self.channels_path,
            {"channels": list(channels), "start": start, "end": end, "frequency": frequency},
        )
        if not isinstance(result, (dict, list)):
            raise PlatformError("The channel API returned invalid records.")
        return result.get("records", result) if isinstance(result, dict) else result
