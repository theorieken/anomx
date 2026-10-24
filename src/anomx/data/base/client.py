"""Dataset access contracts and an optional scoped client binding."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol


class DatasetClient(Protocol):
    """Resolve versioned datasets and load channels without a specific host."""

    def describe_dataset(self, reference: str) -> dict[str, Any]: ...

    def read_dataset(
        self, reference: str, *, version: str, cursor: str | None = None, limit: int = 5000
    ) -> dict[str, Any]: ...

    def load_channels(
        self,
        channels: Sequence[str],
        *,
        start: str | None = None,
        end: str | None = None,
        frequency: str | None = None,
    ) -> object: ...


class SelectableDatasetClient(DatasetClient, Protocol):
    """Optional capability for immutable time-window and aggregation views."""

    def describe_dataset_selection(
        self, reference: str, *, selection: dict[str, Any]
    ) -> dict[str, Any]: ...


_active_client: ContextVar[DatasetClient | None] = ContextVar("anomx_dataset_client", default=None)


def current_dataset_client() -> DatasetClient | None:
    """Return the client bound by an application or execution host, if any."""
    return _active_client.get()


@contextmanager
def use_dataset_client(client: DatasetClient) -> Iterator[DatasetClient]:
    """Temporarily bind a client, restoring the previous binding on exit."""
    token = _active_client.set(client)
    try:
        yield client
    finally:
        _active_client.reset(token)
