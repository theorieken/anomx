"""Adapt execution-host callbacks to the standalone dataset client contract."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast


class WorkDatasetClient:
    """Expose a host's dataset callbacks without coupling data to execution."""

    def __init__(
        self,
        callbacks: Mapping[str, Callable[..., object]],
        data_loader: Callable[..., object] | None,
    ) -> None:
        self.callbacks = callbacks
        self.data_loader = data_loader

    def _request(self, operation: str, **arguments: object) -> dict[str, Any]:
        callback = self.callbacks.get(operation)
        if callback is None:
            raise RuntimeError(f"The work host does not support {operation}.")
        return cast(dict[str, Any], callback(**arguments))

    def describe_dataset(self, reference: str) -> dict[str, Any]:
        return self._request("describe_dataset", reference=reference)

    def read_dataset(
        self, reference: str, *, version: str, cursor: str | None = None, limit: int = 5000
    ) -> dict[str, Any]:
        return self._request(
            "read_dataset", reference=reference, version=version, cursor=cursor, limit=limit
        )

    def load_channels(
        self,
        channels: Sequence[str],
        *,
        start: str | None = None,
        end: str | None = None,
        frequency: str | None = None,
    ) -> object:
        if self.data_loader is None:
            raise RuntimeError("This execution host does not provide a dataset loader.")
        return self.data_loader(channels=channels, start=start, end=end, frequency=frequency)
