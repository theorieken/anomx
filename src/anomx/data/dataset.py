"""Lightweight datasets for time series, generic sequences, and independent samples."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from anomx.integrations.darts.datasets import Dataset as DartsDataset
    from anomx.integrations.platform import PlatformClient


from anomx.data.base.client import DatasetClient, current_dataset_client

DatasetKind = Literal["time_series", "sequence", "independent_samples"]
DatasetMode = Literal["fixed", "automatic"]


class Dataset:
    """Local observations or a lazy, versioned platform dataset reference.

    ``Dataset("data_dataset-...")`` resolves through the active work context or
    an explicit ``PlatformClient`` (falling back to ANOMX_URL/ANOMX_TOKEN).
    ``Dataset(anomx_reference="data_dataset-...", client=client)`` makes the
    remote identity explicit. HTTP endpoint details live in the data source.
    ``iter_batches`` bounds memory; ``to_dataframe`` deliberately materializes
    the whole dataset. Fixed external sources are references, never copied.
    """

    def __init__(
        self,
        reference: str | None = None,
        *,
        anomx_reference: str | None = None,
        data: pd.DataFrame | None = None,
        kind: DatasetKind = "independent_samples",
        mode: DatasetMode = "fixed",
        name: str = "",
        metadata: Mapping[str, Any] | None = None,
        client: DatasetClient | PlatformClient | None = None,
    ) -> None:
        if anomx_reference is not None:
            if reference is not None:
                raise ValueError("Provide reference or anomx_reference, not both.")
            reference = anomx_reference
        if kind not in {"time_series", "sequence", "independent_samples"}:
            raise ValueError("Unknown dataset kind.")
        if mode not in {"fixed", "automatic"}:
            raise ValueError("Dataset mode must be fixed or automatic.")
        if reference is not None and (not isinstance(reference, str) or not reference.strip()):
            raise ValueError("Dataset references must be nonempty strings.")
        if reference and data is not None:
            raise ValueError("Provide local data or a dataset reference, not both.")
        self.reference = reference
        self.kind, self.mode, self.name = kind, mode, name
        self.metadata = dict(metadata or {})
        self.client = client
        self._data = data.copy() if data is not None else None
        self._definition: dict[str, Any] | None = None

    @property
    def anomx_reference(self) -> str | None:
        """The remote dataset identity, or ``None`` for local observations."""
        return self.reference

    def _get_client(self) -> DatasetClient:
        from anomx.data.sources.platform import PlatformDatasetClient
        from anomx.integrations.platform import PlatformClient

        client = self.client or current_dataset_client()
        if client is None:
            client = self.client = PlatformClient.from_env()
        return PlatformDatasetClient(client) if isinstance(client, PlatformClient) else client

    def describe(self, *, refresh: bool = False) -> dict[str, Any]:
        """Resolve source identity and schema without loading all observations."""
        if not self.reference:
            return {
                "name": self.name,
                "kind": self.kind,
                "mode": self.mode,
                "metadata": self.metadata,
            }
        if self._definition is None or refresh:
            definition = self._get_client().describe_dataset(self.reference)
            if not isinstance(definition, dict) or not definition.get("version"):
                raise ValueError("The dataset host did not return a versioned definition.")
            self._definition = definition
            self.kind = definition["kind"]
            self.mode = definition["mode"]
            self.name = definition.get("name", "")
            self.metadata = dict(definition.get("metadata") or {})
        return dict(self._definition)

    def iter_batches(self, batch_size: int = 5000) -> Iterator[pd.DataFrame]:
        """Yield ordered batches pinned to one resolved source version."""
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or not 1 <= batch_size <= 10000
        ):
            raise ValueError("Batch size must be between 1 and 10,000 rows.")
        if self._data is not None:
            for offset in range(0, len(self._data), batch_size):
                yield self._data.iloc[offset : offset + batch_size].copy()
            return
        if not self.reference:
            return
        definition = self.describe(refresh=self.mode == "automatic")
        cursor = None
        seen: set[str] = set()
        columns: list[str] | None = None
        while True:
            response = self._get_client().read_dataset(
                self.reference, version=definition["version"], cursor=cursor, limit=batch_size
            )
            if response.get("version") != definition["version"]:
                raise ValueError("Dataset sources changed while loading; resolve a new version.")
            current_columns = response.get("columns", [])
            if columns is not None and current_columns != columns:
                raise ValueError(
                    "Dataset files have different column schemas; select a coherent set of columns."
                )
            columns = current_columns
            frame = pd.DataFrame(response.get("records", []), columns=current_columns)
            time_column = definition.get("time_column")
            if self.kind == "time_series" and time_column and time_column in frame.columns:
                frame[time_column] = pd.to_datetime(
                    frame[time_column],
                    utc=True,
                    **({"unit": definition["time_unit"]} if definition.get("time_unit") else {}),
                )
                frame = frame.set_index(time_column)
            if not frame.empty:
                yield frame
            cursor = response.get("next_cursor")
            if cursor is None:
                return
            if not isinstance(cursor, str) or cursor in seen:
                raise ValueError("The dataset host returned an invalid pagination cursor.")
            seen.add(cursor)

    def to_dataframe(self) -> pd.DataFrame:
        """Materialize observations; prefer iter_batches for large datasets."""
        if self._data is not None:
            return self._data.copy()
        batches = list(self.iter_batches())
        return (
            pd.concat(batches, ignore_index=self.kind != "time_series")
            if batches
            else pd.DataFrame()
        )

    @property
    def data(self) -> pd.DataFrame:
        """Load and return the observations as a DataFrame."""
        return self.to_dataframe()

    def values(self) -> NDArray[Any]:
        """Return observations as a NumPy array."""
        return self.to_dataframe().to_numpy()

    @classmethod
    def from_dataframe(
        cls,
        frame: pd.DataFrame,
        *,
        time_column: str | None = None,
        kind: DatasetKind | None = None,
        mode: DatasetMode = "fixed",
        name: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Dataset:
        """Create a local dataset, preserving a supplied temporal index."""
        data = frame.copy()
        if time_column is not None:
            data[time_column] = pd.to_datetime(data[time_column], utc=True)
            data = data.set_index(time_column)
        resolved_kind = kind or (
            "time_series" if isinstance(data.index, pd.DatetimeIndex) else "independent_samples"
        )
        if resolved_kind == "time_series" and not isinstance(
            data.index, (pd.DatetimeIndex, pd.RangeIndex)
        ):
            raise ValueError("Time series require a DatetimeIndex or RangeIndex.")
        return cls(data=data, kind=resolved_kind, mode=mode, name=name, metadata=metadata)

    @classmethod
    def from_values(
        cls,
        values: ArrayLike,
        *,
        columns: Sequence[str] | None = None,
        kind: DatasetKind = "time_series",
        name: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Dataset:
        """Create a regularly indexed time series from a 1D or 2D array."""
        array = np.asarray(values)
        if array.ndim not in {1, 2}:
            raise ValueError(
                "Use a table with explicit sample columns for higher-dimensional data."
            )
        return cls.from_dataframe(
            pd.DataFrame(array, columns=list(columns) if columns is not None else None),
            kind=kind,
            name=name,
            metadata=metadata,
        )

    @classmethod
    def from_channels(
        cls,
        channels: Sequence[str],
        *,
        client: DatasetClient | PlatformClient | None = None,
        loader: Callable[..., object] | None = None,
        start: str | None = None,
        end: str | None = None,
        frequency: str | None = None,
    ) -> Dataset:
        """Resolve recorded channels through an API connection or host loader."""
        if (
            isinstance(channels, str)
            or not channels
            or any(not isinstance(channel, str) or not channel.strip() for channel in channels)
        ):
            raise ValueError("Pass a nonempty sequence of channel references.")
        if loader is not None and client is not None:
            raise ValueError("Provide either client or loader, not both.")
        if loader is None:
            from anomx.data.sources.platform import PlatformDatasetClient
            from anomx.integrations.platform import PlatformClient

            client = client or current_dataset_client() or PlatformClient.from_env()
            if isinstance(client, PlatformClient):
                client = PlatformDatasetClient(client)
            loader = client.load_channels
        response = loader(channels=list(channels), start=start, end=end, frequency=frequency)
        records = response.get("records", response) if isinstance(response, dict) else response
        frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
        return cls.from_dataframe(
            frame,
            time_column="timestamp" if "timestamp" in frame else None,
            kind="time_series",
            mode="automatic",
            metadata={"channels": list(channels), "start": start, "end": end},
        )

    def start_time(self) -> object:
        """Return the first temporal index value; materializes a remote selection."""
        return self.to_dataframe().index[0]

    def end_time(self) -> object:
        """Return the last temporal index value; materializes a remote selection."""
        return self.to_dataframe().index[-1]

    def to_darts(self) -> DartsDataset:
        """Convert temporal data for optional Darts models without a core dependency."""
        if self.reference:
            self.describe()
        if self.kind not in {"time_series", "sequence"}:
            raise ValueError("Darts models require a temporal or sequential dataset.")
        from anomx.integrations.darts.datasets import Dataset as DartsDataset

        frame = self.to_dataframe()
        if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
            frame.index = frame.index.tz_convert("UTC").tz_localize(None)
        return cast("DartsDataset", DartsDataset.from_dataframe(frame))


__all__ = ["Dataset", "DatasetClient", "DatasetKind", "DatasetMode"]
