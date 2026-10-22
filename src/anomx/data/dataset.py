"""Lightweight datasets for time series, generic sequences, and independent samples."""

from __future__ import annotations

import json
import math
import os
import warnings
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from contextlib import closing
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from anomx.data.timeseries import TimeSeries
    from anomx.integrations.darts.datasets import Dataset as DartsDataset
    from anomx.integrations.platform import PlatformClient


from anomx.console import operation, progress_bar
from anomx.data._selection import TimeBound, sampling_selection, select_frame, time_selection
from anomx.data._timestamps import parse_timestamps
from anomx.data.base.client import DatasetClient, current_dataset_client
from anomx.data.cache import DatasetCache

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
        cache: bool | str | Path = True,
        cache_ttl: float | None = 0,
        progress: bool = True,
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
        self._selection: dict[str, str] = {}
        if cache_ttl is not None and (not math.isfinite(cache_ttl) or cache_ttl < 0):
            raise ValueError("cache_ttl must be a finite nonnegative number of seconds.")
        self.cache = cache
        self.cache_ttl = cache_ttl
        self.progress = progress
        self.cache_info: dict[str, Any] = {"status": "unused"}

    @property
    def anomx_reference(self) -> str | None:
        """The remote dataset identity, or ``None`` for local observations."""
        return self.reference

    def _view(self, selection: dict[str, str]) -> Dataset:
        view = copy(self)
        view._selection = selection
        view._definition = None
        view.metadata = dict(self.metadata)
        view.cache_info = {"status": "unused"}
        return view

    def select(self, *, start: TimeBound | None = None, end: TimeBound | None = None) -> Dataset:
        """Return a lazy time window, including start and excluding end.

        Strings and datetimes are accepted; timezone-free bounds mean UTC.
        Explicit bounds replace those on this view. Remote selection happens
        on the server and requires its dataset selection capability.
        """
        return self._view(time_selection(self._selection, start=start, end=end))

    def downsample(self, frequency: str, *, aggregation: str = "mean") -> Dataset:
        """Return a lazy view aggregated into fixed UTC-epoch bins.

        Supported reducers are mean, first, last, min, max, sum, and count.
        Time filtering precedes aggregation; missing bins are omitted and no
        values are forward-filled. Bin labels are their left edges, which can
        precede a selection's start. Remote aggregation reduces transferred
        rows and requires a platform with dataset selection support.
        """
        return self._view(sampling_selection(self._selection, frequency, aggregation))

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
                "selection": dict(self._selection),
            }
        if self._definition is None or refresh:
            client = self._get_client()
            if self._selection:
                resolver = getattr(client, "describe_dataset_selection", None)
                if resolver is None:
                    raise NotImplementedError("This dataset host does not support time selection.")
                definition = resolver(self.reference, selection=dict(self._selection))
            else:
                definition = client.describe_dataset(self.reference)
            if not isinstance(definition, dict) or not definition.get("version"):
                raise ValueError("The dataset host did not return a versioned definition.")
            self._definition = definition
            self.kind = definition["kind"]
            self.mode = definition["mode"]
            self.name = definition.get("name", "")
            self.metadata = dict(definition.get("metadata") or {})
        return dict(self._definition)

    def iter_batches(
        self,
        batch_size: int = 5000,
        *,
        progress: bool | None = None,
        refresh: bool = False,
    ) -> Iterator[pd.DataFrame]:
        """Yield bounded, ordered batches pinned to one resolved source version.

        Each remote page is checkpointed before yielding. A new iterator
        replays saved pages, then resumes at the saved cursor. Close the
        iterator when stopping early to release the download lock.
        ``progress=False`` silences tqdm in applications and automated jobs.
        """
        definition = self.describe(refresh=True) if self.reference else {}
        if refresh:
            self._reset_download(definition)
        self.cache_info = {"status": "unused"}
        with (
            progress_bar(
                self.name or "dataset",
                enabled=self.progress if progress is None else progress,
                unit=" rows",
                unit_scale=True,
            ) as bar,
            closing(self._iter_batches(batch_size, definition)) as batches,
        ):
            for frame in batches:
                frame.attrs["anomx_selection"] = self._selection_metadata(definition)
                bar.update(len(frame))
                yield frame

    def _iter_batches(
        self,
        batch_size: int,
        definition: dict[str, Any],
    ) -> Generator[pd.DataFrame, None, None]:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or not 1 <= batch_size <= 10000
        ):
            raise ValueError("Batch size must be between 1 and 10,000 rows.")
        if self._data is not None:
            data = select_frame(self._data, self._selection)
            for offset in range(0, len(data), batch_size):
                yield data.iloc[offset : offset + batch_size].copy()
            return
        if not self.reference:
            return
        from anomx.data.download import DatasetDownload

        store = self._cache_store()
        journal = None
        if store:
            try:
                journal = DatasetDownload(
                    store.path.with_suffix(".parts"), definition=definition, max_age=self.cache_ttl
                ).__enter__()
            except OSError as exc:
                self._cache_warning(exc)
        self.cache_info = {
            "status": "miss" if journal else "disabled",
            "cached_rows": 0,
            "downloaded_rows": 0,
            "complete": False,
        }
        if journal:
            self.cache_info["path"] = str(journal.directory)
            if journal.reused:
                self.cache_info["validation"] = (
                    "snapshot"
                    if not journal.complete and not definition.get("cache_token")
                    else self._cache_validation(definition)
                )
        cursor = None
        seen: set[str] = set()
        columns: list[str] | None = None

        def decode(response: dict[str, Any]) -> pd.DataFrame:
            nonlocal columns
            if response.get("version") != definition["version"]:
                raise ValueError("Dataset sources changed while loading; resolve a new version.")
            current_columns = response.get("columns", [])
            if (
                not isinstance(current_columns, list)
                or not all(isinstance(column, str) for column in current_columns)
                or len(set(current_columns)) != len(current_columns)
            ):
                raise ValueError("The dataset host returned invalid columns.")
            if columns is not None and current_columns != columns:
                raise ValueError("Dataset files have different column schemas.")
            columns = current_columns
            next_cursor = response.get("next_cursor")
            if next_cursor is not None:
                if not isinstance(next_cursor, str) or not next_cursor or next_cursor in seen:
                    raise ValueError("The dataset host returned an invalid pagination cursor.")
                seen.add(next_cursor)
            records = response.get("records", [])
            if not isinstance(records, list):
                raise ValueError("The dataset host returned invalid records; expected a list.")
            frame = pd.DataFrame(records, columns=current_columns)
            time_column = definition.get("time_column")
            if self.kind == "time_series" and time_column and time_column in frame.columns:
                frame[time_column] = parse_timestamps(
                    frame[time_column], unit=definition.get("time_unit")
                )
                frame = frame.set_index(time_column)
            return frame

        try:
            if journal:
                for response in journal.iter_pages():
                    frame = decode(response)
                    cursor = response.get("next_cursor")
                    self.cache_info.update(status="hit", complete=journal.complete)
                    for offset in range(0, len(frame), batch_size):
                        piece = frame.iloc[offset : offset + batch_size].copy()
                        self.cache_info["cached_rows"] += len(piece)
                        yield piece
                if journal.complete:
                    if not journal.row_count:
                        yield frame
                    return
            while True:
                try:
                    response = self._get_client().read_dataset(
                        self.reference,
                        version=definition["version"],
                        cursor=cursor,
                        limit=batch_size,
                    )
                except Exception as exc:
                    if journal and journal.page_count:
                        exc.add_note(
                            f"{journal.row_count:,} rows remain saved in {journal.directory}. "
                            "Re-run the same selection to resume. If the host has expired its "
                            "cursor, refresh=True explicitly starts a new download."
                        )
                    raise
                frame = decode(response)
                complete = response.get("next_cursor") is None
                if complete and definition.get("cache_token"):
                    try:
                        self._validate_download(definition)
                    except ValueError:
                        if journal:
                            journal.invalidate()
                        raise
                if journal:
                    try:
                        journal.append(response, request_cursor=cursor)
                    except OSError as exc:
                        exc.add_note(
                            "Download stopped because its next checkpoint could not be saved. "
                            "Free disk space or restore cache access, then re-run to resume."
                        )
                        raise
                    except (TypeError, ValueError) as exc:
                        self._cache_warning(exc)
                        journal.close()
                        journal = None
                        self.cache_info["status"] = "disabled"
                self.cache_info.update(
                    status=("resumed" if self.cache_info["cached_rows"] else "miss")
                    if journal
                    else "disabled",
                    downloaded_rows=self.cache_info["downloaded_rows"] + len(frame),
                    complete=complete,
                )
                if not frame.empty or (
                    complete
                    and not self.cache_info["downloaded_rows"]
                    and not self.cache_info["cached_rows"]
                ):
                    yield frame
                cursor = response.get("next_cursor")
                if complete:
                    return
        finally:
            if journal:
                journal.close()

    @staticmethod
    def _cache_warning(error: Exception) -> None:
        warnings.warn(
            f"Dataset caching failed ({type(error).__name__}); this transfer may not resume.",
            RuntimeWarning,
            stacklevel=3,
        )

    def _cache_validation(self, definition: dict[str, Any]) -> str:
        if definition.get("cache_token"):
            return "token"
        return "snapshot" if self.cache_ttl is None else "ttl"

    def _validate_download(self, definition: dict[str, Any]) -> None:
        latest = self.describe(refresh=True)
        if (
            latest.get("cache_token") != definition.get("cache_token")
            or latest["version"] != definition["version"]
        ):
            raise ValueError(
                "Dataset data changed while loading; retry to get a coherent snapshot."
            )

    def _selection_metadata(
        self, definition: dict[str, Any], max_rows: int | None = None
    ) -> dict[str, Any]:
        return {
            "dataset_reference": self.reference,
            "version": definition.get("version"),
            "max_rows": max_rows,
            **(definition.get("selection") or self._selection),
        }

    def _reset_download(self, definition: dict[str, Any]) -> None:
        from anomx.data.download import DatasetDownload

        store = self._cache_store()
        if store and definition.get("version"):
            with DatasetDownload(
                store.path.with_suffix(".parts"), definition=definition, refresh=True
            ):
                pass

    def _cache_store(self, max_rows: int | None = None) -> DatasetCache | None:
        if not self.cache or not self.reference:
            return None
        namespace = getattr(self._get_client(), "cache_namespace", None)
        if not namespace:
            return None
        directory = (
            os.environ.get("ANOMX_CACHE_DIR", "~/.anomx/cache/datasets")
            if self.cache is True
            else self.cache
        )
        return DatasetCache(
            directory,
            namespace=namespace,
            reference=(self.reference if max_rows is None else f"{self.reference}:first:{max_rows}")
            + (
                ":" + json.dumps(self._selection, sort_keys=True, separators=(",", ":"))
                if self._selection
                else ""
            ),
        )

    def clear_cache(self, *, max_rows: int | None = None) -> None:
        """Clear this selection's saved pages without contacting the host.

        Pages are shared by all prefix lengths. ``max_rows`` also selects an
        old-format materialized prefix entry to remove, when one exists.
        """
        from anomx.data.download import DatasetDownload

        store = self._cache_store(max_rows)
        if store:
            store.clear()
        pages = self._cache_store()
        if pages:
            pages.clear()
            DatasetDownload.clear(pages.path.with_suffix(".parts"))
        self.cache_info = {"status": "cleared"}

    def to_dataframe(
        self,
        *,
        batch_size: int = 5000,
        progress: bool | None = None,
        refresh: bool = False,
        max_rows: int | None = None,
    ) -> pd.DataFrame:
        """Materialize a dataset or view, resuming saved pages automatically.

        Every load resolves current metadata and permissions. Interrupted
        downloads resume at a pinned source version, even beyond cache_ttl.
        Without a content validator this is a resumable local snapshot, not a
        guarantee against upstream corrections. Completed downloads require a
        matching content token or an explicit TTL; ``cache_ttl=None`` opts into
        snapshot reuse until refresh or a source version change.

        ``refresh=True`` restarts the download. ``max_rows`` caps output rows
        after selection/aggregation; whole downloaded pages are retained so a
        later larger request can reuse their tails. ``iter_batches`` uses the
        same checkpoints without materializing the full dataset in memory.
        """
        if max_rows is not None and (
            isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1
        ):
            raise ValueError("max_rows must be a positive integer or None.")
        if self._data is not None:
            frame = select_frame(self._data, self._selection).iloc[:max_rows].copy()
            if self._selection or max_rows is not None:
                frame.attrs["anomx_selection"] = {
                    **frame.attrs.get("anomx_selection", {}),
                    **self._selection,
                    "max_rows": max_rows,
                }
            return frame
        enabled = self.progress if progress is None else progress
        with operation("resolve dataset", enabled=enabled):
            definition = self.describe(refresh=True) if self.reference else {}
        store = self._cache_store(max_rows)
        selection = self._selection_metadata(definition, max_rows)
        # Read old completed JSON tables, but write only page journals from now on.
        pages = self._cache_store()
        if store and not refresh and not (pages and pages.path.with_suffix(".parts").exists()):
            cached = store.read(definition, max_age=self.cache_ttl)
            if cached is not None:
                self.cache_info = {
                    "status": "hit",
                    "rows": len(cached),
                    "cached_rows": len(cached),
                    "downloaded_rows": 0,
                    "validation": self._cache_validation(definition),
                }
                if enabled:
                    from anomx.console import status

                    status(f"cache hit · {len(cached):,} rows")
                cached.attrs["anomx_selection"] = selection
                return cached
        if refresh:
            self._reset_download(definition)
            if store:
                store.clear()
        self.cache_info = {"status": "disabled", "cached_rows": 0, "downloaded_rows": 0}
        with progress_bar(
            self.name or "dataset", enabled=enabled, unit=" rows", unit_scale=True
        ) as bar:
            batches = []
            loaded = 0
            with closing(self._iter_batches(batch_size, definition)) as stream:
                for batch in stream:
                    if max_rows is not None:
                        batch = batch.iloc[: max_rows - loaded]
                    batches.append(batch)
                    loaded += len(batch)
                    bar.update(len(batch))
                    bar.set_postfix(cached=self.cache_info.get("cached_rows", 0), refresh=False)
                    if max_rows is not None and loaded >= max_rows:
                        break
        frame = (
            pd.concat(batches, ignore_index=self.kind != "time_series")
            if batches
            else pd.DataFrame()
        )
        if definition.get("cache_token") and not self.cache_info.get("complete"):
            try:
                self._validate_download(definition)
            except ValueError:
                self._reset_download(definition)
                raise
        self.cache_info["rows"] = len(frame)
        if store and not definition.get("cache_token"):
            self.cache_info["reason"] = (
                "Host has no data validator; resumed pages are a version-pinned local snapshot."
            )
            if self.cache_info.get("cached_rows") and "validation" not in self.cache_info:
                self.cache_info["validation"] = (
                    "snapshot"
                    if not self.cache_info.get("complete")
                    else self._cache_validation(definition)
                )
        frame.attrs["anomx_selection"] = selection
        return frame

    def to_timeseries(
        self,
        *,
        frequency: str | None = None,
        aggregation: str = "mean",
        fill: Literal["ffill", "interpolate"] | None = None,
        value_columns: Sequence[str] | None = None,
        progress: bool | None = None,
    ) -> TimeSeries:
        """Create an Anomx TimeSeries, optionally resampling irregular observations.

        The native forecasting container uses regular, timezone-free indices;
        aware times are normalized to UTC and their original timezone is stored
        in metadata. Resampling leaves empty bins as NaN unless ``fill`` is
        explicitly selected. Without a frequency, irregular data is rejected
        by TimeSeries; the original observations remain available as a frame.
        """
        from anomx.data.timeseries import TimeSeries

        frame = self.to_dataframe(progress=progress)
        if self.kind != "time_series":
            raise ValueError("to_timeseries requires a time_series dataset.")
        if value_columns is not None:
            frame = frame[list(value_columns)]
        metadata = dict(self.metadata)
        if "anomx_selection" in frame.attrs:
            metadata.setdefault("selection", frame.attrs["anomx_selection"])
        if self.reference is not None:
            metadata["dataset_reference"] = self.reference
        if isinstance(frame.index, pd.DatetimeIndex):
            if frame.index.tz is not None:
                metadata["source_timezone"] = str(frame.index.tz)
                frame.index = frame.index.tz_convert("UTC").tz_localize(None)
            if frequency:
                frame = frame.resample(frequency).agg(aggregation)
        if fill == "ffill":
            frame = frame.ffill()
        elif fill == "interpolate":
            frame = frame.interpolate(method="linear", limit_area="inside")
        elif fill is not None:
            raise ValueError("fill must be None, 'ffill', or 'interpolate'.")
        return TimeSeries.from_dataframe(frame, freq=frequency, metadata=metadata)

    def plot(self, *, ax: Any = None, **kwargs: Any) -> Any:  # noqa: ANN401
        """Plot observations on a Matplotlib axes and return that axes."""
        return self.to_dataframe().plot(ax=ax, **kwargs)

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
            data[time_column] = parse_timestamps(data[time_column])
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
