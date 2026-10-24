"""Read versioned external dataset selections through a local storage mount."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Generator, Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, BinaryIO

import numpy as np
import pandas as pd

from anomx.data._timestamps import parse_timestamps
from anomx.data.dataset import Dataset, DatasetClient

if TYPE_CHECKING:
    from anomx.integrations.platform import PlatformClient


class DatasetSourceChanged(ValueError):
    """A mounted source no longer matches its resolved metadata."""


def _source_timestamp(value: str) -> int:
    try:
        instant = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None:
        raise ValueError("Source modification timestamps must include a timezone.")
    return math.floor(instant.timestamp())


@dataclass(frozen=True)
class MountedSource:
    """One read-only file pinned by a dataset manifest.

    Size and modification time are checked before and after each opened read.
    WebDAV ETags are retained as provenance, not interpreted as content hashes.
    ``sha256()`` provides an explicit content fingerprint when required.
    """

    mount_root: Path
    entry: Mapping[str, Any]
    dataset_version: str

    def __post_init__(self) -> None:
        root = Path(self.mount_root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("The mount root must be an existing directory.")
        object.__setattr__(self, "mount_root", root)

    @property
    def source_path(self) -> str:
        """Return the integration-relative source path."""
        return str(self.entry["path"])

    @property
    def path(self) -> Path:
        """Resolve the path inside the explicitly configured mount root."""
        source = self.source_path
        if (
            not source.startswith("/")
            or "\\" in source
            or any(part in {".", ".."} for part in source.split("/"))
            or any(ord(character) < 32 for character in source)
        ):
            raise ValueError("Dataset source paths must be absolute integration paths.")
        path = (self.mount_root / PurePosixPath(source).relative_to("/")).resolve(strict=True)
        if not path.is_relative_to(self.mount_root):
            raise ValueError("The source resolves outside the configured mount root.")
        return path

    def _check(self, stat: Any) -> None:  # noqa: ANN401
        size = self.entry.get("size_bytes")
        modified = self.entry.get("modified_at")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0 or not modified:
            raise ValueError("Native reads require manifest file sizes and modification times.")
        if stat.st_size != size or math.floor(stat.st_mtime) != _source_timestamp(modified):
            raise DatasetSourceChanged(f"Mounted source changed: {self.source_path}")

    @contextmanager
    def open(self) -> Iterator[BinaryIO]:
        """Open only in binary read mode and check the opened file's identity."""
        import os
        import stat

        path = self.path
        # Non-blocking open lets us reject FIFOs/devices before any read. Refuse
        # a final-component symlink replacement after path resolution as well.
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("Mounted dataset sources must be regular files.")
            self._check(before)
            try:
                yield stream
            finally:
                after = os.fstat(stream.fileno())
                current = path.stat()
                self._check(after)
                identity = lambda item: (  # noqa: E731
                    item.st_dev,
                    item.st_ino,
                    item.st_size,
                    item.st_mtime_ns,
                    item.st_ctime_ns,
                )
                if identity(before) != identity(after) or identity(after) != identity(current):
                    raise DatasetSourceChanged(f"Source changed during read: {self.source_path}")

    def sha256(self) -> str:
        """Hash the source in bounded chunks without writing or copying it."""
        digest = hashlib.sha256()
        with self.open() as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def iter_numpy(self, batch_size: int = 128) -> Iterator[np.ndarray]:
        """Yield copied numeric first-axis NPY batches through a read-only mapping."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Batch size must be a positive integer.")
        with self.open() as stream:
            version = np.lib.format.read_magic(stream)
            if version not in {(1, 0), (2, 0)}:
                raise ValueError("Native NumPy reads support NPY versions 1 and 2.")
            reader = (
                np.lib.format.read_array_header_1_0
                if version == (1, 0)
                else np.lib.format.read_array_header_2_0
            )
            shape, fortran, dtype = reader(stream)
            if not shape or dtype.hasobject or dtype.kind not in "biuf" or len(shape) > 8:
                raise ValueError("Use a numeric NPY array with one to eight dimensions.")
            offset = stream.tell()
            if offset + math.prod(shape) * dtype.itemsize != self.entry["size_bytes"]:
                raise ValueError("The NPY payload size differs from its header.")
            row_bytes = math.prod(shape[1:]) * dtype.itemsize
            if row_bytes > 64 * 1024 * 1024:
                raise ValueError("A NumPy row exceeds 64 MiB; select smaller observations.")
            if not math.prod(shape):
                return
            count = min(batch_size, max(1, 64 * 1024 * 1024 // max(1, row_bytes)))
            array = np.memmap(
                stream,
                mode="r",
                dtype=dtype,
                shape=shape,
                offset=offset,
                order="F" if fortran else "C",
            )
            try:
                for start in range(0, shape[0], count):
                    yield np.array(array[start : start + count], copy=True)
            finally:
                array._mmap.close()


class MountedDataset(Dataset):
    """Use an authorized platform dataset through a native read-only mount.

    ``mount_root`` is the local equivalent of the integration's starting folder,
    NOT the selected dataset subfolder. It is explicit because storage mounts
    differ between facilities. Metadata access still uses the normal Dataset
    client or Work callbacks, and the resolved manifest is pinned for this object.
    """

    def __init__(
        self,
        reference: str,
        *,
        mount_root: str | Path,
        client: DatasetClient | PlatformClient | None = None,
    ) -> None:
        super().__init__(reference, client=client, cache=False)
        self.mount_root = Path(mount_root).expanduser().resolve(strict=True)
        if not self.mount_root.is_dir():
            raise ValueError("The mount root must be an existing directory.")

    def sources(self) -> tuple[MountedSource, ...]:
        """Return the selected files after validating the manifest's version."""
        return tuple(self.iter_sources())

    def iter_sources(self) -> Iterator[MountedSource]:
        """Page source fingerprints lazily for large folder inventories."""
        definition = self.describe()
        manifest = definition.get("source_manifest")
        if not isinstance(manifest, dict) or manifest.get("source") != "integration":
            raise ValueError("Native reads require a resolved external filesystem manifest.")
        fingerprint = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if fingerprint != definition["version"]:
            raise DatasetSourceChanged("The source manifest does not match the dataset version.")
        for entry in self.source_entries():
            yield MountedSource(self.mount_root, dict(entry), fingerprint)

    def _iter_batches(
        self,
        batch_size: int,
        definition: dict[str, Any],
    ) -> Generator[pd.DataFrame, None, None]:
        """Read NPY, CSV, Parquet or selected HDF5 datasets on the compute node.

        Other formats can be consumed via ``sources()[i].open()`` and an explicit
        safe reader, or through the standard API-backed ``Dataset``.
        Each frame's attrs preserve the source path, row offset and version.
        Close the iterator explicitly when stopping early (for example with
        ``contextlib.closing``) so source identity checks and cleanup complete.
        """
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not 1 <= batch_size <= 10000
        ):
            raise ValueError("Batch size must be between 1 and 10,000 rows.")
        config = definition["source_manifest"].get("configuration", {})
        if definition.get("selection"):
            raise ValueError(
                "Selected row/array windows currently require the API Dataset reader. "
                "Use iter_sources() for an explicit mounted reader."
            )
        columns: list[str] | None = None
        for source in self.iter_sources():
            offset = 0
            with closing(_source_frames(source, config, batch_size)) as frames:
                for frame in frames:
                    if config.get("columns"):
                        frame = frame.loc[:, config["columns"]]
                    provenance = config.get("source_columns", {})
                    if set(provenance.values()) & set(frame.columns):
                        raise ValueError("Source provenance columns conflict with data columns.")
                    if "path" in provenance:
                        frame[provenance["path"]] = source.source_path
                    if "row" in provenance:
                        frame[provenance["row"]] = np.arange(offset, offset + len(frame))
                    if columns is not None and list(frame.columns) != columns:
                        raise ValueError("Dataset files have different column schemas.")
                    columns = list(frame.columns)
                    frame.attrs.update(
                        source_path=source.source_path,
                        source_row_offset=offset,
                        dataset_version=source.dataset_version,
                    )
                    offset += len(frame)
                    time_column = config.get("time_column")
                    if self.kind == "time_series":
                        if time_column not in frame:
                            raise ValueError("Time series require the configured timestamp column.")
                        kwargs = {"unit": config["time_unit"]} if config.get("time_unit") else {}
                        if pd.api.types.is_numeric_dtype(frame[time_column]) and not kwargs:
                            raise ValueError("Numeric timestamps require an explicit time_unit.")
                        frame[time_column] = parse_timestamps(
                            frame[time_column], unit=config.get("time_unit")
                        )
                        frame = frame.set_index(time_column)
                    yield frame


class DCacheDataset(MountedDataset):
    """Native dCache reads on Maxwell or another node with a dCache mount."""

    def iter_sources(self) -> Iterator[MountedSource]:
        if (self.describe().get("storage") or {}).get("kind") != "dcache":
            raise ValueError("DCacheDataset requires a dCache storage integration.")
        yield from super().iter_sources()


def _source_frames(
    source: MountedSource, config: Mapping[str, Any], batch_size: int
) -> Iterator[pd.DataFrame]:
    suffix = PurePosixPath(source.source_path).suffix.lower()
    options = config.get("reader_options") or {}
    if suffix == ".npy":
        with closing(source.iter_numpy(batch_size)) as batches:
            for values in batches:
                yield pd.DataFrame({"value": list(values)})
        return
    with source.open() as stream:
        if suffix == ".csv":
            yield from pd.read_csv(stream, sep=options.get("delimiter", ","), chunksize=batch_size)
        elif suffix == ".parquet":
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(stream)
            for batch in parquet.iter_batches(batch_size=batch_size, columns=config.get("columns")):
                yield batch.to_pandas()
        elif suffix in {".h5", ".hd5", ".hdf5", ".hdf"}:
            yield from _hdf5_frames(stream, config, batch_size)
        else:
            raise ValueError(
                f"No native batch reader for {suffix}; use a source.open() reader or Dataset."
            )


def _hdf5_frames(
    stream: BinaryIO, config: Mapping[str, Any], batch_size: int
) -> Iterator[pd.DataFrame]:
    import h5py

    options = config.get("reader_options") or {}
    names = config.get("columns") or options.get("datasets")
    if not names or not isinstance(names, list) or len(names) > 128:
        raise ValueError("Select 1–128 explicit HDF5 dataset paths for native reads.")
    with h5py.File(stream, "r") as file:
        arrays = {}
        for name in names:
            item = file
            for part in name.strip("/").split("/"):
                if (
                    not part
                    or part in {".", ".."}
                    or not isinstance(item, h5py.Group)
                    or not isinstance(item.get(part, getlink=True), h5py.HardLink)
                ):
                    raise ValueError("External, symbolic or missing HDF5 links are unsupported.")
                item = item[part]
            if (
                not isinstance(item, h5py.Dataset)
                or not 1 <= item.ndim <= 8
                or item.dtype.kind not in "biufSU"
                or item.external
                or item.is_virtual
            ):
                raise ValueError("Select numeric or fixed-width string HDF5 datasets.")
            arrays[name] = item
            if item.chunks and math.prod(item.chunks) * item.dtype.itemsize > 64 * 1024 * 1024:
                raise ValueError("An HDF5 chunk exceeds the 64 MiB decode limit.")
        axis = options.get("sample_axis", 0)
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise ValueError("The HDF5 sample axis must be an integer.")
        if any(not -item.ndim <= axis < item.ndim for item in arrays.values()):
            raise ValueError("The HDF5 sample axis is outside an array's dimensions.")
        counts = {item.shape[axis] for item in arrays.values()}
        if len(counts) != 1:
            raise ValueError("Selected HDF5 arrays must have matching sample counts.")
        row_bytes = sum(
            math.prod(item.shape[: axis % item.ndim] + item.shape[axis % item.ndim + 1 :])
            * item.dtype.itemsize
            for item in arrays.values()
        )
        if row_bytes > 64 * 1024 * 1024:
            raise ValueError("One HDF5 row exceeds 64 MiB; select smaller observations.")
        count = min(batch_size, max(1, 64 * 1024 * 1024 // max(1, row_bytes)))
        for start in range(0, counts.pop(), count):
            values = {}
            for name, item in arrays.items():
                selection = [slice(None)] * item.ndim
                selection[axis] = slice(start, start + count)
                values[name] = list(np.moveaxis(item[tuple(selection)], axis, 0))
            yield pd.DataFrame(values)


__all__ = ["DCacheDataset", "DatasetSourceChanged", "MountedDataset", "MountedSource"]
