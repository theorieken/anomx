import hashlib
import json
import os
import subprocess
import sys
from email.utils import formatdate
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anomx import DatasetSourceChanged, DCacheDataset, MountedDataset, MountedSource, WorkContext


def definition(root, paths, *, configuration=None, kind="sequence"):
    entries = [
        {
            "path": "/" + path,
            "size_bytes": (root / path).stat().st_size,
            "modified_at": formatdate((root / path).stat().st_mtime, usegmt=True),
            "etag": "opaque-storage-etag",
        }
        for path in paths
    ]
    manifest = {"source": "integration", "entries": entries, "configuration": configuration or {}}
    return {
        "kind": kind,
        "mode": "fixed",
        "source_manifest": manifest,
        "storage": {"kind": "dcache", "root_path": "/remote"},
        "version": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def context(value):
    return WorkContext(callbacks={"describe_dataset": lambda **_: value}).activate()


@pytest.mark.parametrize("shape", [(7,), (7, 2), (7, 16, 2)])
@pytest.mark.parametrize("fortran", [False, True])
def test_native_numpy_preserves_arrays_and_source_provenance(tmp_path, shape, fortran):
    values = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    if fortran:
        values = np.asfortranarray(values)
    np.save(tmp_path / "data.npy", values)
    resolved = definition(
        tmp_path, ["data.npy"], configuration={"source_columns": {"path": "source", "row": "row"}}
    )
    with context(resolved):
        dataset = DCacheDataset("data_dataset-example", mount_root=tmp_path)
        batches = list(dataset.iter_batches(3))
        assert [len(frame) for frame in batches] == [3, 3, 1]
        assert all(frame.attrs["dataset_version"] == resolved["version"] for frame in batches)
        frame = pd.concat(batches)
        np.testing.assert_array_equal(np.stack(frame.value), values)
        assert frame.row.tolist() == list(range(7))
        assert frame.source.unique().tolist() == ["/data.npy"]
        assert (
            dataset.sources()[0].sha256()
            == hashlib.sha256((tmp_path / "data.npy").read_bytes()).hexdigest()
        )


def test_manifest_version_and_storage_kind_are_checked(tmp_path):
    (tmp_path / "data.csv").write_text("x\n1\n")
    resolved = definition(tmp_path, ["data.csv"])
    with context(resolved):
        resolved["source_manifest"]["entries"][0]["size_bytes"] += 1
        with pytest.raises(DatasetSourceChanged):
            MountedDataset("data_dataset-example", mount_root=tmp_path).sources()
        resolved["storage"]["kind"] = "other"
        with pytest.raises(ValueError, match="dCache storage"):
            DCacheDataset("data_dataset-example", mount_root=tmp_path).sources()


def test_changes_before_and_during_read_fail(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("x\n1\n")
    resolved = definition(tmp_path, ["data.csv"])
    with context(resolved):
        item = DCacheDataset("data_dataset-example", mount_root=tmp_path).sources()[0]
        with pytest.raises(DatasetSourceChanged), item.open():
            source.write_text("x\n123\n")
        with pytest.raises(DatasetSourceChanged), item.open():
            pass


@pytest.mark.parametrize("suffix", ["csv", "npy"])
def test_early_close_propagates_source_change(tmp_path, suffix):
    path = tmp_path / f"data.{suffix}"
    if suffix == "csv":
        path.write_text("x\n1\n2\n3\n")
    else:
        np.save(path, np.arange(12, dtype=np.float32).reshape(3, 4))
    with context(definition(tmp_path, [path.name])):
        batches = MountedDataset("data_dataset-example", mount_root=tmp_path).iter_batches(1)
        next(batches)
        # Append without invalidating an active memory map's original extent.
        with path.open("ab") as stream:
            stream.write(b"x")
        with pytest.raises(DatasetSourceChanged, match="changed"):
            batches.close()
        batches.close()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO boundary")
def test_nonregular_source_is_rejected_without_waiting_for_a_writer(tmp_path):
    os.mkfifo(tmp_path / "source.fifo")
    script = """
from pathlib import Path
import sys
from anomx.data.mounted import MountedSource
source = MountedSource(Path(sys.argv[1]), {"path": "/source.fifo"}, "fixture")
try:
    with source.open():
        raise AssertionError("FIFO was accepted")
except ValueError as error:
    assert "regular files" in str(error), str(error)
"""
    # A subprocess timeout catches a blocking FIFO open without hanging pytest.
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize("path", ["/../secret", "/nested/../../secret", "relative", "/data\\bad"])
def test_path_traversal_is_rejected(tmp_path, path):
    item = MountedSource(tmp_path, {"path": path}, "version")
    with pytest.raises(ValueError):
        _ = item.path


def test_symlink_escape_is_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("not selected")
    (root / "escape").symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        _ = MountedSource(root, {"path": "/escape"}, "version").path


def test_direct_source_normalizes_mount_alias(tmp_path, monkeypatch):
    root = tmp_path / "real"
    root.mkdir()
    path = root / "data.csv"
    path.write_text("x\n1\n")
    (tmp_path / "alias").symlink_to(root, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    entry = definition(root, ["data.csv"])["source_manifest"]["entries"][0]
    source = MountedSource(Path("alias"), entry, "fixture")
    assert source.mount_root == root.resolve()
    assert source.sha256() == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="directory"):
        MountedSource(path, entry, "fixture")


def test_timestamp_and_selection_are_preserved(tmp_path):
    (tmp_path / "data.csv").write_text("t,x,ignore\n1000,2,9\n2000,3,8\n")
    resolved = definition(
        tmp_path,
        ["data.csv"],
        kind="time_series",
        configuration={"columns": ["t", "x"], "time_column": "t", "time_unit": "ms"},
    )
    with context(resolved):
        result = DCacheDataset("data_dataset-example", mount_root=tmp_path).to_dataframe()
    assert result.index[0] == pd.Timestamp("1970-01-01T00:00:01Z")
    assert list(result.columns) == ["x"]


def test_missing_fingerprint_and_object_arrays_are_rejected(tmp_path):
    np.save(tmp_path / "objects.npy", np.array([{"code": "untrusted"}], dtype=object))
    resolved = definition(tmp_path, ["objects.npy"])
    with context(resolved):
        item = DCacheDataset("data_dataset-example", mount_root=tmp_path).sources()[0]
        with pytest.raises(ValueError, match="numeric NPY"):
            list(item.iter_numpy())
        missing = MountedSource(tmp_path, {**item.entry, "modified_at": ""}, item.dataset_version)
        with pytest.raises(ValueError, match="modification times"), missing.open():
            pass


def test_empty_numeric_arrays_are_valid(tmp_path):
    np.save(tmp_path / "empty.npy", np.empty((0, 2), dtype=np.float32))
    with context(definition(tmp_path, ["empty.npy"])):
        assert list(DCacheDataset("data_dataset-example", mount_root=tmp_path).iter_batches()) == []


@pytest.mark.parametrize("suffix", ["h5", "hd5", "hdf5", "hdf"])
def test_hdf5_explicit_axis_and_external_links(tmp_path, suffix):
    h5py = pytest.importorskip("h5py")
    values = np.arange(18).reshape(3, 6)
    filename = f"data.{suffix}"
    with h5py.File(tmp_path / filename, "w") as file:
        file.create_dataset("signal", data=values)
        file["external"] = h5py.ExternalLink("unselected.h5", "/anything")
    resolved = definition(
        tmp_path,
        [filename],
        configuration={"reader_options": {"datasets": ["signal"], "sample_axis": -1}},
    )
    with context(resolved):
        frame = DCacheDataset("data_dataset-example", mount_root=tmp_path).to_dataframe()
        np.testing.assert_array_equal(np.stack(frame.signal), values.T)
    rejected = definition(
        tmp_path, [filename], configuration={"reader_options": {"datasets": ["external"]}}
    )
    with context(rejected), pytest.raises(ValueError, match="links"):
        list(DCacheDataset("data_dataset-example", mount_root=tmp_path).iter_batches())


def test_same_size_timestamp_change_is_rejected(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x\n1\n")
    resolved = definition(tmp_path, ["data.csv"])
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 2))
    with context(resolved), pytest.raises(DatasetSourceChanged):
        list(DCacheDataset("data_dataset-example", mount_root=tmp_path).iter_batches())


def test_parquet_read_is_batched(tmp_path):
    pytest.importorskip("pyarrow")
    pd.DataFrame({"x": range(9)}).to_parquet(tmp_path / "data.parquet")
    with context(definition(tmp_path, ["data.parquet"])):
        frames = list(DCacheDataset("data_dataset-example", mount_root=tmp_path).iter_batches(4))
    assert [len(frame) for frame in frames] == [4, 4, 1]
    assert pd.concat(frames).x.tolist() == list(range(9))


def test_hdf5_rejects_large_decompressed_chunks_before_reading(tmp_path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "large.h5", "w") as file:
        file.create_dataset("signal", shape=(2, 8388608), chunks=(2, 8388608), dtype="float64")
    resolved = definition(
        tmp_path, ["large.h5"],
        configuration={"reader_options": {"datasets": ["signal"]}},
    )
    with context(resolved), pytest.raises(ValueError, match="chunk"):
        list(DCacheDataset("data_dataset-example", mount_root=tmp_path).iter_batches())
