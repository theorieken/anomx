"""Regression checks for validated dataset caching and native series conversion."""

import json
import warnings
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from anomx.data import Dataset, TimeSeries
from anomx.data.cache import DatasetCache
from anomx.data.download import DatasetDownload
from anomx.data.sources.platform import PlatformDatasetClient
from anomx.integrations.platform import PlatformClient

REFERENCE = "data_dataset-test"


class RemoteDataset:
    cache_namespace = "test-authorization-scope"

    def __init__(self, *, token="revision-1", unit=None, records=None):
        self.definition = {
            "version": "selection-1",
            "cache_token": token,
            "name": "detuning",
            "kind": "time_series",
            "mode": "automatic",
            "time_column": "timestamp",
            "time_unit": unit,
        }
        self.records = records or [
            {"timestamp": "2026-10-01T00:00:00.123456789Z", "value": 1.25},
            {"timestamp": "2026-10-01T00:01:00Z", "value": 2.75},
            {"timestamp": "2026-10-01T02:02:00+02:00", "value": 3.5},
        ]
        self.describe_calls = 0
        self.read_calls = []
        self.deny = False
        self.fail_cursor = None
        self.on_read = None

    def describe_dataset(self, reference):
        assert reference == REFERENCE
        self.describe_calls += 1
        if self.deny:
            raise PermissionError("dataset access revoked")
        return deepcopy(self.definition)

    def read_dataset(self, reference, *, version, cursor=None, limit=5000):
        assert reference == REFERENCE
        self.read_calls.append((version, cursor, limit))
        if cursor is not None and cursor == self.fail_cursor:
            raise RuntimeError("interrupted page")
        offset = int(cursor or 0)
        end = offset + limit
        if self.on_read:
            self.on_read()
        return {
            "version": version,
            "columns": ["timestamp", "value"],
            "records": self.records[offset:end],
            "next_cursor": str(end) if end < len(self.records) else None,
        }


def make_dataset(tmp_path, client=None, **kwargs):
    return Dataset(
        REFERENCE,
        client=client or RemoteDataset(),
        cache=tmp_path,
        progress=False,
        **kwargs,
    )


def journal_pages(directory):
    return sorted(directory.glob("*.parts/page-*.json"))


def journal_manifest(directory):
    return json.loads(next(directory.glob("*.parts/manifest.json")).read_text())


def seed_legacy_cache(directory, client):
    """Produce a real old-format cache without also creating a page journal."""
    expected = Dataset(REFERENCE, client=client, cache=False, progress=False).to_dataframe()
    dataset = make_dataset(directory, client)
    store = dataset._cache_store()
    store.write(expected, client.definition)
    return dataset, expected, store.path


def test_mixed_timestamp_precision_offsets_and_nanoseconds_survive_cache(tmp_path):
    client = RemoteDataset()
    dataset = make_dataset(tmp_path, client)
    frame = dataset.to_dataframe(batch_size=2)
    assert frame.index.tolist() == [
        pd.Timestamp("2026-10-01T00:00:00.123456789Z"),
        pd.Timestamp("2026-10-01T00:01:00Z"),
        pd.Timestamp("2026-10-01T00:02:00Z"),
    ]
    pd.testing.assert_frame_equal(dataset.to_dataframe(), frame)
    assert len(client.read_calls) == 2
    assert dataset.cache_info["status"] == "hit"


@pytest.mark.parametrize("unit,scale", [("s", 1), ("ms", 1000), ("us", 10**6), ("ns", 10**9)])
def test_numeric_epoch_units_survive_materialization_and_cache(tmp_path, unit, scale):
    client = RemoteDataset(
        unit=unit,
        records=[{"timestamp": value * scale, "value": value} for value in (1, 2, 3)],
    )
    dataset = make_dataset(tmp_path, client)
    frame = dataset.to_dataframe()
    expected = pd.date_range("1970-01-01T00:00:01Z", periods=3, freq="s")
    pd.testing.assert_index_equal(frame.index, expected.rename("timestamp"))
    pd.testing.assert_frame_equal(dataset.to_dataframe(), frame)


def test_cache_hit_revalidates_metadata_and_permissions_without_refetching_rows(tmp_path):
    client = RemoteDataset()
    dataset = make_dataset(tmp_path, client)
    first = dataset.to_dataframe()
    describes = client.describe_calls
    second = dataset.to_dataframe()
    assert client.describe_calls == describes + 1
    assert len(client.read_calls) == 1
    pd.testing.assert_frame_equal(first, second)
    assert dataset.cache_info["validation"] == "token"


@pytest.mark.parametrize("field", ["cache_token", "version"])
def test_source_or_data_revision_invalidates_cache(tmp_path, field):
    client = RemoteDataset()
    dataset = make_dataset(tmp_path, client)
    dataset.to_dataframe()
    client.definition[field] = "changed"
    client.records[0]["value"] = 99.0
    assert dataset.to_dataframe().iloc[0, 0] == 99.0
    assert len(client.read_calls) == 2
    assert dataset.cache_info["status"] == "miss"


def test_access_revocation_never_falls_back_to_cached_rows(tmp_path):
    client = RemoteDataset()
    dataset = make_dataset(tmp_path, client)
    dataset.to_dataframe()
    client.deny = True
    with pytest.raises(PermissionError, match="revoked"):
        dataset.to_dataframe()
    assert len(client.read_calls) == 1


def test_failed_pagination_preserves_legacy_entry_and_keeps_incomplete_checkpoint(tmp_path):
    client = RemoteDataset()
    dataset, _, path = seed_legacy_cache(tmp_path, client)
    original = path.read_bytes()
    client.definition["cache_token"] = "revision-2"
    client.fail_cursor = "2"
    with pytest.raises(RuntimeError, match="interrupted page"):
        dataset.to_dataframe(batch_size=2)
    assert path.read_bytes() == original
    assert len(journal_pages(tmp_path)) == 1
    assert journal_manifest(tmp_path)["complete"] is False
    assert not list(tmp_path.rglob(".download-*"))


def test_changed_validator_during_fetch_rejects_incoherent_table(tmp_path):
    client = RemoteDataset()
    client.on_read = lambda: client.definition.update(cache_token="changed-during-fetch")
    dataset = make_dataset(tmp_path, client)
    with pytest.raises(ValueError, match="changed while loading"):
        dataset.to_dataframe()
    assert not list(tmp_path.glob("*.json"))
    assert not journal_pages(tmp_path)


@pytest.mark.parametrize("broken", ["not JSON", "[]", "null", '{"format":1}'])
def test_corrupt_or_incomplete_cache_is_a_miss(tmp_path, broken):
    client = RemoteDataset()
    dataset, expected, path = seed_legacy_cache(tmp_path, client)
    path.write_text(broken)
    pd.testing.assert_frame_equal(dataset.to_dataframe(), expected)
    assert len(client.read_calls) == 2


@pytest.mark.parametrize("table", [{}, [], None, 123])
def test_invalid_cached_table_type_is_a_miss(tmp_path, table):
    client = RemoteDataset()
    dataset, expected, path = seed_legacy_cache(tmp_path, client)
    entry = json.loads(path.read_text())
    entry["table"] = table
    path.write_text(json.dumps(entry))
    pd.testing.assert_frame_equal(dataset.to_dataframe(), expected)
    assert len(client.read_calls) == 2


def test_cache_checksum_failure_causes_download(tmp_path):
    client = RemoteDataset()
    dataset, expected, path = seed_legacy_cache(tmp_path, client)
    entry = json.loads(path.read_text())
    entry["sha256"] = "tampered"
    path.write_text(json.dumps(entry))
    pd.testing.assert_frame_equal(dataset.to_dataframe(), expected)
    assert len(client.read_calls) == 2


def test_cache_round_trips_float64_without_precision_loss(tmp_path):
    client = RemoteDataset(
        records=[
            {"timestamp": f"2026-10-01T00:0{i}:00Z", "value": value}
            for i, value in enumerate([1.2345678901234567, np.nextafter(1.0, 2.0), 1e-20])
        ]
    )
    dataset = make_dataset(tmp_path, client)
    first = dataset.to_dataframe()
    second = dataset.to_dataframe()
    np.testing.assert_array_equal(first.values, second.values)


def test_nonfinite_values_are_preserved_or_caching_is_explicitly_skipped(tmp_path):
    client = RemoteDataset(
        records=[
            {"timestamp": f"2026-10-01T00:0{i}:00Z", "value": value}
            for i, value in enumerate([float("inf"), float("-inf"), float("nan")])
        ]
    )
    dataset = make_dataset(tmp_path, client)
    with warnings.catch_warnings(record=True):
        first = dataset.to_dataframe()
        second = dataset.to_dataframe()
    np.testing.assert_array_equal(first.values, second.values)


def test_old_host_without_validator_does_not_silently_reuse_data(tmp_path):
    client = RemoteDataset(token=None)
    dataset = make_dataset(tmp_path, client)
    dataset.to_dataframe()
    dataset.to_dataframe()
    assert len(client.read_calls) == 2
    assert not list(tmp_path.glob("*.json"))
    assert "no data validator" in dataset.cache_info["reason"]
    assert len(journal_pages(tmp_path)) == 1


def test_explicit_ttl_allows_bounded_reuse_without_validator(tmp_path):
    client = RemoteDataset(token=None)
    dataset = make_dataset(tmp_path, client, cache_ttl=60)
    with patch("anomx.data.download.time.time", return_value=1000):
        dataset.to_dataframe()
    with patch("anomx.data.download.time.time", return_value=1059):
        dataset.to_dataframe()
        assert dataset.cache_info["validation"] == "ttl"
    assert len(client.read_calls) == 1
    with patch("anomx.data.download.time.time", return_value=1061):
        dataset.to_dataframe()
    assert len(client.read_calls) == 2


def test_refresh_bypasses_a_valid_cache_and_clear_removes_it(tmp_path):
    client = RemoteDataset()
    dataset = make_dataset(tmp_path, client)
    dataset.to_dataframe()
    dataset.to_dataframe(refresh=True)
    assert len(client.read_calls) == 2
    dataset.clear_cache()
    assert dataset.cache_info["status"] == "cleared"
    assert not list(tmp_path.glob("*.json"))
    assert not journal_pages(tmp_path)


def test_checkpoint_write_permission_failure_stops_download_before_yield(tmp_path):
    client = RemoteDataset()
    dataset = make_dataset(tmp_path, client)
    with (
        patch.object(DatasetDownload, "append", side_effect=PermissionError("read-only cache")),
        pytest.raises(PermissionError, match="read-only cache") as error,
    ):
        next(dataset.iter_batches(batch_size=1, progress=False))
    assert "Download stopped" in " ".join(error.value.__notes__)
    assert "re-run to resume" in " ".join(error.value.__notes__)
    assert [call[1] for call in client.read_calls] == [None]
    assert not journal_pages(tmp_path)


def test_checkpoint_disk_failure_preserves_prefix_for_new_instance_resume(tmp_path):
    client = RemoteDataset()
    original_append = DatasetDownload.append

    def fail_second_checkpoint(journal, response, *, request_cursor):
        if request_cursor == "1":
            raise OSError("No space left on device")
        return original_append(journal, response, request_cursor=request_cursor)

    with (
        patch.object(DatasetDownload, "append", new=fail_second_checkpoint),
        pytest.raises(OSError, match="No space left") as error,
    ):
        make_dataset(tmp_path, client).to_dataframe(batch_size=1)
    assert "next checkpoint could not be saved" in " ".join(error.value.__notes__)
    assert [call[1] for call in client.read_calls] == [None, "1"]
    saved_page = journal_pages(tmp_path)[0].read_bytes()
    assert journal_manifest(tmp_path)["next_cursor"] == "1"
    assert journal_manifest(tmp_path)["row_count"] == 1
    assert journal_manifest(tmp_path)["complete"] is False

    resumed = make_dataset(tmp_path, client)
    frame = resumed.to_dataframe(batch_size=2)
    assert frame.value.tolist() == [1.25, 2.75, 3.5]
    assert [call[1] for call in client.read_calls] == [None, "1", "1"]
    assert journal_pages(tmp_path)[0].read_bytes() == saved_page
    assert resumed.cache_info["status"] == "resumed"
    assert resumed.cache_info["cached_rows"] == 1
    assert resumed.cache_info["downloaded_rows"] == 2
    assert journal_manifest(tmp_path)["complete"] is True


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize(
    "invalid_response,error_message",
    [
        ({"columns": ["timestamp", "value", "value"]}, "invalid columns"),
        ({"records": {"timestamp": ["2026-10-01T00:00:00Z"], "value": [1]}}, "invalid records"),
    ],
)
def test_invalid_remote_page_rejected_before_cache_or_yield(
    tmp_path, cache_enabled, invalid_response, error_message
):
    class InvalidClient(RemoteDataset):
        def read_dataset(self, reference, *, version, cursor=None, limit=5000):
            response = super().read_dataset(reference, version=version, cursor=cursor, limit=limit)
            return {**response, **invalid_response}

    dataset = Dataset(
        REFERENCE,
        client=InvalidClient(),
        cache=tmp_path if cache_enabled else False,
        progress=False,
    )
    with warnings.catch_warnings(), pytest.raises(ValueError, match=error_message):
        warnings.simplefilter("error", RuntimeWarning)
        next(dataset.iter_batches(progress=False))
    assert not journal_pages(tmp_path)


def test_cache_read_permission_failure_falls_back_to_authorized_download(tmp_path):
    client = RemoteDataset()
    dataset = make_dataset(tmp_path, client)
    expected = dataset.to_dataframe()
    with (
        patch("pathlib.Path.read_text", side_effect=PermissionError("unreadable cache")),
        pytest.warns(RuntimeWarning, match="caching failed"),
    ):
        pd.testing.assert_frame_equal(dataset.to_dataframe(), expected)
    assert len(client.read_calls) == 2


def test_authentication_and_server_identity_partition_cache_without_raw_tokens(tmp_path):
    clients = [
        PlatformClient("https://one.example", "secret-a"),
        PlatformClient("https://one.example", "secret-b"),
        PlatformClient("https://two.example", "secret-a"),
    ]
    namespaces = [PlatformDatasetClient(client).cache_namespace for client in clients]
    assert len(set(namespaces)) == 3
    paths = [
        DatasetCache(tmp_path, namespace=value, reference=REFERENCE).path for value in namespaces
    ]
    assert len(set(paths)) == 3
    assert all("secret-" not in str(path) for path in paths)


def test_custom_client_without_auth_namespace_does_not_persist(tmp_path):
    client = RemoteDataset()
    client.cache_namespace = None
    dataset = make_dataset(tmp_path, client)
    dataset.to_dataframe()
    dataset.to_dataframe()
    assert len(client.read_calls) == 2
    assert dataset.cache_info["status"] == "disabled"
    assert not list(tmp_path.iterdir())


def test_local_series_normalizes_utc_and_preserves_timezone_metadata_and_source():
    times = pd.date_range("2026-10-25T01:00:00", periods=5, freq="h", tz="Europe/Stockholm")
    frame = pd.DataFrame({"value": np.arange(5.0)}, index=times)
    dataset = Dataset.from_dataframe(frame, metadata={"facility": "XFEL"})
    series = dataset.to_timeseries(progress=False)
    assert isinstance(series, TimeSeries)
    pd.testing.assert_index_equal(series.time_index, times.tz_convert("UTC").tz_localize(None))
    assert series.metadata["source_timezone"] == "Europe/Stockholm"
    assert series.metadata["facility"] == "XFEL"
    pd.testing.assert_frame_equal(dataset.to_dataframe(), frame)


def test_irregular_series_requires_explicit_resampling_and_fill_policy():
    frame = pd.DataFrame(
        {"value": [0.0, 2.0, 6.0]},
        index=pd.to_datetime(
            ["2026-10-01T00:00:00Z", "2026-10-01T00:01:00Z", "2026-10-01T00:03:00Z"]
        ),
    )
    dataset = Dataset.from_dataframe(frame)
    with pytest.raises(ValueError, match="frequency"):
        dataset.to_timeseries(progress=False)
    series = dataset.to_timeseries(frequency="min", progress=False)
    np.testing.assert_array_equal(series.univariate_values(), [0.0, 2.0, np.nan, 6.0])
    interpolated = dataset.to_timeseries(frequency="min", fill="interpolate", progress=False)
    np.testing.assert_array_equal(interpolated.univariate_values(), [0.0, 2.0, 4.0, 6.0])
    forward = dataset.to_timeseries(frequency="min", fill="ffill", progress=False)
    np.testing.assert_array_equal(forward.univariate_values(), [0.0, 2.0, 2.0, 6.0])


def test_sample_dataset_does_not_gain_artificial_time_semantics():
    dataset = Dataset.from_dataframe(pd.DataFrame({"value": [1.0, 2.0, 3.0]}))
    with pytest.raises(ValueError, match="time_series dataset"):
        dataset.to_timeseries(progress=False)


@pytest.mark.parametrize("ttl", [-1, float("nan"), float("inf")])
def test_invalid_cache_ttl_rejected(tmp_path, ttl):
    with pytest.raises(ValueError, match="finite nonnegative"):
        make_dataset(tmp_path, cache_ttl=ttl)


def test_prefix_and_full_materialization_share_pages_and_keep_distinct_provenance(tmp_path):
    client = RemoteDataset()
    dataset = make_dataset(tmp_path, client)
    prefix = dataset.to_dataframe(max_rows=2, batch_size=1)
    assert len(prefix) == 2
    assert prefix.attrs["anomx_selection"] == {
        "dataset_reference": REFERENCE,
        "version": "selection-1",
        "max_rows": 2,
    }
    assert len(client.read_calls) == 2
    calls = len(client.read_calls)
    again = dataset.to_dataframe(max_rows=2)
    assert dataset.cache_info["status"] == "hit"
    assert len(client.read_calls) == calls
    pd.testing.assert_frame_equal(again, prefix)
    assert again.attrs == prefix.attrs

    full = dataset.to_dataframe()
    assert len(full) == 3
    assert full.attrs["anomx_selection"]["max_rows"] is None
    assert dataset.cache_info["status"] == "resumed"
    assert len(list(tmp_path.glob("*.parts"))) == 1
    assert not list(tmp_path.glob("*.json"))
    pd.testing.assert_frame_equal(dataset.to_dataframe(), full)
    assert dataset.cache_info["status"] == "hit"
    assert len(client.read_calls) == calls + 1


def test_distinct_prefix_sizes_reuse_full_downloaded_pages(tmp_path):
    client = RemoteDataset()
    dataset = make_dataset(tmp_path, client)
    first = dataset.to_dataframe(max_rows=1)
    second = dataset.to_dataframe(max_rows=2)
    assert first.value.tolist() == [1.25]
    assert second.value.tolist() == [1.25, 2.75]
    assert len(client.read_calls) == 1
    assert len(list(tmp_path.glob("*.parts"))) == 1
    assert not list(tmp_path.glob("*.json"))
    pd.testing.assert_frame_equal(dataset.to_dataframe(max_rows=1), first)
    assert len(client.read_calls) == 1


@pytest.mark.parametrize("max_rows", [0, -1, 1.5, True, False, "2"])
def test_invalid_prefix_limit_rejected_before_remote_access(tmp_path, max_rows):
    client = RemoteDataset()
    dataset = make_dataset(tmp_path, client)
    with pytest.raises(ValueError, match="max_rows"):
        dataset.to_dataframe(max_rows=max_rows)
    assert client.describe_calls == 0
    assert client.read_calls == []


def test_prefix_slices_last_page_and_closes_the_source_iterator(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path)
    closed = []
    advanced = []

    def batches(batch_size, definition):
        try:
            yield pd.DataFrame({"value": np.arange(5)}, index=pd.RangeIndex(5))
            advanced.append(True)
            yield pd.DataFrame({"value": np.arange(5, 10)}, index=pd.RangeIndex(5, 10))
        finally:
            closed.append(True)

    monkeypatch.setattr(dataset, "_iter_batches", batches)
    frame = dataset.to_dataframe(max_rows=2)
    assert frame.value.tolist() == [0, 1]
    assert closed == [True]
    assert advanced == []
    cached = dataset.to_dataframe(max_rows=2)
    assert cached.value.tolist() == [0, 1]
    assert cached.attrs["anomx_selection"]["max_rows"] == 2
    assert closed == [True, True]
    assert not journal_pages(tmp_path)


def test_short_source_prefix_preserves_requested_limit_without_claiming_full_data(tmp_path):
    dataset = make_dataset(tmp_path)
    frame = dataset.to_dataframe(max_rows=100)
    assert len(frame) == 3
    assert frame.attrs["anomx_selection"]["max_rows"] == 100
    assert len(dataset.to_dataframe()) == 3
    assert dataset.cache_info["status"] == "hit"


def test_clear_prefix_cache_clears_shared_pages_and_streaming_keeps_full_data(tmp_path):
    client = RemoteDataset()
    dataset = make_dataset(tmp_path, client)
    dataset.to_dataframe(max_rows=1)
    dataset.to_dataframe()
    assert sum(len(batch) for batch in dataset.iter_batches(batch_size=1, progress=False)) == 3
    assert len(client.read_calls) == 1
    dataset.clear_cache(max_rows=1)
    assert not journal_pages(tmp_path)
    calls = len(client.read_calls)
    assert len(dataset.to_dataframe()) == 3
    assert len(client.read_calls) == calls + 1
    assert len(dataset.to_dataframe(max_rows=1)) == 1
    assert len(client.read_calls) == calls + 1


def test_local_prefix_records_selection_without_mutating_original_attrs():
    frame = pd.DataFrame({"value": [1, 2, 3, 4]})
    frame.attrs["source"] = "local observations"
    dataset = Dataset.from_dataframe(frame)
    prefix = dataset.to_dataframe(max_rows=2)
    assert prefix.value.tolist() == [1, 2]
    assert prefix.attrs["anomx_selection"]["max_rows"] == 2
    assert prefix.attrs["source"] == "local observations"
    assert "anomx_selection" not in frame.attrs
    assert len(dataset.to_dataframe()) == 4


def test_native_conversion_preserves_explicit_source_reference_for_local_snapshot():
    frame = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    dataset = Dataset.from_dataframe(
        frame,
        kind="time_series",
        metadata={"dataset_reference": REFERENCE},
    )
    series = dataset.to_timeseries(progress=False)
    assert series.metadata["dataset_reference"] == REFERENCE


def test_default_cache_uses_shared_anomx_home(monkeypatch):
    from anomx.data import Dataset
    from anomx.integrations.platform import PlatformClient

    monkeypatch.delenv("ANOMX_CACHE_DIR", raising=False)
    dataset = Dataset(
        "sample", client=PlatformClient("https://example.test", "secret"), progress=False
    )
    assert dataset._cache_store().path.parent == Path.home() / ".anomx" / "cache" / "datasets"


def test_native_conversion_retains_remote_prefix_selection():
    from anomx.data import Dataset

    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    frame.attrs["anomx_selection"] = {"dataset_reference": "source", "version": "v1", "max_rows": 3}
    series = Dataset.from_dataframe(frame, kind="time_series").to_timeseries(progress=False)
    assert series.metadata["selection"] == frame.attrs["anomx_selection"]


def test_interrupted_materialization_resumes_saved_cursor_in_new_dataset_instance(tmp_path):
    client = RemoteDataset()
    client.fail_cursor = "2"
    with pytest.raises(RuntimeError, match="interrupted page"):
        make_dataset(tmp_path, client).to_dataframe(batch_size=2)
    assert journal_manifest(tmp_path)["next_cursor"] == "2"
    assert journal_manifest(tmp_path)["complete"] is False
    assert len(journal_pages(tmp_path)) == 1
    assert not list(tmp_path.glob("*.json"))

    client.fail_cursor = None
    resumed = make_dataset(tmp_path, client)
    frame = resumed.to_dataframe(batch_size=1)
    assert frame.value.tolist() == [1.25, 2.75, 3.5]
    assert [call[1] for call in client.read_calls] == [None, "2", "2"]
    assert resumed.cache_info["status"] == "resumed"
    assert resumed.cache_info["cached_rows"] == 2
    assert resumed.cache_info["downloaded_rows"] == 1
    assert journal_manifest(tmp_path)["complete"] is True


def test_streaming_commits_before_yield_and_new_iterator_continues_after_close(tmp_path):
    client = RemoteDataset()
    iterator = make_dataset(tmp_path, client).iter_batches(batch_size=1, progress=False)
    first = next(iterator)
    assert first.value.tolist() == [1.25]
    assert journal_manifest(tmp_path)["row_count"] == 1
    assert len(journal_pages(tmp_path)) == 1
    iterator.close()

    dataset = make_dataset(tmp_path, client)
    frames = list(dataset.iter_batches(batch_size=2, progress=False))
    assert pd.concat(frames).value.tolist() == [1.25, 2.75, 3.5]
    assert [call[1] for call in client.read_calls] == [None, "1"]
    assert dataset.cache_info["status"] == "resumed"


def test_prefix_stopping_mid_page_keeps_unreturned_tail_for_larger_request(tmp_path):
    client = RemoteDataset()
    first = make_dataset(tmp_path, client).to_dataframe(max_rows=1, batch_size=2)
    assert first.value.tolist() == [1.25]
    assert journal_manifest(tmp_path)["row_count"] == 2
    assert journal_manifest(tmp_path)["next_cursor"] == "2"

    second = make_dataset(tmp_path, client).to_dataframe(batch_size=1)
    assert second.value.tolist() == [1.25, 2.75, 3.5]
    assert [call[1] for call in client.read_calls] == [None, "2"]


def test_partial_download_outlives_ttl_without_renewing_earliest_page_age(tmp_path):
    client = RemoteDataset(token=None)
    client.fail_cursor = "1"
    with (
        patch("anomx.data.download.time.time", return_value=1000),
        pytest.raises(RuntimeError, match="interrupted page"),
    ):
        make_dataset(tmp_path, client, cache_ttl=10).to_dataframe(batch_size=1)
    client.fail_cursor = None
    dataset = make_dataset(tmp_path, client, cache_ttl=10)
    with patch("anomx.data.download.time.time", return_value=1060):
        assert dataset.to_dataframe().value.tolist() == [1.25, 2.75, 3.5]
        assert [call[1] for call in client.read_calls] == [None, "1", "1"]
        assert dataset.cache_info["status"] == "resumed"
        assert dataset.cache_info["validation"] == "snapshot"
        assert journal_manifest(tmp_path)["first_created"] == 1000
        assert "snapshot" in dataset.cache_info["reason"]
        # Completion does not reset the reused page's age; a later full load refreshes.
        dataset.to_dataframe()
    assert [call[1] for call in client.read_calls] == [None, "1", "1", None]


def test_none_ttl_explicitly_reuses_completed_snapshot_until_refresh_or_version_change(tmp_path):
    client = RemoteDataset(token=None)
    with patch("anomx.data.download.time.time", return_value=1000):
        original = make_dataset(tmp_path, client, cache_ttl=None).to_dataframe()
    dataset = make_dataset(tmp_path, client, cache_ttl=None)
    client.records[0]["value"] = 99.0
    with patch("anomx.data.download.time.time", return_value=10**7):
        pd.testing.assert_frame_equal(dataset.to_dataframe(), original)
        assert dataset.cache_info["validation"] == "snapshot"
        assert len(client.read_calls) == 1
        assert dataset.to_dataframe(refresh=True).value.iloc[0] == 99.0
        client.definition["version"] = "selection-2"
        client.records[0]["value"] = 123.0
        assert dataset.to_dataframe().value.iloc[0] == 123.0
    assert len(client.read_calls) == 3


def test_auth_revocation_blocks_partial_checkpoint_replay_without_network_batch_read(tmp_path):
    client = RemoteDataset()
    iterator = make_dataset(tmp_path, client).iter_batches(batch_size=1, progress=False)
    next(iterator)
    iterator.close()
    saved = [path.read_bytes() for path in journal_pages(tmp_path)]
    client.deny = True
    with pytest.raises(PermissionError, match="revoked"):
        make_dataset(tmp_path, client).to_dataframe()
    assert len(client.read_calls) == 1
    assert [path.read_bytes() for path in journal_pages(tmp_path)] == saved


def test_different_time_selections_have_independent_journals_and_clear_scope(tmp_path):
    class SelectionClient(RemoteDataset):
        def describe_dataset_selection(self, reference, *, selection):
            definition = self.describe_dataset(reference)
            definition.update(
                version="selected:" + json.dumps(selection, sort_keys=True),
                selection=deepcopy(selection),
                capabilities={"selection": True},
            )
            return definition

    client = SelectionClient()
    dataset = make_dataset(tmp_path, client)
    first = dataset.select(start="2026-10-01T00:00:00Z", end="2026-10-01T01:00:00Z")
    second = dataset.select(start="2026-10-01T00:00:00Z", end="2026-10-01T02:00:00Z")
    first.to_dataframe()
    second.to_dataframe()
    assert len(client.read_calls) == 2
    assert len(list(tmp_path.glob("*.parts"))) == 2
    first.clear_cache()
    second.to_dataframe()
    assert len(client.read_calls) == 2
    first.to_dataframe()
    assert len(client.read_calls) == 3


def test_invalid_checkpoint_page_checksum_redownloads_only_its_suffix(tmp_path):
    client = RemoteDataset()
    dataset = make_dataset(tmp_path, client)
    expected = dataset.to_dataframe(batch_size=1)
    pages = journal_pages(tmp_path)
    envelope = json.loads(pages[1].read_text())
    envelope["response"]["records"][0]["value"] = 999999
    pages[1].write_text(json.dumps(envelope))
    calls = len(client.read_calls)
    actual = make_dataset(tmp_path, client).to_dataframe(batch_size=1)
    pd.testing.assert_frame_equal(actual, expected)
    assert [call[1] for call in client.read_calls[calls:]] == ["1", "2"]


def test_fresh_load_still_checks_schema_when_resuming_checkpoint(tmp_path):
    class ChangingSchema(RemoteDataset):
        changed = False

        def read_dataset(self, reference, *, version, cursor=None, limit=5000):
            response = super().read_dataset(reference, version=version, cursor=cursor, limit=limit)
            if self.changed and cursor is not None:
                response["columns"] = ["timestamp", "other"]
                response["records"] = [
                    {"timestamp": row["timestamp"], "other": row["value"]}
                    for row in response["records"]
                ]
            return response

    client = ChangingSchema()
    dataset = make_dataset(tmp_path, client)
    iterator = dataset.iter_batches(batch_size=1, progress=False)
    next(iterator)
    iterator.close()
    client.changed = True
    with pytest.raises(ValueError, match="schema"):
        make_dataset(tmp_path, client).to_dataframe()
    assert journal_manifest(tmp_path)["complete"] is False
    assert len(journal_pages(tmp_path)) == 1


@pytest.mark.parametrize("identity_field", ["version", "cache_token"])
def test_changed_partial_source_identity_discards_stale_prefix(tmp_path, identity_field):
    client = RemoteDataset()
    dataset = make_dataset(tmp_path, client)
    dataset.to_dataframe(max_rows=1, batch_size=1)
    client.definition[identity_field] = "changed"
    client.records[0]["value"] = 99.0
    frame = make_dataset(tmp_path, client).to_dataframe()
    assert frame.value.tolist() == [99.0, 2.75, 3.5]
    assert [call[1] for call in client.read_calls] == [None, None]


def test_changed_timestamp_unit_discards_partial_checkpoint_even_with_same_token(tmp_path):
    client = RemoteDataset(
        unit="s", records=[{"timestamp": number, "value": number} for number in [1, 2, 3]]
    )
    first = make_dataset(tmp_path, client).to_dataframe(max_rows=1, batch_size=1)
    assert first.index[0] == pd.Timestamp("1970-01-01T00:00:01Z")
    client.definition["time_unit"] = "ms"
    second = make_dataset(tmp_path, client).to_dataframe()
    assert second.index[0] == pd.Timestamp("1970-01-01T00:00:00.001Z")
    assert [call[1] for call in client.read_calls] == [None, None]


def test_empty_and_short_remote_pages_are_journaled_until_explicit_eof(tmp_path):
    class SparseClient(RemoteDataset):
        def read_dataset(self, reference, *, version, cursor=None, limit=5000):
            self.read_calls.append((version, cursor, limit))
            records, next_cursor = {
                None: ([], "empty-window"),
                "empty-window": ([self.records[0]], "short-page"),
                "short-page": ([], None),
            }[cursor]
            return {
                "version": version,
                "columns": ["timestamp", "value"],
                "records": records,
                "next_cursor": next_cursor,
            }

    client = SparseClient()
    dataset = make_dataset(tmp_path, client)
    first = dataset.to_dataframe(batch_size=5000)
    assert first.value.tolist() == [1.25]
    assert [call[1] for call in client.read_calls] == [None, "empty-window", "short-page"]
    assert len(journal_pages(tmp_path)) == 3
    assert journal_manifest(tmp_path)["complete"] is True
    pd.testing.assert_frame_equal(make_dataset(tmp_path, client).to_dataframe(), first)
    assert len(client.read_calls) == 3


def test_clear_saved_download_does_not_resolve_revoked_or_offline_source(tmp_path):
    client = RemoteDataset()
    dataset = make_dataset(tmp_path, client)
    dataset.to_dataframe(max_rows=1, batch_size=1)
    describes = client.describe_calls
    client.deny = True
    dataset.clear_cache()
    assert client.describe_calls == describes
    assert not list(tmp_path.glob("*.parts/page-*.json"))
    assert dataset.cache_info["status"] == "cleared"
