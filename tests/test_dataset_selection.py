"""Public dataset selection contracts, independent of live APIs or credentials."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from anomx.data import Dataset
from anomx.data.sources.platform import PlatformDatasetClient
from anomx.integrations.platform import PlatformClient, PlatformError


def local_dataset():
    frame = pd.DataFrame(
        {"value": [2.0, 4.0, 8.0, 16.0, 32.0]},
        index=pd.to_datetime(
            [
                "2026-10-01T00:00:05Z",
                "2026-10-01T00:00:25Z",
                "2026-10-01T00:01:05Z",
                "2026-10-01T00:01:25Z",
                "2026-10-01T00:03:05Z",
            ]
        ),
    )
    frame.index.name = "timestamp"
    return Dataset.from_dataframe(frame, name="detuning", metadata={"unit": "Hz"})


class SelectionHost:
    """A host that explicitly supports selections without widening batch arguments."""

    def __init__(self):
        self.calls = []
        self.read_calls = []

    def describe_dataset(self, reference):
        self.calls.append(("base", reference))
        return {
            "version": "base-version",
            "kind": "time_series",
            "mode": "fixed",
            "time_column": "timestamp",
        }

    def describe_dataset_selection(self, reference, *, selection):
        self.calls.append(("selection", reference, deepcopy(selection)))
        return {
            "version": "selected-version",
            "kind": "time_series",
            "mode": "fixed",
            "time_column": "timestamp",
            "capabilities": {"selection": True},
            "selection": deepcopy(selection),
        }

    def read_dataset(self, reference, *, version, cursor=None, limit=5000):
        self.read_calls.append((reference, version, cursor, limit))
        # This already-selected server response must not be downsampled again.
        return {
            "version": version,
            "columns": ["timestamp", "value"],
            "records": [{"timestamp": "2026-10-01T00:01:00Z", "value": 12.0}],
            "next_cursor": None,
        }


def test_selection_is_lazy_and_does_not_mutate_original_dataset():
    host = SelectionHost()
    original = Dataset("dataset", client=host, cache=False, progress=False)
    selected = original.select(start="2026-10-01T00:00:30Z", end="2026-10-01T00:02:00Z")
    downsampled = selected.downsample("1min", aggregation="mean")
    assert selected is not original
    assert downsampled is not selected
    assert isinstance(downsampled, Dataset)
    assert host.calls == []
    assert host.read_calls == []
    downsampled.to_dataframe()
    assert host.calls[0][0] == "selection"
    selection = host.calls[0][2]
    assert pd.Timestamp(selection["start"]) == pd.Timestamp("2026-10-01T00:00:30Z")
    assert pd.Timestamp(selection["end"]) == pd.Timestamp("2026-10-01T00:02:00Z")
    assert pd.tseries.frequencies.to_offset(selection["frequency"]) == pd.offsets.Minute()
    assert selection["aggregation"] == "mean"
    assert original.describe()["version"] == "base-version"
    assert host.calls[-1] == ("base", "dataset")


def test_local_time_selection_uses_inclusive_start_and_exclusive_end():
    original = local_dataset()
    selected = original.select(start="2026-10-01T02:00:25+02:00", end="2026-10-01T02:01:25+02:00")
    assert selected.to_dataframe().value.tolist() == [4.0, 8.0]
    assert original.to_dataframe().value.tolist() == [2.0, 4.0, 8.0, 16.0, 32.0]
    assert str(selected.to_dataframe().index.tz) == "UTC"


def test_selection_filters_before_downsampling_and_preserves_selection_provenance():
    frame = (
        local_dataset()
        .select(start="2026-10-01T00:00:20Z", end="2026-10-01T00:02:00Z")
        .downsample("1min")
        .to_dataframe()
    )
    np.testing.assert_allclose(frame.value, [4.0, 12.0])
    assert frame.index.tolist() == [
        pd.Timestamp("2026-10-01T00:00:00Z"),
        pd.Timestamp("2026-10-01T00:01:00Z"),
    ]
    attrs = frame.attrs["anomx_selection"]
    assert pd.Timestamp(attrs["start"]) == pd.Timestamp("2026-10-01T00:00:20Z")
    assert pd.Timestamp(attrs["end"]) == pd.Timestamp("2026-10-01T00:02:00Z")
    assert pd.tseries.frequencies.to_offset(attrs["frequency"]) == pd.offsets.Minute()
    assert attrs["aggregation"] == "mean"


@pytest.mark.parametrize(
    "aggregation,expected",
    [
        ("mean", [3, 12, 32]),
        ("last", [4, 16, 32]),
        ("first", [2, 8, 32]),
        ("min", [2, 8, 32]),
        ("max", [4, 16, 32]),
        ("sum", [6, 24, 32]),
        ("count", [2, 2, 1]),
    ],
)
def test_downsampling_omits_empty_bins_and_never_forward_fills(aggregation, expected):
    frame = local_dataset().downsample("1min", aggregation=aggregation).to_dataframe()
    np.testing.assert_allclose(frame.value, expected)
    assert pd.Timestamp("2026-10-01T00:02:00Z") not in frame.index


def test_fixed_frequency_bins_are_epoch_anchored_across_midnight():
    times = pd.to_datetime(["2026-10-01T23:59:59Z", "2026-10-02T00:00:01Z"])
    dataset = Dataset.from_dataframe(pd.DataFrame({"value": [2.0, 8.0]}, index=times))
    actual = dataset.downsample("7min").to_dataframe()
    expected = (
        pd.DataFrame({"value": [2.0, 8.0]}, index=times)
        .resample("7min", origin="epoch", label="left", closed="left")
        .mean()
        .dropna(how="all")
    )
    pd.testing.assert_frame_equal(actual, expected, check_freq=False)


def test_max_rows_limits_output_bins_after_selection_and_aggregation():
    actual = local_dataset().downsample("1min", aggregation="sum").to_dataframe(max_rows=2)
    assert actual.value.tolist() == [6.0, 24.0]
    assert actual.attrs["anomx_selection"]["max_rows"] == 2


def test_selection_streaming_matches_materialization_for_split_bins():
    selected = local_dataset().downsample("1min", aggregation="mean")
    expected = selected.to_dataframe()
    batches = list(selected.iter_batches(batch_size=1, progress=False))
    actual = pd.concat(batches)
    pd.testing.assert_frame_equal(actual, expected, check_freq=False)


def test_remote_batches_are_pinned_to_selected_version_with_unchanged_batch_signature():
    host = SelectionHost()
    dataset = (
        Dataset("dataset", client=host, cache=False, progress=False)
        .select(start="2026-10-01T00:00:30Z", end="2026-10-01T00:02:00Z")
        .downsample("1min")
    )
    frame = dataset.to_dataframe(batch_size=7)
    assert frame.value.tolist() == [12.0]
    assert host.read_calls == [("dataset", "selected-version", None, 7)]


def test_legacy_host_without_selection_support_fails_explicitly_before_loading():
    class OldHost:
        def describe_dataset(self, reference):
            return {"version": "base", "kind": "time_series", "mode": "fixed"}

        def read_dataset(self, reference, *, version, cursor=None, limit=5000):
            raise AssertionError("Unsupported selections must not download the entire dataset.")

    dataset = Dataset("dataset", client=OldHost(), cache=False, progress=False)
    with pytest.raises(
        (ValueError, TypeError, NotImplementedError, RuntimeError), match="[Ss]elect"
    ):
        dataset.select(start="2026-10-01T00:00:00Z").to_dataframe()


@pytest.mark.parametrize(
    "changes",
    [
        {"capabilities": None},
        {"capabilities": {}},
        {"capabilities": {"selection": False}},
        {"selection": None},
    ],
)
def test_platform_adapter_rejects_hosts_that_do_not_confirm_selection(monkeypatch, changes):
    selection = {"start": "2026-10-01T00:00:00Z", "end": "2026-10-01T01:00:00Z"}
    payload = {
        "version": "selected",
        "kind": "time_series",
        "mode": "fixed",
        "selection": selection,
        "capabilities": {"selection": True},
        **changes,
    }
    monkeypatch.setattr(PlatformClient, "request", lambda *args, **kwargs: deepcopy(payload))
    client = PlatformDatasetClient(PlatformClient("https://example.test", "test-token"))
    with pytest.raises(PlatformError, match="[Ss]elect"):
        client.describe_dataset_selection("dataset", selection=selection)


@pytest.mark.parametrize("frequency", ["0s", "-1min", "1ms", "1ME", "invalid", 60])
def test_invalid_or_nonfixed_downsampling_frequency_rejected_without_remote_access(frequency):
    host = SelectionHost()
    dataset = Dataset("dataset", client=host, cache=False, progress=False)
    with pytest.raises((ValueError, TypeError)):
        dataset.downsample(frequency).to_dataframe()
    assert host.calls == []
    assert host.read_calls == []


@pytest.mark.parametrize(
    "start,end",
    [
        ("2026-10-02T00:00:00Z", "2026-10-01T00:00:00Z"),
        ("2026-10-01T00:00:00Z", "2026-10-01T00:00:00Z"),
        ("invalid", "2026-10-01T00:00:00Z"),
    ],
)
def test_invalid_time_selection_rejected_without_remote_access(start, end):
    host = SelectionHost()
    dataset = Dataset("dataset", client=host, cache=False, progress=False)
    with pytest.raises(ValueError):
        dataset.select(start=start, end=end).to_dataframe()
    assert host.calls == []
    assert host.read_calls == []


def test_invalid_aggregation_rejected_without_remote_access():
    host = SelectionHost()
    dataset = Dataset("dataset", client=host, cache=False, progress=False)
    with pytest.raises(ValueError):
        dataset.downsample("1min", aggregation="median").to_dataframe()
    assert host.calls == []


def test_selected_empty_local_dataset_retains_columns_and_datetime_index():
    result = local_dataset().select(start="2027-01-01T00:00:00Z").downsample("1min").to_dataframe()
    assert result.empty
    assert result.columns.tolist() == ["value"]
    assert isinstance(result.index, pd.DatetimeIndex)


@pytest.mark.parametrize(
    "effective",
    [
        {"start": "2026-10-01T00:00:00Z", "end": "2026-10-02T00:00:00Z"},
        {"frequency": "10s", "aggregation": "mean"},
        {"frequency": "1min", "aggregation": "last"},
    ],
)
def test_platform_adapter_rejects_ignored_or_changed_downsampling(monkeypatch, effective):
    payload = {"version": "v1", "capabilities": {"selection": True}, "selection": effective}
    monkeypatch.setattr(PlatformClient, "request", lambda *args, **kwargs: deepcopy(payload))
    source = PlatformDatasetClient(PlatformClient("https://example.test", "test-token"))
    with pytest.raises(PlatformError, match="downsampling"):
        source.describe_dataset_selection(
            "dataset", selection={"frequency": "1min", "aggregation": "mean"}
        )


def test_platform_adapter_accepts_normalized_equivalent_frequency(monkeypatch):
    payload = {
        "version": "v1",
        "capabilities": {"selection": True},
        "selection": {"frequency": "60000000000ns", "aggregation": "mean"},
    }
    monkeypatch.setattr(PlatformClient, "request", lambda *args, **kwargs: deepcopy(payload))
    source = PlatformDatasetClient(PlatformClient("https://example.test", "test-token"))
    assert source.describe_dataset_selection("dataset", selection={"frequency": "1min"}) == payload
