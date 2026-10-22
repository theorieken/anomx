"""Datasets inside hosted work use callbacks, including selected cached views."""

import json

import pandas as pd
import pytest

from anomx.data import Dataset
from anomx.execution.work import WorkContext
from anomx.execution.work.data import WorkDatasetClient


def test_work_dataset_selection_and_cache_are_scoped_to_host_and_user(tmp_path):
    reads = []
    resolutions = []
    namespace = "host:user-a"
    authorized = True

    def describe(reference, selection):
        if not authorized:
            raise PermissionError("Dataset access revoked")
        resolutions.append(selection)
        return {
            "version": json.dumps(selection, sort_keys=True),
            "cache_token": "observations-1",
            "kind": "time_series",
            "mode": "automatic",
            "time_column": "timestamp",
            "selection": selection,
        }

    def read(reference, version, cursor=None, limit=5000):
        reads.append((reference, version))
        return {
            "version": version,
            "columns": ["timestamp", "value"],
            "records": [{"timestamp": "2026-10-01T00:00:00Z", "value": 3.0}],
            "next_cursor": None,
        }

    work = WorkContext(callbacks={
        "describe_dataset_selection": describe,
        "read_dataset": read,
        "dataset_cache_namespace": lambda: namespace,
    })
    with work.activate():
        dataset = Dataset("data_dataset-example", cache=tmp_path, progress=False).select(
            start="2026-10-01T00:00:00Z", end="2026-10-01T01:00:00Z"
        ).downsample("1min")
        first = dataset.to_dataframe()
        pd.testing.assert_frame_equal(dataset.to_dataframe(), first, check_freq=False)
        assert len(reads) == 1
        assert len(resolutions) >= 2  # Cache reuse still checks current authorization.
        assert json.loads(reads[0][1]) == resolutions[0]
        assert resolutions[0]["aggregation"] == "mean"

        dataset.downsample("5min").to_dataframe()
        assert len(reads) == 2
        namespace = "host:user-b"
        dataset.to_dataframe()
        assert len(reads) == 3
        authorized = False
        with pytest.raises(PermissionError, match="revoked"):
            dataset.to_dataframe()
        assert len(reads) == 3


def test_work_dataset_unsupported_selection_never_loads_unfiltered_data():
    def read(**arguments):
        pytest.fail("A selected dataset must not fall back to unfiltered reads")

    with (
        WorkContext(callbacks={"read_dataset": read}).activate(),
        pytest.raises(RuntimeError, match="describe_dataset_selection"),
    ):
        Dataset("dataset", progress=False).select(start="2026-10-01").to_dataframe()


def test_work_host_must_explicitly_opt_into_scoped_caching():
    assert WorkDatasetClient({}, None).cache_namespace is None
    with pytest.raises(ValueError, match="cache namespace"):
        _ = WorkDatasetClient({"dataset_cache_namespace": lambda: ""}, None).cache_namespace
