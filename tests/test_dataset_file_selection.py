"""Indexed folder selections stay lazy, version-pinned and bounded."""

from copy import deepcopy

import pytest

from anomx.data import Dataset
from anomx.execution.work.data import WorkDatasetClient


class Catalog:
    def __init__(self):
        self.calls = []

    def describe_dataset(self, reference):
        return {
            "version": "snapshot-v1",
            "kind": "sequence",
            "mode": "automatic",
            "source_manifest": {"snapshot": "s1"},
        }

    def describe_dataset_selection(self, reference, *, selection):
        self.calls.append(("select", deepcopy(selection)))
        return {
            **self.describe_dataset(reference),
            "selection": selection,
            "capabilities": {"selection": True},
        }

    def dataset_sources(self, reference, *, version, offset, limit):
        self.calls.append(("sources", version, offset, limit))
        entries = [{"path": "/a.npy"}, {"path": "/b.npy"}, {"path": "/c.npy"}]
        return {
            "version": version,
            "entries": entries[offset : offset + limit],
            "next_offset": offset + limit if offset + limit < len(entries) else None,
        }


def test_lazy_selection_and_multiple_catalog_pages():
    host = Catalog()
    base = Dataset("data_dataset-id", client=host, cache=False)
    selected = base.select_files(
        variables=["a"], row_start=2, row_end=10, row_step=3, sample_step=8
    )
    assert host.calls == []
    assert list(selected.source_entries(page_size=2)) == [
        {"path": "/a.npy"},
        {"path": "/b.npy"},
        {"path": "/c.npy"},
    ]
    assert host.calls == [
        (
            "select",
            {"variables": ["a"], "row_start": 2, "row_end": 10, "row_step": 3, "sample_step": 8},
        ),
        ("sources", "snapshot-v1", 0, 2),
        ("sources", "snapshot-v1", 2, 2),
    ]
    assert base.describe().get("selection", {}) == {}


@pytest.mark.parametrize(
    "arguments",
    [
        {"row_step": 0},
        {"sample_step": True},
        {"row_start": 4, "row_end": 4},
        {"path_prefix": "/a/../b"},
        {"variables": "scalar"},
        {"variables": []},
    ],
)
def test_bad_selections_fail_before_source_access(arguments):
    with pytest.raises(ValueError):
        Dataset("x", client=Catalog(), cache=False).select_files(**arguments)


def test_catalog_version_mismatch_is_rejected():
    host = Catalog()
    host.dataset_sources = lambda *args, **kwargs: {
        "version": "changed",
        "entries": [],
        "next_offset": None,
    }
    with pytest.raises(ValueError, match="pinned version"):
        list(Dataset("x", client=host, cache=False).source_entries())


def test_jobs_support_source_pagination_without_direct_network():
    seen = []
    client = WorkDatasetClient(
        {"dataset_sources": lambda **kwargs: seen.append(kwargs) or {"entries": []}}, None
    )
    assert client.dataset_sources("x", version="v", offset=50, limit=25) == {"entries": []}
    assert seen == [{"reference": "x", "version": "v", "offset": 50, "limit": 25}]
