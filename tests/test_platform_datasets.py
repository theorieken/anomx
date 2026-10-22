"""Dataset-specific routes and payloads belong to the data source adapter."""

import io
import json
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from anomx.data import Dataset
from anomx.data.sources import PlatformDatasetClient
from anomx.execution import WorkContext
from anomx.integrations.platform import PlatformClient, PlatformError
from anomx.integrations.platform.tracking import PlatformBackend


@pytest.fixture
def send(monkeypatch):
    open_request = Mock()
    monkeypatch.setattr(
        "anomx.integrations.platform.client.build_opener",
        lambda *handlers: SimpleNamespace(open=open_request),
    )
    return open_request


def response(value):
    return io.BytesIO(json.dumps(value).encode())


def definition():
    return {"version": "v1", "kind": "time_series", "mode": "fixed", "time_column": "time"}


@pytest.mark.parametrize(
    "url,prefix",
    [
        ("https://example.test", "/api"),
        ("https://example.test/api", "/api"),
        ("https://example.test/api/v1", "/api/v1/data"),
    ],
)
def test_resolution_and_paging_use_transport_and_pin_version(url, prefix, send):
    send.side_effect = [
        response(definition()),
        response(definition()),
        *[
            response(
                {
                    "version": "v1",
                    "columns": ["time", "x"],
                    "records": [{"time": f"2026-10-21T00:00:0{i}Z", "x": i}],
                    "next_cursor": "next" if i == 0 else None,
                }
            )
            for i in range(2)
        ],
    ]
    client = PlatformClient(url, "secret")
    dataset = Dataset(
        anomx_reference="data_dataset-example", client=client, cache=False, progress=False
    )
    assert dataset.anomx_reference == dataset.reference == "data_dataset-example"
    assert dataset.describe()["version"] == "v1"
    batches = list(dataset.iter_batches(batch_size=1))
    assert [frame.x.iloc[0] for frame in batches] == [0, 1]
    assert all(str(frame.index.tz) == "UTC" for frame in batches)
    requests = [call.args[0] for call in send.call_args_list]
    base = f"https://example.test{prefix}/datasets/data_dataset-example"
    assert requests[0].full_url == base + "/resolve"
    assert requests[0].get_method() == "GET"
    assert requests[0].data is None
    assert requests[1].full_url == base + "/resolve"
    assert requests[1].get_method() == "GET"
    for i, request in enumerate(requests[2:]):
        assert request.full_url == base + "/data"
        assert request.get_method() == "POST"
        assert json.loads(request.data) == {
            "version": "v1",
            "cursor": None if i == 0 else "next",
            "limit": 1,
        }


def test_explicit_transport_takes_precedence_over_host(send):
    send.return_value = response(definition())
    host = Mock(side_effect=AssertionError("Explicit client must take precedence"))
    with WorkContext(callbacks={"describe_dataset": host}).activate():
        assert (
            Dataset("dataset", client=PlatformClient("https://example.test", "secret")).describe()[
                "version"
            ]
            == "v1"
        )
    host.assert_not_called()


def test_environment_fallback_accepts_site_origin(send, monkeypatch):
    monkeypatch.setenv("ANOMX_URL", "https://example.test")
    monkeypatch.setenv("ANOMX_TOKEN", "secret")
    send.return_value = response(definition())
    Dataset(anomx_reference="dataset").describe()
    assert send.call_args.args[0].full_url == "https://example.test/api/datasets/dataset/resolve"


def test_empty_storage_windows_advance_without_losing_the_dataset_schema(send):
    send.side_effect = [
        response(definition()),
        response(
            {"version": "v1", "columns": ["time", "x"], "records": [], "next_cursor": "window-2"}
        ),
        response(
            {
                "version": "v1",
                "columns": ["time", "x"],
                "records": [{"time": "2026-10-21T00:00:00Z", "x": 3}],
                "next_cursor": "window-3",
            }
        ),
        response({"version": "v1", "columns": ["time", "x"], "records": [], "next_cursor": None}),
    ]
    dataset = Dataset(
        anomx_reference="dataset",
        client=PlatformClient("https://example.test", "secret"),
        cache=False,
    )
    assert dataset.to_dataframe().x.tolist() == [3]
    requests = [call.args[0] for call in send.call_args_list[1:]]
    assert [json.loads(request.data)["cursor"] for request in requests] == [
        None,
        "window-2",
        "window-3",
    ]


@pytest.mark.parametrize(
    "url,path",
    [
        ("https://example.test/api", "/api/jobs/channel-data"),
        ("https://example.test/api/v1", "/api/v1/jobs/jobs/channel-data"),
    ],
)
def test_channel_endpoint_belongs_to_data_source(url, path, send):
    send.return_value = response({"records": [{"timestamp": "2026-10-21", "x": 2}]})
    dataset = Dataset.from_channels(["channel"], client=PlatformClient(url, "secret"))
    assert dataset.to_dataframe().x.tolist() == [2]
    request = send.call_args.args[0]
    assert request.full_url == "https://example.test" + path
    assert json.loads(request.data)["channels"] == ["channel"]


def test_darts_uses_same_channel_adapter(send):
    pytest.importorskip("darts")
    from anomx.integrations.darts.datasets import Dataset as DartsDataset

    send.return_value = response(
        {"records": [{"timestamp": f"2026-10-21T00:00:0{i}Z", "x": i} for i in range(3)]}
    )
    dataset = DartsDataset.from_channels(
        ["channel"], client=PlatformClient("https://example.test", "secret"), frequency="1s"
    )
    assert dataset.values().ravel().tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    "operation,arguments", [("describe_dataset", {}), ("read_dataset", {"version": "v1"})]
)
def test_dataset_response_validation_is_owned_by_data_adapter(operation, arguments, send):
    send.return_value = response([])
    source = PlatformDatasetClient(PlatformClient("https://example.test", "secret"))
    with pytest.raises(PlatformError, match="dataset API"):
        getattr(source, operation)("dataset", **arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        {"reference": "a", "anomx_reference": "b"},
        {"anomx_reference": ""},
    ],
)
def test_ambiguous_or_empty_reference_is_rejected(arguments):
    with pytest.raises(ValueError):
        Dataset(**arguments)


def test_run_backend_owns_its_endpoints(send):
    send.side_effect = [response({}), response({})]
    identifier = str(uuid4())
    backend = PlatformBackend(PlatformClient("https://example.test", "secret"))
    backend.write({"run_id": identifier, "kind": "start"})
    backend.write({"run_id": identifier, "kind": "history", "value": 2})
    requests = [call.args[0] for call in send.call_args_list]
    assert [request.full_url for request in requests] == [
        "https://example.test/api/runs/start",
        f"https://example.test/api/runs/{identifier}/events",
    ]
    assert all(request.get_method() == "POST" for request in requests)
