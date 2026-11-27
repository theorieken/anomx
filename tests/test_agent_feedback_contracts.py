"""Feedback regressions must not turn failed or partial reads into false conclusions."""

import io
import json
import runpy
from pathlib import Path
from urllib.error import URLError

import pytest

from anomx.agent import AgentMode, AnomxHome
from anomx.agent.helpers.anomx_api import AnomxApiConnection, AnomxApiError, call_anomx_api
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks


class Response(io.BytesIO):
    status = 200
    headers = {"content-type": "application/json", "X-Anomx-Data-Scope": "accessible"}


@pytest.fixture
def runtime(tmp_path):
    home = AnomxHome(tmp_path / "home")
    home.ensure()
    home.set_platform_connection(url="https://platform.test/api", token="test-token")
    return AgentRuntime(home, tmp_path, mode=AgentMode.AUTOMATIC)


def test_api_tool_forwards_slow_read_timeout_and_preserves_access_scope(runtime, monkeypatch):
    deadlines = []

    def request(req, *, timeout):
        deadlines.append(timeout)
        return Response(b'{"items": [], "total": 0}')

    monkeypatch.setattr("anomx.agent.helpers.anomx_api.urlopen", request)
    result = json.loads(runtime._execute_tool(
        "use_anomx_api", {"method": "GET", "path": "/jobs", "timeout_seconds": 180},
        RuntimeCallbacks(),
    ))
    assert result["ok"] and deadlines == [180]
    assert result["scope"] == "accessible"
    assert "other accounts" in result["collection_note"]


@pytest.mark.parametrize("timeout", [None, True, "120", 0, -1, 301, float("inf"), float("nan")])
def test_invalid_timeout_never_sends_request(tmp_path, monkeypatch, timeout):
    monkeypatch.setattr(
        "anomx.agent.helpers.anomx_api.urlopen",
        lambda *_args, **_kwargs: pytest.fail("Invalid deadline sent a request"),
    )
    with pytest.raises(AnomxApiError, match="between 1 and 300"):
        call_anomx_api(
            AnomxApiConnection("https://platform.test/api", "secret", tmp_path),
            method="GET", path="/channels", timeout=timeout,
        )


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_timeout_is_not_empty_success_or_replayed_mutation(tmp_path, monkeypatch, method, wrapped):
    requests = []

    def request(req, **kwargs):
        requests.append(req)
        error = TimeoutError()
        raise URLError(error) if wrapped else error

    monkeypatch.setattr("anomx.agent.helpers.anomx_api.urlopen", request)
    result = call_anomx_api(
        AnomxApiConnection("https://platform.test/api", "secret", tmp_path),
        method=method, path="/events", timeout=120,
    )
    assert not result["ok"] and result["error_code"] == "request_timeout"
    assert result["timeout_seconds"] == 120 and len(requests) == 1
    assert result["retryable"] is (method == "GET")
    if method == "POST":
        assert "may already have completed" in result["error"]


@pytest.fixture
def read_collection():
    script = Path(__file__).parents[1] / "src/anomx/agent/skills/use-anomx-api/api.py"
    return runpy.run_path(str(script))["read_collection"]


@pytest.mark.parametrize("body", [[], {"items": []}, {"results": []}, {"objects": []}])
def test_collection_reader_accepts_empty_supported_shapes(tmp_path, read_collection, body):
    path = tmp_path / "response.json"
    path.write_text(json.dumps(body))
    assert read_collection({"ok": True, "response_path": str(path)}) == []


@pytest.mark.parametrize("body", [{"error": "unavailable"}, {"items": None}, {}, "error"])
def test_collection_reader_rejects_unknown_shape(tmp_path, read_collection, body):
    path = tmp_path / "response.json"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="Expected a collection"):
        read_collection({"ok": True, "response_path": str(path)})


def test_collection_reader_rejects_failed_calls_before_reading(read_collection):
    with pytest.raises(RuntimeError, match="HTTP 503"):
        read_collection({"ok": False, "status_code": 503})


def test_bounded_catalog_search_cannot_claim_channel_is_unregistered(runtime, monkeypatch):
    payload = {
        "items": [{"value": "XFEL/X/Y/Z", "source": "discovery", "channel": None}],
        "catalog_limited": True,
        "latest_registration_batch": {"state": "failed", "stored": 0, "expected": 100},
    }
    monkeypatch.setattr(
        "anomx.agent.helpers.anomx_api.urlopen",
        lambda *_args, **_kwargs: Response(json.dumps(payload).encode()),
    )
    result = json.loads(runtime._execute_tool(
        "search_anomx_data_channels", {"query": "XFEL/X/Y/", "refresh": False},
        RuntimeCallbacks(),
    ))
    assert result["ok"] and result["matches"][0]["availability"] == "unknown"
    assert result["latest_registration_batch"]["state"] == "failed"
