"""Authentication recovery must never replay an accepted or forbidden mutation."""

import io
import json
from urllib.error import HTTPError

import pytest

from anomx.agent.helpers.anomx_api import (
    AnomxApiConnection,
    AnomxApiError,
    call_anomx_api,
    connection_from_home,
)
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.store import AnomxHome


class Response(io.BytesIO):
    status = 200
    headers = {"content-type": "application/json"}


def reject(request, status=403, payload=None):
    return HTTPError(
        request.full_url, status, "Rejected", {"content-type": "application/json"},
        io.BytesIO(json.dumps(payload or {"code": "invalid_bearer_token"}).encode()),
    )


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "PATCH", "DELETE"])
@pytest.mark.parametrize("status,payload", [
    (403, {"code": "invalid_bearer_token", "detail": "Translated message"}),
    (401, {"code": "invalid_bearer_token"}),
    (403, {"detail": "Valid bearer token required."}),
])
def test_auth_rejection_renews_once_and_preserves_request(
    tmp_path, monkeypatch, method, status, payload,
):
    requests, refreshes = [], []

    def token_provider(force):
        refreshes.append(force)
        return "renewed" if force else "expired"

    def request(req, **kwargs):
        requests.append(req)
        if len(requests) == 1:
            raise reject(req, status, payload)
        return Response(b'{"saved": true}')

    monkeypatch.setattr("anomx.agent.helpers.anomx_api.urlopen", request)
    connection = AnomxApiConnection("https://example.test/api", "secret", tmp_path, token_provider)
    result = call_anomx_api(connection, method=method, path="/events/1", body={"name": "Updated"})

    assert result["ok"] and result["response"] == {"saved": True}
    assert refreshes == [False, True]
    assert len(requests) == 2
    assert requests[0].data == requests[1].data
    assert requests[0].method == requests[1].method == method
    assert requests[0].get_header("Authorization") == "Bearer expired"
    assert requests[1].get_header("Authorization") == "Bearer renewed"
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert "secret" not in repr(connection)


@pytest.mark.parametrize("status,payload", [
    (403, {"detail": "Background mode cannot perform this operation."}),
    (401, {"detail": "External service refused access."}),
    (429, {"code": "invalid_bearer_token"}),
    (500, {"detail": "Internal server error"}),
    (302, {"detail": "Redirect"}),
])
def test_other_failures_never_replay_writes(tmp_path, monkeypatch, status, payload):
    requests, refreshes = [], []

    def request(req, **kwargs):
        requests.append(req)
        raise reject(req, status, payload)

    monkeypatch.setattr("anomx.agent.helpers.anomx_api.urlopen", request)
    connection = AnomxApiConnection(
        "https://example.test/api", "secret", tmp_path,
        lambda force: refreshes.append(force) or "secret",
    )
    result = call_anomx_api(connection, method="POST", path="/events", body={"name": "Created"})
    assert result["ok"] is False
    assert result["status_code"] == status
    assert len(requests) == 1
    assert refreshes == [False]


def test_transport_failure_after_write_is_never_replayed(tmp_path, monkeypatch):
    requests = []

    def request(req, **kwargs):
        requests.append(req)
        raise TimeoutError("Response lost after the server may have saved the event")

    monkeypatch.setattr("anomx.agent.helpers.anomx_api.urlopen", request)
    connection = AnomxApiConnection(
        "https://example.test/api", "secret", tmp_path, lambda _: "secret",
    )
    result = call_anomx_api(connection, method="POST", path="/events")
    assert result["ok"] is False
    assert len(requests) == 1


def test_repeated_auth_rejection_stops_after_one_retry(tmp_path, monkeypatch):
    requests = []

    def request(req, **kwargs):
        requests.append(req)
        raise reject(req)

    monkeypatch.setattr("anomx.agent.helpers.anomx_api.urlopen", request)
    connection = AnomxApiConnection(
        "https://example.test/api", "secret", tmp_path, lambda _: "secret",
    )
    result = call_anomx_api(connection, method="PATCH", path="/events/1", body={"name": "Updated"})
    assert result["ok"] is False
    assert len(requests) == 2


def test_revoked_credential_is_not_replayed_or_exposed(tmp_path, monkeypatch):
    requests = []

    def request(req, **kwargs):
        requests.append(req)
        raise reject(req)

    def refresh(force):
        if force:
            raise RuntimeError("DB error with sensitive-token-value")
        return "sensitive-token-value"

    monkeypatch.setattr("anomx.agent.helpers.anomx_api.urlopen", request)
    connection = AnomxApiConnection("https://example.test/api", "secret", tmp_path, refresh)
    with pytest.raises(AnomxApiError) as error:
        call_anomx_api(connection, method="DELETE", path="/events/1")
    assert "sensitive-token-value" not in str(error.value)
    assert len(requests) == 1


def test_cli_connection_without_worker_callback_does_not_renew(tmp_path, monkeypatch):
    requests = []

    def request(req, **kwargs):
        requests.append(req)
        raise reject(req)

    monkeypatch.setattr("anomx.agent.helpers.anomx_api.urlopen", request)
    result = call_anomx_api(
        AnomxApiConnection("https://example.test/api", "secret", tmp_path),
        method="GET", path="/events",
    )
    assert result["status_code"] == 403
    assert len(requests) == 1


def test_connection_rereads_renewed_credentials_and_shell_environment(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    home.set_platform_connection(url="https://example.test/api", token="old")
    runtime = AgentRuntime(home, tmp_path, mode=AgentMode.AUTOMATIC)

    def refresh(force):
        home.set_platform_connection(url="https://example.test/api", token="new")

    runtime.platform_token_refresh = refresh
    connection = connection_from_home(home, token_refresh=refresh)
    requests = []
    monkeypatch.setattr(
        "anomx.agent.helpers.anomx_api.urlopen",
        lambda req, **kwargs: requests.append(req) or Response(b"{}"),
    )
    assert call_anomx_api(connection, method="GET", path="/events")["ok"]
    assert requests[0].get_header("Authorization") == "Bearer new"
    assert runtime.has_platform_connection()
    for key in ("ANOMX_TOKEN", "ANOMX_PLATFORM_TOKEN", "ANOMX_PLATFORM_API_KEY", "ANOMX_API_KEY"):
        assert runtime.tool_manager.subprocess_env[key] == "new"


@pytest.mark.parametrize("header", ["Authorization", "authorization", "Host"])
def test_extra_headers_cannot_override_managed_identity(tmp_path, monkeypatch, header):
    monkeypatch.setattr(
        "anomx.agent.helpers.anomx_api.urlopen",
        lambda *args, **kwargs: pytest.fail("No request should be sent"),
    )
    with pytest.raises(AnomxApiError, match="manages Authorization"):
        call_anomx_api(
            AnomxApiConnection("https://example.test/api", "secret", tmp_path),
            method="GET", path="/events", headers={header: "untrusted"},
        )


def test_started_and_restored_children_keep_host_managed_authentication(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    home.set_platform_connection(url="https://example.test/api", token="managed")
    session = home.create_session(tmp_path, provider="openai", model="example")
    refreshes = []

    def refresh(force):
        refreshes.append(force)

    def respond(child, session_path, callbacks=None, **kwargs):
        connection = connection_from_home(child.home, token_refresh=child.platform_token_refresh)
        assert call_anomx_api(connection, method="GET", path="/events")["ok"]
        return "Checked"

    monkeypatch.setattr(AgentRuntime, "backend_response", respond)
    monkeypatch.setattr(
        "anomx.agent.helpers.anomx_api.urlopen", lambda *args, **kwargs: Response(b"{}"),
    )
    parent = AgentRuntime(home, tmp_path, subagents_enabled=True, platform_token_refresh=refresh)
    parent.configure_session_features(session.path, subagents_enabled=True)
    started = json.loads(parent._execute_tool(
        "start_subagent", {"name": "Reader", "prompt": "Read events"},
        RuntimeCallbacks(), session.path,
    ))
    child = parent._subagents[started["agent_id"]]
    child.worker.join(2)
    assert child.status == "ready", child.error

    resumed = AgentRuntime(home, tmp_path, platform_token_refresh=refresh)
    resumed._prepare_session_features(session.path)
    prompted = json.loads(resumed._execute_tool(
        "prompt_subagent", {"agent_id": child.agent_id, "prompt": "Read again"},
        RuntimeCallbacks(), session.path,
    ))
    assert prompted["prompted"]
    restored = resumed._subagents[child.agent_id]
    restored.worker.join(2)
    assert restored.status == "ready", restored.error
    assert refreshes == [False, False]
