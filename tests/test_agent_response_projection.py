import io
import json
from urllib.error import HTTPError

import pytest

from anomx.agent.helpers.anomx_api import AnomxApiConnection, AnomxApiError, call_anomx_api
from anomx.agent.helpers.response_projection import project_response_fields, response_field_paths


class Response:
    status = 200
    headers = {"content-type": "application/json"}

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_job_overview_needs_one_request_and_preserves_full_evidence(tmp_path, monkeypatch):
    jobs = [
        {
            "name": "Acquisition", "status": "problematic", "is_active": True,
            "status_message": "Health observation is stale.",
            "_anomx": {"object_reference": "jobs_job-example", "can_write": False},
            "active_run": {"status": "running", "config_snapshot": {"large": "x" * 100_000}},
            "channels": [{"samples": [1, 2, 3]}] * 100,
        },
        {"name": "Paused", "status": "paused", "is_active": False, "active_run": None},
    ]
    requests = []

    def request(req, **kwargs):
        requests.append(req)
        return Response(jobs)

    monkeypatch.setattr("anomx.agent.helpers.anomx_api.urlopen", request)
    result = call_anomx_api(
        AnomxApiConnection("https://example.test/api", "test", tmp_path),
        method="GET", path="/jobs", query={"limit": 20},
        response_fields=["name", "status", "status_message", "is_active",
                         "_anomx.object_reference", "active_run.status", "missing_field"],
    )
    assert len(requests) == 1
    assert requests[0].full_url == "https://example.test/api/jobs?limit=20"
    assert result["ok"] and not result["response_truncated"]
    assert result["response"][0] == {
        "name": "Acquisition", "status": "problematic", "is_active": True,
        "status_message": "Health observation is stale.",
        "_anomx": {"object_reference": "jobs_job-example"},
        "active_run": {"status": "running"},
    }
    assert result["response"][1]["is_active"] is False
    assert result["response"][1]["active_run"] is None
    assert result["missing_response_fields"] == ["missing_field"]
    assert len(json.dumps(result)) < 1600
    assert json.loads(next(tmp_path.glob("*.json")).read_text()) == jobs


def test_wrapped_pages_keep_pagination_nested_arrays_and_zero_values():
    result = project_response_fields({
        "count": 2, "next": "page-2", "results": [
            {"name": "A", "runs": [{"status": "failed", "value": 0, "large": "omit"}]},
            {"name": "B", "runs": []},
        ],
    }, ("count", "next", "results.name", "results.runs.status", "results.runs.value"))
    assert result["response"] == {
        "count": 2, "next": "page-2", "results": [
            {"name": "A", "runs": [{"status": "failed", "value": 0}]},
            {"name": "B", "runs": []},
        ],
    }
    assert result["missing_response_fields"] == []


@pytest.mark.parametrize("fields", [("run", "run.status"), ("run.status", "run")])
def test_explicit_parent_selection_preserves_its_children(fields):
    payload = {"run": {"status": "failed", "error": "Keep this"}, "unselected": 1}
    assert project_response_fields(payload, fields)["response"] == {"run": payload["run"]}


@pytest.mark.parametrize("value", [[], "name", [None], [""], ["a..b"], ["runs[0]"], ["*"]])
def test_invalid_fields_fail_before_request(tmp_path, monkeypatch, value):
    monkeypatch.setattr("anomx.agent.helpers.anomx_api.urlopen", lambda *_a, **_kw: pytest.fail())
    with pytest.raises(AnomxApiError):
        call_anomx_api(AnomxApiConnection("https://example.test/api", "test", tmp_path),
                       method="GET", path="/jobs", response_fields=value)
    assert not list(tmp_path.iterdir())


def test_api_errors_are_never_projected_away(tmp_path, monkeypatch):
    payload = {"detail": "Permission denied", "code": "permission_denied"}

    def reject(req, **kwargs):
        raise HTTPError(req.full_url, 403, "Forbidden", {"content-type": "application/json"},
                        io.BytesIO(json.dumps(payload).encode()))

    monkeypatch.setattr("anomx.agent.helpers.anomx_api.urlopen", reject)
    result = call_anomx_api(AnomxApiConnection("https://example.test/api", "test", tmp_path),
                            method="GET", path="/jobs", response_fields=["name"])
    assert not result["ok"]
    assert result["response"] == payload
    assert "response_fields" not in result


def test_field_selection_cannot_accidentally_mutate(tmp_path, monkeypatch):
    monkeypatch.setattr("anomx.agent.helpers.anomx_api.urlopen", lambda *_a, **_kw: pytest.fail())
    with pytest.raises(AnomxApiError, match="GET"):
        call_anomx_api(AnomxApiConnection("https://example.test/api", "test", tmp_path),
                       method="POST", path="/jobs", body={"name": "X"}, response_fields=["name"])
    assert response_field_paths(None) == ()


def test_application_failure_in_http_success_is_not_hidden(tmp_path, monkeypatch):
    payload = {"ok": False, "error": "No worker is available."}
    monkeypatch.setattr(
        "anomx.agent.helpers.anomx_api.urlopen", lambda *_a, **_kw: Response(payload)
    )
    result = call_anomx_api(AnomxApiConnection("https://example.test/api", "test", tmp_path),
                            method="GET", path="/jobs", response_fields=["name"])
    assert result["response"] == payload
    assert "response_fields" not in result
