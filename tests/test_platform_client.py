"""Wire-level contracts for the resource-independent API transport."""

import io
import json
from http.client import IncompleteRead
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, build_opener

import pytest

from anomx.integrations.platform import PlatformClient, PlatformError, PlatformTimeoutError
from anomx.integrations.platform.client import _NoRedirect


@pytest.fixture
def send(monkeypatch):
    opener = Mock(return_value=io.BytesIO(b"{}"))
    monkeypatch.setattr(
        "anomx.integrations.platform.client.build_opener",
        lambda *handlers: SimpleNamespace(open=opener),
    )
    return opener


@pytest.mark.parametrize(
    "base,expected",
    [
        ("https://example.test", "https://example.test/api"),
        ("https://example.test/", "https://example.test/api"),
        ("https://example.test/api/", "https://example.test/api"),
        ("https://example.test/api/v1/", "https://example.test/api/v1"),
        ("http://localhost:8000/proxy/api", "http://localhost:8000/proxy/api"),
    ],
)
def test_base_url_is_normalized_once(base, expected, send):
    client = PlatformClient(base, "secret")
    client.get("/objects")
    assert client.url == expected
    assert send.call_args.args[0].full_url == expected + "/objects"


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/file",
        "//example.test",
        "https://user:password@example.test",
        "https://example.test?token=value",
        "https://example.test/#fragment",
        "https://example.test\n",
    ],
)
def test_invalid_base_urls_fail_before_any_request(url, send):
    with pytest.raises(ValueError, match="HTTP"):
        PlatformClient(url, "secret")
    send.assert_not_called()


@pytest.mark.parametrize(
    "path",
    [
        "objects",
        "//elsewhere.test/objects",
        "https://elsewhere.test/objects",
        "/../outside",
        "/objects/../outside",
        "/%2e%2e/outside",
        "/objects#fragment",
        "/objects\n",
        "/objects\\outside",
    ],
)
def test_request_cannot_leave_api_prefix(path, send):
    with pytest.raises(ValueError, match="API prefix"):
        PlatformClient("https://example.test", "secret").get(path)
    send.assert_not_called()


@pytest.mark.parametrize("verb", ["get", "post", "put", "patch", "delete", "head"])
def test_http_verbs_authentication_query_encoding_and_timeout(verb, send):
    client = PlatformClient("https://example.test", "secret", timeout=8)
    payload = {"nested": [1, "two"]} if verb in {"post", "put", "patch"} else None
    args = ["/objects", payload] if payload else ["/objects"]
    getattr(client, verb)(*args, params={"query": "a/b & c", "ids": [1, 2], "unused": None})
    request = send.call_args.args[0]
    assert request.get_method() == verb.upper()
    assert request.get_header("Authorization") == "Bearer secret"
    assert request.get_header("Accept") == "application/json"
    assert parse_qs(urlsplit(request.full_url).query) == {"query": ["a/b & c"], "ids": ["1", "2"]}
    assert send.call_args.kwargs["timeout"] == 8
    assert (json.loads(request.data) if request.data else None) == payload
    assert "secret" not in repr(client)


def test_request_retains_explicit_method_and_existing_query(send):
    client = PlatformClient("https://example.test", "secret")
    client.request("/objects?existing=yes", {"name": "changed"}, method="put", params={"page": 2})
    request = send.call_args.args[0]
    assert request.get_method() == "PUT"
    assert request.full_url.endswith("/objects?existing=yes&page=2")


@pytest.mark.parametrize(
    "body,expected", [(b"", None), (b"null", None), (b"[]", []), (b"5", 5), (b"true", True)]
)
def test_empty_and_json_responses(body, expected, send):
    send.return_value = io.BytesIO(body)
    assert PlatformClient("https://example.test", "secret").get("/objects") == expected


def test_http_error_exposes_diagnostic_fields_without_credentials_or_query(send):
    send.side_effect = HTTPError(
        "https://example.test/api/objects", 404, "Not found", {}, io.BytesIO(b"secret")
    )
    with pytest.raises(PlatformError) as error:
        PlatformClient("https://example.test", "secret").get(
            "/objects", params={"private": "query-secret"}
        )
    assert error.value.status_code == 404
    assert error.value.method == "GET"
    assert error.value.url == "https://example.test/api/objects?private=query-secret"
    assert "GET https://example.test/api/objects" in str(error.value)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "failure", [URLError("offline"), TimeoutError(), IncompleteRead(b"partial"), OSError("closed")]
)
def test_connection_failures_are_platform_errors(failure, send):
    send.side_effect = failure
    with pytest.raises(PlatformError) as error:
        PlatformClient("https://example.test", "secret").get("/objects")
    assert error.value.status_code is None


@pytest.mark.parametrize("failure", [TimeoutError(), URLError(TimeoutError())])
def test_timeout_reports_configured_limit_and_has_a_distinct_type(failure, send):
    send.side_effect = failure
    with pytest.raises(
        PlatformTimeoutError, match=r"POST https://example.test/api/objects \(timeout=12s\)"
    ) as error:
        PlatformClient("https://example.test", "secret", timeout=12).post("/objects", {})
    assert error.value.status_code is None
    assert error.value.method == "POST"
    assert error.value.url == "https://example.test/api/objects"


@pytest.mark.parametrize(
    "body,alive",
    [
        (b'{"status":"ok"}', True),
        (b'{"status":"error"}', False),
        (b"{}", False),
        (b"[]", False),
        (b"<html>login</html>", False),
        (b"", False),
    ],
)
def test_is_alive_validates_health_response(body, alive, send):
    send.return_value = io.BytesIO(body)
    assert PlatformClient("https://example.test", "secret").is_alive() is alive
    request = send.call_args.args[0]
    assert request.get_method() == "GET"
    assert request.full_url == "https://example.test/api/system/health"


@pytest.mark.parametrize(
    "failure",
    [URLError("offline"), TimeoutError(), HTTPError("url", 401, "Unauthorized", {}, None)],
)
def test_is_alive_returns_false_for_failed_requests(failure, send):
    send.side_effect = failure
    assert PlatformClient("https://example.test", "secret").is_alive() is False


def test_health_path_can_be_configured(send):
    send.return_value = io.BytesIO(b'{"status":"ok"}')
    assert PlatformClient("https://example.test", "secret", health_path="/healthz").is_alive()
    assert send.call_args.args[0].full_url.endswith("/healthz")


def test_redirect_is_an_error_and_does_not_forward_token():
    opener = build_opener(_NoRedirect())
    request = Request(
        "https://example.test/api/objects", headers={"Authorization": "Bearer secret"}
    )
    with pytest.raises(HTTPError) as error:
        opener.error(
            "http", request, io.BytesIO(b""), 302, "Moved", {"Location": "https://elsewhere.test/"}
        )
    assert error.value.code == 302


def test_client_has_no_resource_specific_methods():
    client = PlatformClient("https://example.test", "secret")
    for name in ("describe_dataset", "read_dataset", "load_channels", "channels_path"):
        assert not hasattr(client, name)


def test_non_json_is_a_clear_error(send):
    send.return_value = io.BytesIO(b"<html>Not an API response</html>")
    with pytest.raises(PlatformError, match="valid JSON.*GET https://example.test/api/objects"):
        PlatformClient("https://example.test", "secret").get("/objects")
