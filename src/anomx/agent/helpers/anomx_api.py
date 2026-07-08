"""Anomx Platform API helpers for agent tools and skill scripts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen
from uuid import uuid4

from anomx import __version__
from anomx.agent.store import AnomxHome

DEFAULT_API_TIMEOUT_SECONDS = 30
ANOMX_PLATFORM_ENV_KEYS = (
    "ANOMX_PLATFORM_URL",
    "ANOMX_PLATFORM_API_URL",
    "ANOMX_PLATFORM_TOKEN",
    "ANOMX_PLATFORM_API_KEY",
    "ANOMX_API_KEY",
    "ANOMX_RESPONSES_DIR",
)
ROOT_ONLY_PATHS = frozenset({"/docs", "/openapi.json"})


@dataclass(frozen=True)
class AnomxApiConnection:
    """Resolved platform API connection for the current agent runtime."""

    base_url: str
    token: str
    responses_dir: Path


class AnomxApiError(RuntimeError):
    """Raised when an Anomx Platform API call cannot be attempted."""


def platform_environment(home: AnomxHome) -> dict[str, str]:
    """Return environment variables that expose the connected platform API."""

    connection = home.platform_connection()
    if connection is None:
        return {}

    token = connection["token"]
    api_url = platform_api_base_url(connection["url"])
    return {
        "ANOMX_PLATFORM_URL": connection["url"],
        "ANOMX_PLATFORM_API_URL": api_url,
        "ANOMX_PLATFORM_TOKEN": token,
        "ANOMX_PLATFORM_API_KEY": token,
        "ANOMX_API_KEY": token,
        "ANOMX_RESPONSES_DIR": str(home.responses_dir),
    }


def connection_from_home(home: AnomxHome) -> AnomxApiConnection | None:
    """Return the current home platform connection, if configured."""

    connection = home.platform_connection()
    if connection is None:
        return None
    return AnomxApiConnection(
        base_url=platform_api_base_url(connection["url"]),
        token=connection["token"],
        responses_dir=home.responses_dir,
    )


def platform_api_base_url(value: str) -> str:
    """Return the canonical REST API base URL for a configured platform URL."""

    parsed = urlparse(value.strip().rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/api/v1"):
        normalized_path = path
    elif path.endswith("/api"):
        normalized_path = f"{path}/v1"
    else:
        normalized_path = f"{path}/api/v1" if path else "/api/v1"
    return urlunparse((parsed.scheme, parsed.netloc, normalized_path, "", "", ""))


def call_anomx_api(
    connection: AnomxApiConnection,
    *,
    method: str,
    path: str,
    query: Mapping[str, object] | None = None,
    body: object | None = None,
    headers: Mapping[str, str] | None = None,
    output_name: str | None = None,
    timeout: float = DEFAULT_API_TIMEOUT_SECONDS,
) -> dict[str, object]:
    """Call the Anomx Platform API and write the response payload to JSON."""

    normalized_method = method.strip().upper()
    if normalized_method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
        raise AnomxApiError("method must be one of GET, POST, PUT, PATCH, DELETE.")

    url = _build_url(connection.base_url, path, query)
    request_body = None
    request_headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {connection.token}",
        "User-Agent": f"anomx-agent/{__version__}",
    }
    if headers:
        request_headers.update({str(key): str(value) for key, value in headers.items()})
    if body is not None and normalized_method != "GET":
        request_body = json.dumps(body).encode("utf-8")
        request_headers["Content-Type"] = "application/json"

    request = Request(url, data=request_body, headers=request_headers, method=normalized_method)
    status_code = 0
    content_type = ""
    raw = b""
    error = ""
    try:
        with urlopen(request, timeout=timeout) as response:
            status_code = int(getattr(response, "status", 200))
            content_type = str(response.headers.get("content-type", ""))
            raw = response.read()
    except HTTPError as exc:
        status_code = int(exc.code)
        content_type = str(exc.headers.get("content-type", "")) if exc.headers else ""
        raw = exc.read()
        error = str(exc)
    except (OSError, TimeoutError, URLError) as exc:
        error = str(exc)

    payload, parsed_as_json = _decode_payload(raw, content_type)
    output_path = _write_response(
        connection.responses_dir,
        output_name=output_name,
        payload=payload,
    )
    length = len(raw)
    result_count = _result_count(payload)
    meta = {
        "ok": 200 <= status_code < 300,
        "status_code": status_code,
        "method": normalized_method,
        "url": url,
        "content_type": content_type,
        "length_bytes": length,
        "parsed_as_json": parsed_as_json,
        "result_count": result_count,
        "response_path": str(output_path),
    }
    if error:
        meta["error"] = error
    return meta


def _build_url(base_url: str, path: str, query: Mapping[str, object] | None) -> str:
    normalized_path = path.strip()
    if not normalized_path:
        raise AnomxApiError("path is required.")
    root_path = normalized_path if normalized_path.startswith("/") else f"/{normalized_path}"
    if normalized_path.startswith(("http://", "https://")):
        url = normalized_path
    elif root_path in ROOT_ONLY_PATHS:
        parsed = urlparse(base_url)
        origin = urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
        url = urljoin(f"{origin.rstrip('/')}/", root_path.lstrip("/"))
    else:
        url = urljoin(f"{base_url.rstrip('/')}/", normalized_path.lstrip("/"))
    query_string = _query_string(query)
    if query_string:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}{query_string}"
    return url


def _query_string(query: Mapping[str, object] | None) -> str:
    if not query:
        return ""
    items: list[tuple[str, str]] = []
    for key, value in query.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            items.extend((str(key), str(item)) for item in value if item is not None)
        else:
            items.append((str(key), str(value)))
    return urlencode(items, doseq=True)


def _decode_payload(raw: bytes, content_type: str) -> tuple[object, bool]:
    text = raw.decode("utf-8", errors="replace")
    if "json" in content_type.lower() or text.strip().startswith(("{", "[")):
        try:
            return json.loads(text), True
        except json.JSONDecodeError:
            pass
    return {"content": text}, False


def _write_response(
    responses_dir: Path,
    *,
    output_name: str | None,
    payload: object,
) -> Path:
    responses_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_output_name(output_name) or f"anomx-api-{uuid4().hex[:12]}"
    path = responses_dir / f"{safe_name}.json"
    counter = 2
    while path.exists():
        path = responses_dir / f"{safe_name}-{counter}.json"
        counter += 1
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def _safe_output_name(value: str | None) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in str(value or "").strip().lower()
    ).strip("-_")
    return cleaned[:80]


def _result_count(payload: object) -> int | None:
    if isinstance(payload, list):
        return len(payload)
    if not isinstance(payload, dict):
        return None
    for key in ("results", "items", "objects", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    count = payload.get("count")
    if isinstance(count, int):
        return count
    return None
