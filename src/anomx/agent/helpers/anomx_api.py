"""Anomx Platform API helpers for agent tools and skill scripts."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from email.message import Message
from http.client import HTTPException
from pathlib import Path
from typing import IO
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse, urlunparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

from anomx import __version__
from anomx.agent.helpers.response_projection import project_response_fields, response_field_paths
from anomx.agent.store import AnomxHome

DEFAULT_API_TIMEOUT_SECONDS = 30
RESPONSE_PREVIEW_MAX_CHARACTERS = 20_000
RESPONSE_PREVIEW_MAX_ITEMS = 20
ANOMX_PLATFORM_ENV_KEYS = (
    "ANOMX_URL",
    "ANOMX_TOKEN",
    "ANOMX_PLATFORM_URL",
    "ANOMX_PLATFORM_API_URL",
    "ANOMX_PLATFORM_TOKEN",
    "ANOMX_PLATFORM_API_KEY",
    "ANOMX_API_KEY",
    "ANOMX_RESPONSES_DIR",
)


@dataclass(frozen=True)
class AnomxApiConnection:
    """Resolved platform API connection for the current agent runtime."""

    base_url: str
    token: str = field(repr=False)
    responses_dir: Path
    token_provider: Callable[[bool], str] | None = field(default=None, repr=False, compare=False)


class AnomxApiError(RuntimeError):
    """Raised when an Anomx Platform API call cannot be attempted."""


class _NoRedirect(HTTPRedirectHandler):
    """Keep credentials and authentication retries on the requested endpoint."""

    def redirect_request(
        self, req: Request, fp: IO[bytes], code: int, msg: str, headers: Message, newurl: str,
    ) -> None:
        return None


urlopen = build_opener(_NoRedirect()).open


def platform_environment(home: AnomxHome) -> dict[str, str]:
    """Return environment variables that expose the connected platform API."""

    connection = home.platform_connection()
    if connection is None:
        return {}

    token = connection["token"]
    api_url = platform_api_base_url(connection["url"])
    return {
        "ANOMX_URL": api_url,
        "ANOMX_TOKEN": token,
        "ANOMX_PLATFORM_URL": connection["url"],
        "ANOMX_PLATFORM_API_URL": api_url,
        "ANOMX_PLATFORM_TOKEN": token,
        "ANOMX_PLATFORM_API_KEY": token,
        "ANOMX_API_KEY": token,
        "ANOMX_RESPONSES_DIR": str(home.responses_dir),
    }


def connection_from_home(
    home: AnomxHome, *, token_refresh: Callable[[bool], None] | None = None,
) -> AnomxApiConnection | None:
    """Return the current home platform connection, if configured."""

    connection = home.platform_connection()
    if connection is None:
        return None
    base_url = platform_api_base_url(connection["url"])

    def token_provider(force: bool) -> str:
        if token_refresh is not None:
            token_refresh(force)
        current = home.platform_connection()
        if current is None or platform_api_base_url(current["url"]) != base_url:
            raise AnomxApiError("The platform connection changed during this request.")
        return current["token"]

    return AnomxApiConnection(
        base_url=base_url,
        token=connection["token"],
        responses_dir=home.responses_dir,
        token_provider=token_provider if token_refresh is not None else None,
    )


def platform_api_base_url(value: str) -> str:
    """Return the canonical REST API base URL for a configured platform URL."""

    parsed = urlparse(value.strip().rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/api/v1"):
        normalized_path = path[: -len("/v1")]
    elif path.endswith("/api"):
        normalized_path = path
    else:
        normalized_path = f"{path}/api" if path else "/api"
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
    response_fields: object = None,
) -> dict[str, object]:
    """Call the API, renewing once only after a confirmed authentication rejection.

    Authentication runs before endpoint execution, so that rejection can safely
    be retried for writes too. Permission failures, transport errors and server
    failures are never replayed: a mutation may already have taken effect.
    """

    normalized_method = method.strip().upper()
    if (
        isinstance(timeout, bool) or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout) or not 1 <= timeout <= 300
    ):
        raise AnomxApiError("timeout_seconds must be a finite number between 1 and 300.")
    try:
        selected_fields = response_field_paths(response_fields)
    except ValueError as exc:
        raise AnomxApiError(str(exc)) from exc
    if selected_fields and normalized_method != "GET":
        raise AnomxApiError("response_fields is supported only for GET requests.")
    if normalized_method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
        raise AnomxApiError("method must be one of GET, POST, PUT, PATCH, DELETE.")

    url = _build_url(connection.base_url, path, query)
    request_body = None
    request_headers = {
        "Accept": "application/json",
        "User-Agent": f"anomx-agent/{__version__}",
    }
    if headers:
        if any(str(key).lower() in {"authorization", "host"} for key in headers):
            raise AnomxApiError(
                "The platform connection manages Authorization and Host headers."
            )
        request_headers.update({str(key): str(value) for key, value in headers.items()})
    if body is not None and normalized_method != "GET":
        request_body = json.dumps(body).encode("utf-8")
        request_headers["Content-Type"] = "application/json"

    for attempt in range(2):
        token = connection.token
        if connection.token_provider is not None:
            try:
                token = connection.token_provider(bool(attempt))
            except Exception:
                # Host callbacks may include database errors; never expose them
                # or their credential-bearing arguments to the model.
                raise AnomxApiError(
                    "The worker could not renew this run's API credential."
                ) from None
        request_headers["Authorization"] = f"Bearer {token}"
        request = Request(url, data=request_body, headers=request_headers, method=normalized_method)
        status_code = 0
        content_type = ""
        raw = b""
        error = ""
        timed_out = False
        response_scope = ""
        try:
            with urlopen(request, timeout=timeout) as response:
                status_code = int(getattr(response, "status", 200))
                content_type = str(response.headers.get("content-type", ""))
                response_scope = str(response.headers.get("X-Anomx-Data-Scope", ""))
                raw = response.read()
        except HTTPError as exc:
            status_code = int(exc.code)
            content_type = str(exc.headers.get("content-type", "")) if exc.headers else ""
            try:
                raw = exc.read()
            except (OSError, HTTPException):
                raw = b""
            finally:
                exc.close()
            error = str(exc)
        except (OSError, TimeoutError, URLError, HTTPException) as exc:
            timed_out = isinstance(exc, TimeoutError) or isinstance(
                getattr(exc, "reason", None), TimeoutError,
            )
            hint = (
                "Use a narrower query or increase timeout_seconds (maximum 300)."
                if normalized_method == "GET"
                else "The write may already have completed. Inspect its outcome before retrying."
            )
            error = (
                f"Request timed out after {timeout:g} seconds. {hint}" if timed_out else str(exc)
            )

        payload, parsed_as_json = _decode_payload(raw, content_type)
        if not (
            attempt == 0 and connection.token_provider is not None
            and _authentication_rejected(status_code, payload)
        ):
            break
    output_path = _write_response(
        connection.responses_dir,
        output_name=output_name,
        payload=payload,
    )
    length = len(raw)
    result_count = _result_count(payload)
    projection = (
        project_response_fields(payload, selected_fields)
        if selected_fields and parsed_as_json and 200 <= status_code < 300 and not error
        and not (
            isinstance(payload, dict) and (payload.get("ok") is False or payload.get("error"))
        )
        else {}
    )
    response_preview, response_truncated = _response_preview(projection.get("response", payload))
    if status_code == 404 and not parsed_as_json:
        response_preview = {
            "detail": (
                "No API route matched this URL. Use an unversioned path relative to "
                f"{connection.base_url}, such as /channels or /jobs. "
                "Read /openapi.json for supported paths. If documented paths also return "
                "404, report the API routing/configuration error instead of guessing "
                "version or module prefixes. The raw response is saved in response_path."
            )
        }
        response_truncated = True
    meta = {
        "ok": 200 <= status_code < 300 and not error,
        "status_code": status_code,
        "method": normalized_method,
        "url": url,
        "content_type": content_type,
        "length_bytes": length,
        "parsed_as_json": parsed_as_json,
        "result_count": result_count,
        "response_path": str(output_path),
        "response": response_preview,
        "response_truncated": response_truncated,
    }
    if error:
        meta["error"] = error
    if timed_out:
        meta.update(
            error_code="request_timeout", timeout_seconds=timeout,
            retryable=normalized_method == "GET",
        )
    if response_scope:
        meta["scope"] = response_scope
    if normalized_method == "GET" and result_count is not None:
        meta["collection_note"] = (
            "Only accessible records are returned. An empty list does not establish that no "
            "records exist for other accounts. result_count counts this response, not all pages."
        )
    if projection:
        meta.update({key: value for key, value in projection.items() if key != "response"})
    return meta


def _authentication_rejected(status_code: int, payload: object) -> bool:
    if status_code not in {401, 403} or not isinstance(payload, dict):
        return False
    return payload.get("code") == "invalid_bearer_token" or payload.get("detail") == (
        "Valid bearer token required."
    )


def _build_url(base_url: str, path: str, query: Mapping[str, object] | None) -> str:
    normalized_path = path.strip()
    if not normalized_path:
        raise AnomxApiError("path is required.")
    if normalized_path.startswith(("http://", "https://")):
        base = urlparse(base_url)
        requested = urlparse(normalized_path)
        if (requested.scheme, requested.netloc) != (base.scheme, base.netloc):
            raise AnomxApiError("Absolute API URLs must use the connected platform origin.")
        url = normalized_path
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


def _response_preview(payload: object) -> tuple[object, bool]:
    """Return a bounded structured preview suitable for a model tool result."""

    remaining = [RESPONSE_PREVIEW_MAX_CHARACTERS]
    truncated = [False]

    def visit(value: object, depth: int = 0) -> object:
        if remaining[0] <= 0:
            truncated[0] = True
            return "[truncated]"
        if depth >= 8:
            truncated[0] = True
            return "[maximum depth reached]"
        if isinstance(value, dict):
            preview: dict[str, object] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= 50 or remaining[0] <= 0:
                    truncated[0] = True
                    break
                normalized_key = str(key)
                remaining[0] -= len(normalized_key) + 4
                preview[normalized_key] = visit(item, depth + 1)
            return preview
        if isinstance(value, list):
            if len(value) > RESPONSE_PREVIEW_MAX_ITEMS:
                truncated[0] = True
            return [visit(item, depth + 1) for item in value[:RESPONSE_PREVIEW_MAX_ITEMS]]
        if isinstance(value, str):
            allowed = max(0, min(len(value), remaining[0], 4_000))
            excerpt = value[:allowed]
            remaining[0] -= allowed
            if allowed < len(value):
                truncated[0] = True
                return f"{excerpt}…"
            return excerpt
        serialized = json.dumps(value, ensure_ascii=False, default=str)
        if len(serialized) > remaining[0]:
            truncated[0] = True
            remaining[0] = 0
            return "[truncated]"
        remaining[0] -= len(serialized)
        return value

    return visit(payload), truncated[0]
