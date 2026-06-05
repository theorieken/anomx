"""HTTP helpers for connecting the CLI agent to an Anomx Platform."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from anomx import __version__
from anomx.agent.store import AnomxHome

DEFAULT_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class PlatformLoginResult:
    """Result returned after a successful CLI agent platform login."""

    url: str
    token: str
    user_email: str
    organization_url: str
    hostname: str


class PlatformClientError(RuntimeError):
    """Raised when the CLI cannot talk to the configured platform."""


def normalize_platform_url(value: str) -> str:
    """Return a normalized platform API origin from user input."""

    candidate = value.strip().rstrip("/")
    if not candidate:
        raise PlatformClientError("Platform URL is required.")
    if "://" not in candidate:
        scheme = "http" if _looks_local(candidate) else "https"
        candidate = f"{scheme}://{candidate}"

    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PlatformClientError("Platform URL must include a valid http or https origin.")

    return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))


def platform_domain(url: str) -> str:
    """Return a compact domain label for UI helper text."""

    parsed = urlparse(url)
    return parsed.netloc or url


def local_hostname() -> str:
    """Return the hostname reported to the platform for this CLI agent."""

    return socket.gethostname() or "unknown-host"


def connect_platform(url: str, email: str, password: str) -> PlatformLoginResult:
    """Log into the platform as a CLI agent and return the issued token."""

    normalized_url = normalize_platform_url(url)
    hostname = local_hostname()
    payload = _request_json(
        normalized_url,
        "/auth/login",
        {
            "email": email,
            "password": password,
            "client": "cli_agent",
            "client_hostname": hostname,
            "client_version": __version__,
        },
    )
    token = str(payload.get("token") or "").strip()
    if not token:
        raise PlatformClientError("Platform login did not return a bearer token.")

    user = payload.get("user") if isinstance(payload.get("user"), dict) else {}
    organization = user.get("organization") if isinstance(user.get("organization"), dict) else {}
    return PlatformLoginResult(
        url=normalized_url,
        token=token,
        user_email=str(user.get("email") or email),
        organization_url=str(organization.get("url") or ""),
        hostname=hostname,
    )


def heartbeat_platform_connection(home: AnomxHome) -> bool:
    """Refresh the current CLI agent token's last-used timestamp if configured."""

    connection = home.platform_connection()
    if connection is None:
        return False
    try:
        _request_json(
            connection["url"],
            "/auth/me/agent/heartbeat",
            {
                "client_hostname": local_hostname(),
                "client_version": __version__,
            },
            token=connection["token"],
        )
    except PlatformClientError:
        return False
    return True


def _request_json(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    token: str | None = None,
) -> dict[str, Any]:
    normalized_path = path if path.startswith("/") else f"/{path}"
    url = f"{base_url}{normalized_path}"
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": f"anomx-cli/{__version__}",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            response_body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raise PlatformClientError(_format_http_error(exc)) from exc
    except URLError as exc:
        raise PlatformClientError(f"Could not reach the platform: {exc.reason}") from exc
    except TimeoutError as exc:
        raise PlatformClientError("Timed out while connecting to the platform.") from exc

    if not response_body.strip():
        return {}
    try:
        data = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise PlatformClientError("Platform returned a non-JSON response.") from exc
    if not isinstance(data, dict):
        raise PlatformClientError("Platform returned an unexpected response.")
    return data


def _format_http_error(exc: HTTPError) -> str:
    raw_body = exc.read().decode("utf-8", errors="replace").strip()
    if raw_body:
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            detail = payload.get("detail")
            if detail:
                return str(detail)
    return f"Platform request failed with HTTP {exc.code}."


def _looks_local(value: str) -> bool:
    host = value.split("/", 1)[0].split(":", 1)[0].lower()
    return host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".local")
