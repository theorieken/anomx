"""Authenticated JSON transport without dataset, run, or workspace operations."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from email.message import Message
from http.client import HTTPException
from typing import IO, Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

JSONValue = dict[str, Any] | list[Any] | str | int | float | bool | None
QueryValue = str | int | float | bool | None
QueryParams = Mapping[str, QueryValue | list[QueryValue] | tuple[QueryValue, ...]]


class PlatformError(RuntimeError):
    """A failed API request, with HTTP status and request location when available."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        method: str | None = None,
        url: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.method = method
        self.url = url


class PlatformTimeoutError(PlatformError):
    """The request exceeded the configured socket timeout."""


class _NoRedirect(HTTPRedirectHandler):
    """Do not forward bearer credentials or silently change a request's method."""

    def redirect_request(
        self, req: Request, fp: IO[bytes], code: int, msg: str, headers: Message, newurl: str
    ) -> None:
        return None


@dataclass(frozen=True)
class PlatformClient:
    """An authenticated API connection, independent of any resource class.

    A site origin such as ``https://host`` becomes ``https://host/api``.
    Explicit API prefixes (including ``/api/v1`` and reverse-proxy paths) are
    preserved. Paths passed to HTTP methods are relative to that API prefix,
    starting with ``/``. Responses are decoded JSON; empty responses are ``None``.

    Credentials come only from arguments or ``from_env()``. Requests have a
    bounded timeout, are never retried, and do not follow redirects.
    """

    url: str
    token: str = field(repr=False)
    timeout: float = 30.0
    health_path: str = "/system/health"

    def __post_init__(self) -> None:
        parsed = urlsplit(self.url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or any(character.isspace() for character in self.url)
        ):
            raise ValueError(
                "Anomx URL must be an absolute HTTP(S) URL without credentials or query."
            )
        if not self.token or "\r" in self.token or "\n" in self.token:
            raise ValueError("An Anomx API token is required and must not contain line breaks.")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("A positive, finite request timeout is required.")
        base = self.url.rstrip("/")
        object.__setattr__(self, "url", base if parsed.path.strip("/") else base + "/api")
        self._request_url(self.health_path)

    @classmethod
    def from_env(cls) -> PlatformClient:
        """Read ``ANOMX_URL`` and ``ANOMX_TOKEN`` for a standalone connection."""
        url = os.environ.get("ANOMX_URL", "")
        token = os.environ.get("ANOMX_TOKEN", "")
        if not url or not token:
            raise PlatformError(
                "Set ANOMX_URL and ANOMX_TOKEN, or provide PlatformClient explicitly."
            )
        return cls(url=url, token=token)

    def _request_url(self, path: str, params: QueryParams | None = None) -> str:
        parsed = urlsplit(path)
        decoded = unquote(parsed.path)
        if (
            not path.startswith("/")
            or path.startswith("//")
            or parsed.scheme
            or parsed.netloc
            or parsed.fragment
            or "\\" in decoded
            or any(part in {".", ".."} for part in decoded.split("/"))
            or any(ord(character) < 32 or ord(character) == 127 for character in path)
        ):
            raise ValueError("API paths must stay within the configured API prefix.")
        query = urlencode(
            {key: value for key, value in (params or {}).items() if value is not None}, doseq=True
        )
        return self.url + path + (("&" if parsed.query else "?") + query if query else "")

    def request(
        self,
        path: str,
        payload: JSONValue = None,
        *,
        method: str = "POST",
        params: QueryParams | None = None,
    ) -> JSONValue:
        """Send a JSON request and decode JSON of any shape, or an empty response.

        ``params`` URL-encodes query values and repeats keys for list values.
        HTTP errors expose ``status_code``, ``method``, and ``url``. Error messages
        omit query values, credentials, and response bodies.
        """
        method = method.upper()
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
            raise ValueError(f"Unsupported HTTP method: {method}.")
        url = self._request_url(path, params)
        location = url.split("?", 1)[0]
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}
        body = json.dumps(payload, allow_nan=False).encode("utf-8") if payload is not None else None
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = Request(url, data=body, method=method, headers=headers)
        try:
            with build_opener(_NoRedirect()).open(request, timeout=self.timeout) as response:
                content = response.read()
                if method == "HEAD" or not content.strip():
                    return None
                return json.loads(content)  # type: ignore[no-any-return]
        except HTTPError as exc:
            exc.close()
            raise PlatformError(
                f"Anomx API returned HTTP {exc.code} for {method} {location}.",
                status_code=exc.code,
                method=method,
                url=url,
            ) from None
        except (URLError, OSError, HTTPException) as exc:
            if isinstance(exc, TimeoutError) or isinstance(
                getattr(exc, "reason", None), TimeoutError
            ):
                raise PlatformTimeoutError(
                    f"Request timed out for {method} {location} (timeout={self.timeout:g}s).",
                    method=method,
                    url=url,
                ) from None
            raise PlatformError(
                f"Could not complete {method} {location} ({type(exc).__name__}).",
                method=method,
                url=url,
            ) from None
        except (ValueError, UnicodeError):
            raise PlatformError(
                f"Anomx API did not return valid JSON for {method} {location}.",
                method=method,
                url=url,
            ) from None

    def get(self, path: str, *, params: QueryParams | None = None) -> JSONValue:
        """Read a resource, optionally passing query parameters."""
        return self.request(path, method="GET", params=params)

    def post(
        self, path: str, payload: JSONValue = None, *, params: QueryParams | None = None
    ) -> JSONValue:
        """Submit JSON to a resource."""
        return self.request(path, payload, method="POST", params=params)

    def put(
        self, path: str, payload: JSONValue = None, *, params: QueryParams | None = None
    ) -> JSONValue:
        """Replace a resource with JSON."""
        return self.request(path, payload, method="PUT", params=params)

    def patch(
        self, path: str, payload: JSONValue = None, *, params: QueryParams | None = None
    ) -> JSONValue:
        """Partially update a resource with JSON."""
        return self.request(path, payload, method="PATCH", params=params)

    def delete(self, path: str, *, params: QueryParams | None = None) -> JSONValue:
        """Delete a resource; an empty success response returns ``None``."""
        return self.request(path, method="DELETE", params=params)

    def head(self, path: str, *, params: QueryParams | None = None) -> None:
        """Check a resource's HTTP status without decoding a body."""
        self.request(path, method="HEAD", params=params)

    def is_alive(self) -> bool:
        """Check the configured health resource for ``status: ok``.

        Return ``False`` for HTTP, connection, timeout, or invalid-response errors.
        This checks API liveness, not access to a particular dataset or resource.
        """
        try:
            result = self.get(self.health_path)
        except PlatformError:
            return False
        return isinstance(result, dict) and result.get("status") == "ok"
