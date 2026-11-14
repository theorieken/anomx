"""CLI-owned MCP settings, platform bridge and mention catalog."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, NoReturn
from uuid import uuid4

from anomx.agent.helpers.anomx_api import platform_api_base_url
from anomx.agent.helpers.platform_client import PlatformClientError, _request_json
from anomx.agent.mcp import McpError, McpIntegration, mcp_request, validate_mcp_config
from anomx.agent.store import AnomxHome

MENTION_PATTERN = re.compile(r"mcp://(local|platform)/([a-zA-Z0-9-]+)")


def local_integrations(home: AnomxHome) -> list[dict[str, Any]]:
    """Read private local settings, separate from workspace and transcript files."""
    path = home.root / "integrations.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ValueError
        return data
    except (OSError, ValueError):
        raise McpError("Local integration settings could not be read.") from None


def save_local_integration(
    home: AnomxHome, name: str, config: dict[str, Any], *, identifier: str = ""
) -> str:
    """Save an MCP-only connection atomically with owner-only file permissions."""
    normalized = validate_mcp_config(config, allow_stdio=True)
    identifier = identifier or str(uuid4())
    entries = [item for item in local_integrations(home) if item["id"] != identifier]
    entries.append({"id": identifier, "name": name.strip(), "config": normalized})
    _write_local(home, entries)
    return identifier


def _write_local(home: AnomxHome, entries: list[dict[str, Any]]) -> None:
    home.ensure()
    fd, temporary = tempfile.mkstemp(prefix=".integrations-", dir=home.root)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(entries, stream, indent=4)
        os.replace(temporary, home.root / "integrations.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def remove_local_integration(home: AnomxHome, identifier: str) -> None:
    """Remove only the selected local MCP connection."""
    _write_local(home, [item for item in local_integrations(home) if item["id"] != identifier])


def platform_mcp_request(
    home: AnomxHome, path: str, *, method: str = "GET", payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Call the platform without writing connection credentials to response files."""
    connection = home.platform_connection()
    if connection is None:
        raise McpError("Connect an Anomx Platform first.")
    try:
        return _request_json(
            platform_api_base_url(connection["url"]),
            path,
            payload or {},
            method=method,
            token=connection["token"],
            timeout=75,
        )
    except PlatformClientError:
        raise McpError(
            "The platform MCP request failed. Check your connection and integration access."
        ) from None


def integration_catalog(home: AnomxHome) -> list[dict[str, Any]]:
    """Fetch names and identifiers only; no MCP server is contacted here."""
    entries = [
        {"id": item["id"], "name": item["name"], "source": "local"}
        for item in local_integrations(home)
    ]
    if home.platform_connection():
        remote = platform_mcp_request(home, "/integrations/mcp-catalog")
        entries.extend({**item, "source": "platform"} for item in remote.get("integrations", []))
    return entries


def _platform_discover(home: AnomxHome, identifier: str) -> list[dict[str, Any]]:
    return platform_mcp_request(home, f"/integrations/{identifier}/mcp-tools")["tools"]


def _platform_call(
    home: AnomxHome, identifier: str, tool: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    return platform_mcp_request(
        home,
        f"/integrations/{identifier}/mcp-call",
        method="POST",
        payload={"name": tool, "arguments": arguments},
    )


def _platform_unavailable(*_arguments: object) -> NoReturn:
    raise McpError("The platform connection is unavailable. Reconnect it to use this integration.")


def selected_cli_integrations(home: AnomxHome, session_path: Path | None) -> list[McpIntegration]:
    """Load selections from user messages, never from assistant or tool output."""
    if session_path is None:
        return []
    selected = set()
    for event in home.read_session_events(session_path):
        payload = event.get("payload") or {}
        if payload.get("type") == "user_message":
            selected.update(
                MENTION_PATTERN.findall(
                    str(payload.get("backend_message") or payload.get("message") or "")
                )
            )
    if not selected:
        return []
    result = []
    for item in local_integrations(home):
        if ("local", item["id"]) not in selected:
            continue
        config = item["config"]
        revision = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        result.append(
            McpIntegration(
                f"local/{item['id']}",
                item["name"],
                revision,
                partial(mcp_request, config),
                partial(mcp_request, config),
                config=config,
            )
        )
    if any(source == "platform" for source, _ in selected) and home.platform_connection():
        try:
            catalog = platform_mcp_request(home, "/integrations/mcp-catalog").get(
                "integrations", []
            )
        except McpError:
            # A disconnected platform must not disable selected local connections.
            result.extend(
                McpIntegration(
                    f"platform/{identifier}",
                    "Platform integration",
                    "unavailable",
                    _platform_unavailable,
                    _platform_unavailable,
                )
                for source, identifier in selected
                if source == "platform"
            )
            return result
        for item in catalog:
            if ("platform", item["id"]) in selected:
                result.append(
                    McpIntegration(
                        f"platform/{item['id']}",
                        item["name"],
                        item["revision"],
                        partial(_platform_discover, home, item["id"]),
                        partial(_platform_call, home, item["id"]),
                    )
                )
    return result
