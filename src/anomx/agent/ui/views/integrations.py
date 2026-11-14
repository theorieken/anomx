"""MCP-only integration management and prefetched mention suggestions."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import suppress
from functools import partial
from typing import Any, TypeVar

from anomx.agent.helpers.mcp_integrations import (
    integration_catalog,
    local_integrations,
    platform_mcp_request,
    remove_local_integration,
    save_local_integration,
)
from anomx.agent.mcp import McpError, mcp_request, validate_mcp_config
from anomx.agent.ui.models import CursesWindow, MenuChoice

T = TypeVar("T")


class McpIntegrationsViewMixin:
    """Keep network discovery outside the terminal input/render loop."""

    def _prefetch_mcp_integrations(self, *, force: bool = False) -> None:
        connection = self.home.platform_connection() or {}
        scope = (str(self.home.root), connection.get("url"), connection.get("token"))
        if scope != getattr(self, "_mcp_catalog_scope", None):
            self._mcp_catalog_scope = scope
            self._mcp_catalog = []
            self._mcp_catalog_at = 0.0
        if getattr(self, "_mcp_catalog_loading", False) or (
            not force and time.monotonic() - getattr(self, "_mcp_catalog_at", 0) < 30
        ):
            return
        self._mcp_catalog_loading = True

        def load() -> None:
            try:
                entries = integration_catalog(self.home)
                if self._mcp_catalog_scope == scope:
                    self._mcp_catalog = entries
                    self._mcp_catalog_error = ""
            except McpError as error:
                if self._mcp_catalog_scope == scope:
                    self._mcp_catalog_error = str(error)
                    with suppress(McpError):
                        self._mcp_catalog = [
                            {"id": item["id"], "name": item["name"], "source": "local"}
                            for item in local_integrations(self.home)
                        ]
            finally:
                self._mcp_catalog_at = time.monotonic()
                self._mcp_catalog_loading = False

        threading.Thread(target=load, daemon=True, name="mcp-mentions").start()

    def _mcp_reference_choices(self, query: str) -> list[MenuChoice]:
        self._prefetch_mcp_integrations()
        matches = [
            item
            for item in getattr(self, "_mcp_catalog", [])
            if query.casefold() in item["name"].casefold()
        ]
        rows = [MenuChoice("MCP Integrations", "", selectable=False)]
        rows.extend(
            MenuChoice(
                item["name"],
                f"mcp://{item['source']}/{item['id']}",
                "Platform" if item["source"] == "platform" else "Local",
                highlight=query,
            )
            for item in matches
        )
        if not matches:
            detail = (
                "Loading…"
                if getattr(self, "_mcp_catalog_loading", False)
                else "Unavailable"
                if getattr(self, "_mcp_catalog_error", "")
                else "Add one in /config → Manage Integrations"
            )
            rows.append(MenuChoice(detail, "", selectable=False))
        return rows

    def _reference_selected_index(
        self, choices: list[MenuChoice], selected: int, direction: int = 0
    ) -> int:
        available = [index for index, choice in enumerate(choices) if choice.selectable]
        if not available:
            return 0
        if direction > 0:
            return next((index for index in available if index > selected), available[-1])
        if direction < 0:
            return next((index for index in reversed(available) if index < selected), available[0])
        return (
            selected
            if selected in available
            else next((index for index in available if index >= selected), available[-1])
        )

    def _mcp_operation(self, stdscr: CursesWindow, title: str, operation: Callable[[], T]) -> T:
        result: Future[T] = Future()

        def run() -> None:
            try:
                result.set_result(operation())
            except Exception as error:
                result.set_exception(
                    error
                    if isinstance(error, McpError)
                    else McpError("The integration operation failed.")
                )

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        while thread.is_alive():
            self._draw_overlay(stdscr, title=title, subtitle="Connecting…", footer="")
            thread.join(0.05)
        return result.result()

    def _run_manage_integrations_panel(self, stdscr: CursesWindow) -> None:
        while True:
            try:
                entries = self._mcp_operation(
                    stdscr, "Manage Integrations", lambda: integration_catalog(self.home)
                )
            except McpError as error:
                self._message(stdscr, "Integrations", str(error))
                entries = [
                    {"id": item["id"], "name": item["name"], "source": "local"}
                    for item in local_integrations(self.home)
                ]
            choices = [
                MenuChoice("Add MCP Integration", "add", "Connect a server and mention it with @")
            ]
            choices.extend(
                MenuChoice(
                    item["name"],
                    str(index),
                    "Platform" if item["source"] == "platform" else "Local",
                )
                for index, item in enumerate(entries)
            )
            selected = self._menu(stdscr, "Manage Integrations", "MCP connections", tuple(choices))
            if selected is None:
                self._prefetch_mcp_integrations(force=True)
                return
            try:
                if selected == "add":
                    self._edit_mcp_integration(stdscr)
                else:
                    item = entries[int(selected)]
                    action = self._menu(
                        stdscr,
                        item["name"],
                        "MCP integration",
                        (
                            MenuChoice("Check Connection", "check"),
                            MenuChoice("Edit", "edit"),
                            MenuChoice("Remove", "remove"),
                        ),
                    )
                    if action == "check":
                        if item["source"] == "local":
                            config = next(
                                entry["config"]
                                for entry in local_integrations(self.home)
                                if entry["id"] == item["id"]
                            )
                            tools = self._mcp_operation(
                                stdscr, item["name"], partial(mcp_request, config)
                            )
                        else:
                            tools = self._mcp_operation(
                                stdscr,
                                item["name"],
                                partial(
                                    platform_mcp_request,
                                    self.home,
                                    f"/integrations/{item['id']}/mcp-tools",
                                ),
                            )["tools"]
                        self._message(
                            stdscr,
                            item["name"],
                            f"Connected · {len(tools)} tools available. "
                            "Mention this integration with @ to use them.",
                        )
                    elif action == "edit":
                        self._edit_mcp_integration(stdscr, item)
                    elif (
                        action == "remove"
                        and self._menu(
                            stdscr,
                            "Remove Integration",
                            item["name"],
                            (MenuChoice("Cancel", "cancel"), MenuChoice("Remove", "remove")),
                        )
                        == "remove"
                    ):
                        if item["source"] == "local":
                            remove_local_integration(self.home, item["id"])
                        else:
                            self._mcp_operation(
                                stdscr,
                                "Removing Integration",
                                partial(
                                    platform_mcp_request,
                                    self.home,
                                    f"/integrations/{item['id']}",
                                    method="DELETE",
                                ),
                            )
                self._prefetch_mcp_integrations(force=True)
            except (McpError, ValueError) as error:
                self._message(stdscr, "Integration", str(error))

    def _edit_mcp_integration(
        self, stdscr: CursesWindow, item: dict[str, Any] | None = None
    ) -> None:
        source = item["source"] if item else "local"
        if item is None and self.home.platform_connection():
            source = self._menu(
                stdscr,
                "Add MCP Integration",
                "Where should this connection be available?",
                (
                    MenuChoice(
                        "Anomx Platform", "platform", "Available in the platform and connected CLI"
                    ),
                    MenuChoice("This Computer", "local", "Available only in this CLI"),
                ),
            )
            if source is None:
                return
        existing = {}
        if item:
            existing = (
                next(entry for entry in local_integrations(self.home) if entry["id"] == item["id"])
                if source == "local"
                else self._mcp_operation(
                    stdscr,
                    "Loading Integration",
                    lambda: platform_mcp_request(self.home, f"/integrations/{item['id']}"),
                )
            )
        config = dict(existing.get("config") or {})
        name = self._run_overlay_text(
            stdscr,
            "Integration Name",
            "Choose a name to find with @",
            optional=False,
            default=item["name"] if item else "",
        )
        if name is None:
            return
        transports = [
            MenuChoice("Streamable HTTP", "streamable_http"),
            MenuChoice("SSE", "sse", "Legacy remote servers"),
        ]
        if source == "local":
            transports.append(
                MenuChoice("Local Command", "stdio", "Run an MCP executable on this computer")
            )
        transport = self._menu(
            stdscr, "Transport", "Choose the server connection", tuple(transports)
        )
        if transport is None:
            return
        config["transport"] = transport
        fields = (
            [
                ("command", "Executable", "For example: uvx or npx", False, False),
                (
                    "args",
                    "Arguments",
                    'JSON array, e.g. ["-y", "@modelcontextprotocol/server-filesystem", "/path"]',
                    False,
                    True,
                ),
                ("env", "Environment", "Optional JSON object of environment variables", True, True),
            ]
            if transport == "stdio"
            else [
                ("url", "Server URL", "For example: https://example.com/mcp", False, False),
                (
                    "token",
                    "Bearer Token",
                    "Optional; leave empty for a public server. Clear to remove a saved token.",
                    True,
                    True,
                ),
                (
                    "headers",
                    "Custom Headers",
                    "Optional JSON object of headers. Clear to remove saved headers.",
                    True,
                    True,
                ),
            ]
        )
        for key, label, hint, masked, optional in fields:
            value = config.get(key, "")
            default = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            value = self._run_overlay_text(
                stdscr, label, hint, mask=masked, optional=optional, default=default
            )
            if value is None:
                return
            config[key] = (
                json.loads(value or ("[]" if key == "args" else "{}"))
                if key in {"args", "env"}
                else value
            )
        if source == "local":
            validate_mcp_config(config, allow_stdio=True)
            save_local_integration(self.home, name, config, identifier=item["id"] if item else "")
        else:
            path = f"/integrations/{item['id']}" if item else "/integrations/mcp-catalog"
            self._mcp_operation(
                stdscr,
                "Saving Integration",
                lambda: platform_mcp_request(
                    self.home,
                    path,
                    method="PATCH" if item else "POST",
                    payload={"name": name, "config": config},
                ),
            )
