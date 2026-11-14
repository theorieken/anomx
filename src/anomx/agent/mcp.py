"""Optional MCP transport and dynamically selected agent tools.

Install ``anomx[mcp]`` for direct HTTP, SSE and local stdio connections.
Hosts supply integration discovery and execution without platform dependencies.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import queue
import re
import threading
from collections.abc import AsyncIterator, Callable, Mapping
from concurrent.futures import Future
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from anomx.agent.base.tools import BaseTool, ToolExecutionContext

if TYPE_CHECKING:
    from mcp import ClientSession

McpResult = list[dict[str, Any]] | dict[str, Any]


class McpError(RuntimeError):
    """A safe, credential-free MCP connection error."""


def validate_mcp_config(config: Mapping[str, Any], *, allow_stdio: bool = False) -> dict[str, Any]:
    """Validate connection settings without opening a server connection."""
    result = dict(config)
    transport = str(result.get("transport") or "streamable_http")
    if transport not in (
        {"streamable_http", "sse", "stdio"} if allow_stdio else {"streamable_http", "sse"}
    ):
        raise McpError("Choose Streamable HTTP or SSE for a remote MCP server.")
    result["transport"] = transport
    if transport == "stdio":
        if not isinstance(result.get("command"), str) or not result["command"].strip():
            raise McpError("An executable is required for a local MCP server.")
        args = result.get("args", [])
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise McpError("Command arguments must be a JSON array of strings.")
        result["args"] = args
    else:
        url = str(result.get("url") or "").strip()
        try:
            parsed = urlsplit(url)
            valid = parsed.scheme in {"http", "https"} and parsed.hostname and parsed.port != 0
        except ValueError:
            valid = False
        if not valid or parsed.username or parsed.password or parsed.fragment or parsed.query:
            raise McpError(
                "Enter an HTTP(S) server URL without credentials, query or fragment. "
                "Use headers for authentication."
            )
        result["url"] = url
    for key in ("headers", "env"):
        values = result.get(key) or {}
        if isinstance(values, str):
            try:
                values = json.loads(values)
            except ValueError:
                raise McpError(f"{key.title()} must be a JSON object of string values.") from None
        if not isinstance(values, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in values.items()
        ):
            raise McpError(f"{key.title()} must be a JSON object of string values.")
        if key == "headers" and any(
            not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", k) or "\r" in v or "\n" in v
            for k, v in values.items()
        ):
            raise McpError("Headers contain an invalid name or value.")
        result[key] = values
    token = result.get("token") or ""
    if not isinstance(token, str) or "\r" in token or "\n" in token:
        raise McpError("The bearer token is invalid.")
    result["token"] = token
    return result


@asynccontextmanager
async def _mcp_session(config: dict[str, Any]) -> AsyncIterator[ClientSession]:
    try:
        import httpx
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.sse import sse_client
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamable_http_client
    except ImportError:
        raise McpError(
            'MCP support is missing. Install it with: pip install "anomx[mcp]"'
        ) from None
    async with AsyncExitStack() as stack:
        headers = dict(config["headers"])
        if config["token"]:
            headers["Authorization"] = f"Bearer {config['token']}"
        if config["transport"] == "stdio":
            import os

            errors = stack.enter_context(open(os.devnull, "w"))  # noqa: SIM115
            streams = await stack.enter_async_context(
                stdio_client(
                    StdioServerParameters(
                        command=config["command"], args=config["args"], env=config["env"]
                    ),
                    errlog=errors,
                )
            )
        elif config["transport"] == "sse":
            streams = await stack.enter_async_context(
                sse_client(config["url"], headers=headers, timeout=15, sse_read_timeout=60)
            )
        else:
            client = await stack.enter_async_context(
                httpx.AsyncClient(
                    headers=headers, timeout=httpx.Timeout(60, connect=15), follow_redirects=False
                )
            )
            streams = await stack.enter_async_context(
                streamable_http_client(config["url"], http_client=client)
            )
        session = await stack.enter_async_context(
            ClientSession(streams[0], streams[1], read_timeout_seconds=timedelta(seconds=60))
        )
        async with asyncio.timeout(60):
            await session.initialize()
        yield session


async def _session_request(
    session: ClientSession, tool: str | None, arguments: dict[str, Any]
) -> McpResult:
    async with asyncio.timeout(60):
        if tool is not None:
            return (await session.call_tool(tool, arguments=arguments)).model_dump(
                mode="json", by_alias=True, exclude_none=True
            )
        tools: list[dict[str, Any]] = []
        cursor = None
        seen = set()
        for _ in range(100):
            page = await session.list_tools(cursor=cursor)
            tools.extend(
                item.model_dump(mode="json", by_alias=True, exclude_none=True)
                for item in page.tools
            )
            cursor = page.nextCursor
            if not cursor:
                return tools
            if cursor in seen:
                break
            seen.add(cursor)
        raise McpError("The MCP server returned too many tool pages or a repeated cursor.")


async def _mcp_request(
    config: dict[str, Any], tool: str | None, arguments: dict[str, Any]
) -> McpResult:
    async with _mcp_session(config) as session:
        return await _session_request(session, tool, arguments)


class McpClient:
    """Own a session in one async task, retaining stdio processes between tool calls."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = validate_mcp_config(config, allow_stdio=True)
        self.requests: queue.Queue = queue.Queue()
        self.closed = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="mcp-session", daemon=True)
        self.thread.start()

    def request(
        self, tool: str | None = None, arguments: dict[str, Any] | None = None
    ) -> McpResult:
        future: Future = Future()
        with self.lock:
            if self.closed:
                raise McpError("The MCP session ended. Mention the integration again to reconnect.")
            self.requests.put((future, tool, arguments or {}))
        try:
            return future.result(timeout=125)
        except TimeoutError:
            self.close()
            raise McpError("The MCP request timed out. Check the server before retrying.") from None

    def close(self) -> None:
        with self.lock:
            self.closed = True
            self.requests.put(None)

    def _run(self) -> None:
        error = McpError("The MCP session ended. Check the server connection and credentials.")
        try:
            asyncio.run(self._serve())
        except McpError as caught:
            error = caught
        except Exception:
            pass
        finally:
            with self.lock:
                self.closed = True
                while not self.requests.empty():
                    request = self.requests.get_nowait()
                    if request and not request[0].done():
                        request[0].set_exception(error)

    async def _serve(self) -> None:
        # All SDK context managers enter and exit in this single owning task.
        async with _mcp_session(self.config) as session:
            while True:
                request = await asyncio.to_thread(self.requests.get)
                if request is None:
                    return
                future, tool, arguments = request
                try:
                    result = await _session_request(session, tool, arguments)
                except Exception:
                    future.set_exception(
                        McpError("The MCP request failed. Check the server before retrying.")
                    )
                    return
                else:
                    future.set_result(result)


def mcp_request(
    config: Mapping[str, Any], tool: str | None = None, arguments: dict[str, Any] | None = None
) -> McpResult:
    """Discover tools or execute one call, closing all transport resources afterwards."""
    normalized = validate_mcp_config(config, allow_stdio=True)
    try:
        return asyncio.run(_mcp_request(normalized, tool, arguments or {}))
    except McpError:
        raise
    except Exception:
        # Transport exceptions may contain URLs, request headers or credential values.
        raise McpError(
            "The MCP request failed. Check the server URL, transport and credentials, then retry."
        ) from None


@dataclass(frozen=True)
class McpIntegration:
    """A selected, host-authorized integration; never serialized into model context."""

    key: str
    name: str
    revision: str
    discover: Callable[[], list[dict[str, Any]]] = field(repr=False)
    call: Callable[[str, dict[str, Any]], dict[str, Any]] = field(repr=False)
    config: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)


class McpTool(BaseTool):
    """Preserve the server's schema and namespace tools independently of their names."""

    integration_key: str
    remote_name: str
    registry: McpToolRegistry

    def __init__(
        self, integration: McpIntegration, spec: dict[str, Any], registry: McpToolRegistry
    ) -> None:
        remote_name = str(spec["name"])
        digest = hashlib.sha256(f"{integration.key}:{remote_name}".encode()).hexdigest()[:16]
        suffix = re.sub(r"[^a-zA-Z0-9_-]", "_", remote_name)[:40]
        super().__init__(
            name=f"mcp_{digest}_{suffix}",
            description=f"{integration.name}: {spec.get('description') or remote_name}",
            parameters=spec.get("inputSchema") or {"type": "object", "properties": {}},
        )
        object.__setattr__(self, "integration_key", integration.key)
        object.__setattr__(self, "remote_name", remote_name)
        object.__setattr__(self, "registry", registry)

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        # Server annotations are hints, not authority to bypass the host's write policy.
        if context.runtime.tool_manager.mode.policy.recommendations_only:
            return context.json_result({"error": "MCP calls are unavailable in Background mode."})
        try:
            integration = self.registry.resolve(self.integration_key)
        except McpError as error:
            return context.json_result({"error": str(error)})
        if integration is None:
            return context.json_result(
                {"error": "This MCP integration is no longer available in this chat."}
            )
        context.emit_operator_statement(
            self.name, {}, default_statement=f"Using {integration.name}"
        )
        try:
            return context.json_result(self.registry.call(integration, self.remote_name, arguments))
        except McpError as error:
            return context.json_result({"error": str(error)})


class McpToolRegistry:
    """Discover only selected integrations and refresh schemas after configuration changes."""

    def __init__(self, provider: Callable[[], list[McpIntegration]]) -> None:
        self.provider = provider
        self.cache: dict[tuple[str, str], tuple[BaseTool, ...]] = {}
        self.errors: dict[str, str] = {}
        self.clients: dict[tuple[str, str], McpClient] = {}

    def resolve(self, key: str) -> McpIntegration | None:
        return next((item for item in self.provider() if item.key == key), None)

    def close(self) -> None:
        for client in self.clients.values():
            client.close()
        self.clients.clear()
        self.cache.clear()

    def call(
        self, integration: McpIntegration, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        key = (integration.key, integration.revision)
        if integration.config is not None:
            if key not in self.clients:
                self.clients[key] = McpClient(integration.config)
            return self.clients[key].request(name, arguments)
        return integration.call(name, arguments)

    def tools(self) -> tuple[BaseTool, ...]:
        result: list[BaseTool] = []
        try:
            integrations = self.provider()
        except McpError as error:
            self.close()
            self.errors = {"connection": str(error)}
            return ()
        active = {(item.key, item.revision) for item in integrations}
        for key in tuple(self.clients):
            if key not in active:
                self.clients.pop(key).close()
        self.cache = {key: value for key, value in self.cache.items() if key in active}
        self.errors = {
            key: value
            for key, value in self.errors.items()
            if any(item.key == key for item in integrations)
        }
        for item in integrations:
            cache_key = (item.key, item.revision)
            if cache_key not in self.cache:
                try:
                    if item.config is not None:
                        if cache_key not in self.clients or self.clients[cache_key].closed:
                            self.clients[cache_key] = McpClient(item.config)
                        specs = self.clients[cache_key].request()
                    else:
                        specs = item.discover()
                    self.cache[cache_key] = tuple(
                        McpTool(item, spec, self) for spec in specs if spec.get("name")
                    )
                    self.errors.pop(item.key, None)
                except McpError as error:
                    self.errors[item.key] = f"{item.name}: {error}"
                    self.cache[cache_key] = ()
            result.extend(self.cache[cache_key])
        return tuple(result)


__all__ = [
    "McpClient",
    "McpError",
    "McpIntegration",
    "McpTool",
    "McpToolRegistry",
    "mcp_request",
    "validate_mcp_config",
]
