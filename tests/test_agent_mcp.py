"""MCP activation, credential boundaries and real SDK transport behavior."""

from __future__ import annotations

import json
import stat
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from anomx.agent.base.tools import ToolExecutionContext
from anomx.agent.helpers.mcp_integrations import (
    local_integrations,
    save_local_integration,
    selected_cli_integrations,
)
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.mcp import (
    McpClient,
    McpError,
    McpIntegration,
    McpToolRegistry,
    mcp_request,
    validate_mcp_config,
)
from anomx.agent.store import AnomxHome
from anomx.agent.ui.models import MenuChoice
from anomx.agent.ui.views.integrations import McpIntegrationsViewMixin


def context(mode=AgentMode.AUTOMATIC):
    return ToolExecutionContext(
        runtime=SimpleNamespace(
            tool_manager=SimpleNamespace(mode=mode), home=SimpleNamespace(brain_dir=Path("/tmp"))
        ),
        callbacks=SimpleNamespace(command=None, tool_message=None, status=None),
    )


def test_only_user_selected_integrations_are_discovered_and_revocation_blocks_calls(tmp_path):
    calls = []
    schema = {
        "type": "object",
        "properties": {"statement": {"type": "integer"}},
        "required": ["statement"],
    }
    integration = McpIntegration(
        "one",
        "Research",
        "1",
        lambda: calls.append("discover") or [{"name": "read/data", "inputSchema": schema}],
        lambda name, arguments: calls.append((name, arguments)) or {"content": []},
    )
    selected = []
    registry = McpToolRegistry(lambda: selected)
    assert registry.tools() == ()
    assert calls == []
    selected.append(integration)
    (tool,) = registry.tools()
    assert tool.parameters == schema
    assert registry.tools() == (tool,)
    assert calls == ["discover"]
    assert json.loads(tool.execute({"statement": 7}, context())) == {"content": []}
    assert calls[-1] == ("read/data", {"statement": 7})
    assert "Background" in tool.execute({}, context(AgentMode.BACKGROUND))
    selected.clear()
    assert "no longer available" in tool.execute({}, context())
    assert registry.tools() == ()


def test_names_are_unique_and_schema_refreshes_when_configuration_changes():
    spec = {"name": "same.tool", "inputSchema": {"type": "object"}}
    first = McpIntegration("one", "Same name", "1", lambda: [spec], lambda *_: {})
    second = replace(first, key="two")
    selected = [first, second]
    registry = McpToolRegistry(lambda: selected)
    tools = registry.tools()
    assert tools[0].name != tools[1].name
    selected[0] = replace(first, revision="2", discover=lambda: [{"name": "new"}])
    assert registry.tools()[0].remote_name == "new"


def test_missing_server_does_not_remove_other_tools_or_retry_during_same_turn():
    attempts = []

    def fail():
        attempts.append(True)
        raise McpError("Unavailable")

    bad = McpIntegration("bad", "Offline", "1", fail, lambda *_: {})
    good = McpIntegration("good", "Online", "1", lambda: [{"name": "read"}], lambda *_: {})
    registry = McpToolRegistry(lambda: [bad, good])
    assert len(registry.tools()) == 1
    assert len(registry.tools()) == 1
    assert attempts == [True]
    assert registry.errors == {"bad": "Offline: Unavailable"}


def test_local_credentials_are_private_and_tool_output_cannot_activate_mcp(tmp_path):
    home = AnomxHome(tmp_path / "home")
    identifier = save_local_integration(
        home, "Data Source", {"url": "https://example.com/mcp", "token": "secret"}
    )
    path = home.root / "integrations.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert local_integrations(home)[0]["config"]["token"] == "secret"
    session = home.create_session(tmp_path, provider="blablador", model="coding")
    reference = f"mcp://local/{identifier}"
    home.append_session_event(session.path, "agent_message", {"message": reference})
    home.append_session_event(session.path, "tool_result", {"message": reference})
    assert selected_cli_integrations(home, session.path) == []
    home.append_session_event(
        session.path,
        "user_message",
        {"message": "Use @Data Source", "backend_message": f"Use @Data Source [{reference}]"},
    )
    (selected,) = selected_cli_integrations(home, session.path)
    assert selected.name == "Data Source"
    assert "secret" not in repr(selected)
    other_session = home.create_session(tmp_path, provider="blablador", model="coding")
    assert selected_cli_integrations(home, other_session.path) == []


def test_platform_outage_keeps_selected_local_integrations_available(tmp_path, monkeypatch):
    from anomx.agent.helpers import mcp_integrations

    home = AnomxHome(tmp_path / "home")
    identifier = save_local_integration(home, "Local data", {"url": "https://example.com/mcp"})
    session = home.create_session(tmp_path, provider="blablador", model="coding")
    home.append_session_event(
        session.path,
        "user_message",
        {
            "message": f"Use mcp://local/{identifier} and mcp://platform/server",
        },
    )
    monkeypatch.setattr(AnomxHome, "platform_connection", lambda _: {"url": "https://example.com"})

    def disconnected(*_args, **_kwargs):
        raise McpError("Offline")

    monkeypatch.setattr(mcp_integrations, "platform_mcp_request", disconnected)
    local, remote = selected_cli_integrations(home, session.path)
    assert local.key == f"local/{identifier}"
    assert remote.key == "platform/server"
    with pytest.raises(McpError, match="unavailable"):
        remote.discover()


@pytest.mark.parametrize(
    "config",
    [
        {"url": "file:///etc/passwd"},
        {"url": "https://secret@example.com/mcp"},
        {"url": "https://example.com/mcp?token=secret"},
        {"url": "https://example.com/mcp", "headers": '{"X-Key": ["bad"]}'},
        {"url": "https://example.com/mcp", "token": "secret\r\nX:bad"},
        {"transport": "stdio", "command": "arbitrary"},
    ],
)
def test_invalid_or_unsafe_remote_configuration_is_rejected(config):
    with pytest.raises(McpError):
        validate_mcp_config(config)


def test_reference_navigation_skips_section_headings():
    view = McpIntegrationsViewMixin()
    choices = [
        MenuChoice("MCP Integrations", "", selectable=False),
        MenuChoice("Server", "mcp://local/1"),
        MenuChoice("Files", "", selectable=False),
        MenuChoice("File", "file.py"),
    ]
    assert view._reference_selected_index(choices, 0) == 1
    assert view._reference_selected_index(choices, 1, 1) == 3
    assert view._reference_selected_index(choices, 3, -1) == 1


def test_real_stdio_server_preserves_session_and_returns_structured_content(tmp_path):
    pytest.importorskip("mcp")
    server = tmp_path / "server.py"
    server.write_text("""from mcp.server.fastmcp import FastMCP
mcp = FastMCP("fixture")
count = 0
@mcp.tool()
def increment(amount: int = 1) -> dict[str, int]:
    global count
    count += amount
    return {"count": count}
mcp.run()
""")
    client = McpClient({"transport": "stdio", "command": sys.executable, "args": [str(server)]})
    try:
        tools = client.request()
        assert tools[0]["name"] == "increment"
        assert tools[0]["inputSchema"]["properties"]["amount"]["type"] == "integer"
        assert client.request("increment", {"amount": 2})["structuredContent"] == {"count": 2}
        assert client.request("increment", {"amount": 3})["structuredContent"] == {"count": 5}
    finally:
        client.close()
        client.thread.join(5)
    assert not client.thread.is_alive()


def test_transport_failures_do_not_echo_credentials():
    pytest.importorskip("mcp")
    with pytest.raises(McpError) as caught:
        mcp_request(
            {
                "transport": "stdio",
                "command": "/missing/mcp-server",
                "env": {"KEY": "secret-canary"},
            }
        )
    assert "secret-canary" not in str(caught.value)


def test_management_failure_returns_without_hanging():
    view = McpIntegrationsViewMixin()
    view._draw_overlay = lambda *args, **kwargs: None

    def fail():
        raise McpError("Connection unavailable")

    with pytest.raises(McpError, match="Connection unavailable"):
        view._mcp_operation(None, "Connect", fail)


def test_streamable_http_initializes_and_sends_authentication(monkeypatch):
    pytest.importorskip("mcp")
    import asyncio

    import httpx
    from mcp.server.fastmcp import FastMCP

    from anomx.agent.mcp import _mcp_request

    server = FastMCP("fixture", stateless_http=True, json_response=True)

    @server.tool()
    def echo(message: str) -> dict[str, str]:
        return {"message": message}

    app = server.streamable_http_app()
    seen = []

    async def authenticated(scope, receive, send):
        if scope["type"] == "http":
            seen.append(dict(scope["headers"]))
        await app(scope, receive, send)

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(
            transport=httpx.ASGITransport(app=authenticated),
            **kwargs,
        ),
    )

    async def run():
        async with app.router.lifespan_context(app):
            config = validate_mcp_config(
                {
                    "url": "http://localhost:8000/mcp",
                    "token": "canary",
                    "headers": '{"X-Tenant": "research"}',
                }
            )
            tools = await _mcp_request(config, None, {})
            assert tools[0]["name"] == "echo"
            result = await _mcp_request(config, "echo", {"message": "hello"})
            assert result["structuredContent"] == {"message": "hello"}

    asyncio.run(run())
    assert seen
    assert all(headers[b"authorization"] == b"Bearer canary" for headers in seen)
    assert all(headers[b"x-tenant"] == b"research" for headers in seen)
