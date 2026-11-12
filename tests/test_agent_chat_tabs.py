import json
from types import SimpleNamespace

import pytest

from anomx.agent.base.tools import ToolExecutionContext
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.store import AnomxHome
from anomx.agent.tools.chat_tabs import ChatTabTool


@pytest.mark.parametrize("operation,arguments,expected", [
    ("list", {}, {"operation": "list"}),
    ("open", {"object_reference": " pages_page-12345678-1234-1234-1234-123456789abc "}, {"operation": "open", "object_reference": "pages_page-12345678-1234-1234-1234-123456789abc"}),
    ("select", {"tab_id": "tab-1"}, {"operation": "select", "tab_id": "tab-1"}),
    ("close", {"tab_id": "tab-1"}, {"operation": "close", "tab_id": "tab-1"}),
])
def test_tab_tools_delegate_only_validated_arguments(operation, arguments, expected, tmp_path):
    calls = []
    context = ToolExecutionContext(runtime=SimpleNamespace(can_output_response=lambda: True, tool_manager=None, home=SimpleNamespace(brain_dir=tmp_path)), callbacks=RuntimeCallbacks(manage_tabs=lambda request: calls.append(request) or {"tabs": {"revision": 2}}))
    result = json.loads(ChatTabTool(operation, statement_description="Working").execute({**arguments, "untrusted": "ignored"}, context))
    assert calls == [expected]
    assert result == {"ok": True, "tabs": {"revision": 2}}


def test_tab_permission_failure_is_reported_without_finishing_turn():
    def deny(_request):
        raise ValueError("Object is not accessible.")

    runtime = SimpleNamespace(can_output_response=lambda: True, produced_output=None)
    context = ToolExecutionContext(runtime=runtime, callbacks=RuntimeCallbacks(manage_tabs=deny))
    result = json.loads(ChatTabTool("open", statement_description="Working").execute({"object_reference": "unknown"}, context))
    assert result == {"ok": False, "error": "Object is not accessible."}
    assert runtime.produced_output is None


@pytest.mark.parametrize("platform", [False, True])
def test_tabs_only_exposed_to_platform_chat(tmp_path, monkeypatch, platform):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path, platform_chat_id="chat-id" if platform else "")
    monkeypatch.setattr(runtime, "has_platform_connection", lambda: True)
    names = {tool.name for tool in runtime._available_tools()}
    assert {"list_tabs", "open_tab", "select_tab", "close_tab"}.issubset(names) is platform


@pytest.mark.parametrize("operation", ["open", "select", "close"])
def test_missing_tab_arguments_never_call_host(operation):
    calls = []
    context = ToolExecutionContext(runtime=SimpleNamespace(can_output_response=lambda: True), callbacks=RuntimeCallbacks(manage_tabs=lambda request: calls.append(request) or {}))
    result = json.loads(ChatTabTool(operation, statement_description="Working").execute({}, context))
    assert result["ok"] is False
    assert calls == []
