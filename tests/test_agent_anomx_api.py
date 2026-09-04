import json

import pytest

from anomx.agent.base.tools import ToolExecutionContext
from anomx.agent.helpers.anomx_api import (
    AnomxApiConnection,
    AnomxApiError,
    _build_url,
    _response_preview,
)
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.helpers.tool_manager import ApprovalChoice
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.store import AnomxHome


def connected_runtime(tmp_path, mode: AgentMode) -> AgentRuntime:
    home = AnomxHome(tmp_path / "home")
    home.ensure()
    home.set_platform_connection(
        url="https://platform.example.test",
        token="platform-token",
        user_email="user@example.test",
        organization_url="example",
        hostname="agent-host",
    )
    return AgentRuntime(home, tmp_path, mode=mode)


def execute_api(runtime, *, method, path, callbacks=None):
    tool = runtime._tool_for_call("use_anomx_api")
    assert tool is not None
    return json.loads(
        tool.execute(
            {
                "statement": "Calling the platform API",
                "method": method,
                "path": path,
            },
            ToolExecutionContext(
                runtime=runtime,
                callbacks=callbacks or RuntimeCallbacks(),
            ),
        )
    )


def test_recommend_mode_exposes_only_read_and_recommendation_tools(tmp_path):
    runtime = connected_runtime(tmp_path, AgentMode.RECOMMEND)
    tools = {tool.name for tool in runtime._available_tools()}

    assert "use_anomx_api" in tools
    assert "ask_question" in tools
    assert "start_process" not in tools
    assert "memorize" not in tools
    assert "send_feedback" not in tools


def test_recommend_mode_allows_only_recommendation_creation(tmp_path, monkeypatch):
    runtime = connected_runtime(tmp_path, AgentMode.RECOMMEND)
    calls = []

    def fake_call(_connection, **kwargs):
        calls.append(kwargs)
        return {"ok": True, "response": {"id": "recommendation-id"}}

    monkeypatch.setattr("anomx.agent.tools.use_anomx_api.call_anomx_api", fake_call)

    created = execute_api(runtime, method="POST", path="/recommendations")
    created_absolute = execute_api(
        runtime,
        method="POST",
        path="https://platform.example.test/api/recommendations",
    )
    blocked_update = execute_api(runtime, method="PATCH", path="/recommendations/value")
    blocked_write = execute_api(runtime, method="POST", path="/folders")
    blocked_outside_api = execute_api(
        runtime,
        method="POST",
        path="https://platform.example.test/recommendations",
    )

    assert created["ok"] is True
    assert created_absolute["ok"] is True
    assert [call["path"] for call in calls] == [
        "/recommendations",
        "https://platform.example.test/api/recommendations",
    ]
    assert blocked_update["blocked_by_mode"] is True
    assert blocked_write["blocked_by_mode"] is True
    assert blocked_outside_api["blocked_by_mode"] is True


def test_standard_api_write_uses_command_approval_pipeline(tmp_path, monkeypatch):
    runtime = connected_runtime(tmp_path, AgentMode.STANDARD)
    approvals = []

    monkeypatch.setattr(
        "anomx.agent.tools.use_anomx_api.call_anomx_api",
        lambda _connection, **_kwargs: {"ok": True, "response": {}},
    )

    result = execute_api(
        runtime,
        method="POST",
        path="/folders",
        callbacks=RuntimeCallbacks(
            approval=lambda request: approvals.append(request) or ApprovalChoice.ALLOW,
        ),
    )

    assert result["ok"] is True
    assert len(approvals) == 1
    assert approvals[0].canonical_command == "anomx-api POST /folders"


def test_api_response_preview_is_bounded_and_structured():
    preview, truncated = _response_preview(
        {"count": 100, "results": [{"value": "x" * 5_000} for _ in range(100)]}
    )

    assert truncated is True
    assert preview["count"] == 100
    assert len(preview["results"]) == 20


def test_api_absolute_urls_cannot_leave_connected_origin(tmp_path):
    connection = AnomxApiConnection(
        base_url="https://platform.example.test/api",
        token="token",
        responses_dir=tmp_path,
    )

    with pytest.raises(AnomxApiError, match="connected platform origin"):
        _build_url(connection.base_url, "https://attacker.example/api/objects", None)
