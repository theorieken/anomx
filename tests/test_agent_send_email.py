import json

import pytest

from anomx.agent.base.tools import ToolExecutionContext
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.store import AnomxHome
from anomx.agent.tools.send_email import SendEmailTool


@pytest.fixture
def email_context(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    home.ensure()
    home.set_platform_connection(
        url="https://example.invalid",
        token="test-token",
        user_email="agent@example.invalid",
        organization_url="example",
        hostname="test",
    )
    runtime = AgentRuntime(home, tmp_path, mode=AgentMode.AUTONOMOUS, agent_user=True)
    calls = []

    def call(connection, **kwargs):
        calls.append(kwargs)
        return {"ok": True, "response": {"sent": True}}

    monkeypatch.setattr("anomx.agent.tools.use_anomx_api.call_anomx_api", call)
    return runtime, calls, ToolExecutionContext(runtime=runtime, callbacks=RuntimeCallbacks())


def message():
    return {
        "recipient": "operator@example.invalid",
        "subject": "Status",
        "body": "Inspection complete.",
        "delivery_key": "inspection-123",
    }


def test_email_is_available_only_to_connected_autonomous_accounts(email_context):
    runtime, calls, context = email_context
    assert runtime._tool_for_call("send_email") is not None
    runtime.agent_user = False
    assert runtime._tool_for_call("send_email") is None
    assert json.loads(SendEmailTool().execute(message(), context))["ok"] is False
    assert calls == []


def test_delivery_retries_preserve_the_same_request(email_context):
    _, calls, context = email_context
    tool = SendEmailTool()
    tool.execute(message(), context)
    tool.execute(message(), context)
    assert len(calls) == 2
    assert calls[0]["path"] == "/agents/emails"
    assert calls[0]["method"] == "POST"
    assert calls[0]["body"] == calls[1]["body"] == message()


@pytest.mark.parametrize(
    "field,value",
    [
        ("recipient", "operator@example.invalid\nBcc: other@example.invalid"),
        ("subject", "Status\r\nBcc: other@example.invalid"),
        ("body", ""),
        ("delivery_key", "x" * 201),
    ],
)
def test_invalid_messages_do_not_reach_the_api(email_context, field, value):
    _, calls, context = email_context
    assert json.loads(SendEmailTool().execute({**message(), field: value}, context))["ok"] is False
    assert calls == []


def test_email_preserves_background_mode_restrictions(email_context):
    runtime, calls, context = email_context
    runtime.tool_manager.mode = AgentMode.BACKGROUND
    result = json.loads(SendEmailTool().execute(message(), context))
    assert result["blocked_by_mode"] is True
    assert calls == []
