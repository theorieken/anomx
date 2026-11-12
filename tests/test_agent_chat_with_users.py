import json
from uuid import uuid4

import pytest

from anomx.agent.base.tools import ToolExecutionContext
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.store import AnomxHome
from anomx.agent.tools.chat_with_users import ChatWithUsersTool


@pytest.fixture
def messaging(tmp_path, monkeypatch):
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
    requests = []

    def call(connection, **kwargs):
        requests.append(kwargs)
        return {"ok": True, "response": {"id": str(uuid4())}}

    monkeypatch.setattr("anomx.agent.tools.use_anomx_api.call_anomx_api", call)
    return runtime, requests, ToolExecutionContext(runtime=runtime, callbacks=RuntimeCallbacks())


def test_only_connected_agent_users_receive_messaging(messaging, tmp_path):
    runtime, _, _ = messaging
    assert runtime._tool_for_call("chat_with_users") is not None
    runtime.agent_user = False
    assert runtime._tool_for_call("chat_with_users") is None
    disconnected = AgentRuntime(AnomxHome(tmp_path / "disconnected"), tmp_path, agent_user=True)
    assert disconnected._tool_for_call("chat_with_users") is None


def test_message_retries_have_stable_delivery_ids_and_new_messages_do_not(messaging):
    _, requests, context = messaging
    tool = ChatWithUsersTool()
    args = {
        "action": "send",
        "thread_id": str(uuid4()),
        "content": "Ready to work",
        "delivery_key": "created:agent",
    }
    assert json.loads(tool.execute(args, context))["ok"]
    tool.execute(args, context)
    tool.execute({**args, "delivery_key": "next-message"}, context)
    assert requests[0]["body"]["client_message_id"] == requests[1]["body"]["client_message_id"]
    assert requests[0]["body"]["client_message_id"] != requests[2]["body"]["client_message_id"]
    assert requests[0]["path"] == f"/chats/chats/{args['thread_id']}/messages/"


@pytest.mark.parametrize(
    "args",
    [
        {"action": "start_thread", "user_id": "../../auth/tokens"},
        {"action": "send", "thread_id": str(uuid4()), "content": "Hello"},
        {"action": "read", "thread_id": str(uuid4()), "limit": True},
        {"action": "react", "message_id": str(uuid4()), "emoji": "👍", "remove": "false"},
        {"action": "unknown"},
    ],
)
def test_invalid_messaging_arguments_do_not_make_requests(messaging, args):
    _, requests, context = messaging
    assert json.loads(ChatWithUsersTool().execute(args, context))["ok"] is False
    assert not requests


def test_reaction_and_read_receipt_target_one_message(messaging):
    _, requests, context = messaging
    message_id = str(uuid4())
    tool = ChatWithUsersTool()
    tool.execute({"action": "react", "message_id": message_id, "emoji": "👍"}, context)
    tool.execute(
        {"action": "react", "message_id": message_id, "emoji": "👍", "remove": True}, context
    )
    tool.execute({"action": "mark_read", "message_id": message_id}, context)
    assert requests[0]["body"] == {"emoji": "👍", "remove": False}
    assert requests[1]["body"] == {"emoji": "👍", "remove": True}
    assert requests[2]["path"] == f"/chats/messages/{message_id}/mark-read/"


def test_messaging_still_enforces_runtime_write_policy(messaging):
    runtime, requests, context = messaging
    runtime.tool_manager.mode = AgentMode.RECOMMEND
    result = json.loads(
        ChatWithUsersTool().execute({"action": "start_thread", "user_id": str(uuid4())}, context)
    )
    assert result["approved"] is False
    assert not requests
