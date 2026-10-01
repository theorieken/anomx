import json
from types import SimpleNamespace

import pytest

from anomx.agent import AgentMode
from anomx.agent.base.tools import ToolExecutionContext
from anomx.agent.runtime import AgentRuntime
from anomx.agent.store import AnomxHome
from anomx.agent.tools.produce_output import ProduceOutputTool


def output_context(*, enabled=True):
    emitted = []
    runtime = SimpleNamespace(can_output_response=lambda: enabled, produced_output=None)
    callbacks = SimpleNamespace(output_response=emitted.append)
    return ToolExecutionContext(runtime=runtime, callbacks=callbacks), emitted


def test_output_preserves_body_order_and_moves_sources_last():
    context, emitted = output_context()
    items = [
        {"kind": "reference", "content": {"url": "https://example.org/source"}},
        {"kind": "text", "content": "**Result**"},
        {"kind": "object", "content": {"object_reference": "content_page_example"}},
        {"kind": "objects", "content": ["data_channel_a", "data_channel_b"]},
        {
            "kind": "database",
            "content": {"model_reference": "data_channel", "search": "temperature"},
        },
        {"kind": "reference", "content": {"object_reference": "data_channel_a"}},
    ]
    result = json.loads(ProduceOutputTool().execute({"items": items}, context))
    assert result["ok"] is True
    assert emitted[0]["items"] == [items[1], items[2], items[3], items[4], items[0], items[5]]
    assert context.runtime.produced_output == "**Result**"


@pytest.mark.parametrize(
    "item",
    [
        {"kind": "object", "content": {}},
        {"kind": "objects", "content": [3]},
        {"kind": "database", "content": {"model_reference": "data_channel", "view": "unknown"}},
        {"kind": "reference", "content": {"url": "javascript:alert(1)"}},
        {"kind": "reference", "content": {"url": "https://[invalid"}},
        {"kind": "reference", "content": {"url": "https://user:secret@example.org"}},
        {"kind": "reference", "content": {"url": "https://example.org", "object_reference": "a"}},
        {"kind": "text", "content": ""},
        {"kind": "other", "content": {}},
    ],
)
def test_invalid_output_does_not_partially_publish(item):
    context, emitted = output_context()
    result = json.loads(
        ProduceOutputTool().execute(
            {"items": [{"kind": "text", "content": "Valid"}, item]}, context
        )
    )
    assert result["ok"] is False
    assert emitted == []
    assert context.runtime.produced_output is None


def test_output_cannot_publish_from_cli():
    context, emitted = output_context(enabled=False)
    result = json.loads(
        ProduceOutputTool().execute({"items": [{"kind": "text", "content": "Hello"}]}, context)
    )
    assert result["ok"] is False
    assert emitted == []


def test_missing_and_legacy_modes_use_automatic():
    assert AgentMode.parse(None) is AgentMode.AUTOMATIC
    assert AgentMode.parse("invalid") is AgentMode.AUTOMATIC
    assert AgentMode.parse("plan") is AgentMode.AUTOMATIC
    assert "plan" not in {mode.value for mode in AgentMode}


def test_connected_cli_does_not_advertise_platform_output(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    runtime = AgentRuntime(home, tmp_path)
    monkeypatch.setattr(runtime, "has_platform_connection", lambda: True)
    assert "produce_output" not in {tool.name for tool in runtime._available_tools()}
    config = home.load_config()
    config.update(running_in_anomx_platform=True, platform_output_response_enabled=True)
    monkeypatch.setattr(home, "load_config", lambda: config)
    assert "produce_output" not in {tool.name for tool in runtime._available_tools()}


def test_platform_chat_advertises_output_to_the_model(tmp_path, monkeypatch):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path, platform_chat_id="chat-id")
    monkeypatch.setattr(runtime, "has_platform_connection", lambda: False)
    assert "produce_output" not in {tool.name for tool in runtime._available_tools()}
    monkeypatch.setattr(runtime, "has_platform_connection", lambda: True)
    assert {"produce_output", "focus_object"} <= {tool["name"] for tool in runtime._openai_tools()}
    assert "# Output Contract" in runtime._instructions()
