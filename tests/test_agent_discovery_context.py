"""Regressions shaped by the October 8 planned-discovery production failures."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from anomx.agent import AgentMode, AnomxHome
from anomx.agent.backends.desy_assistant import _DesyReasoningBackend
from anomx.agent.base.backends import (
    OpenAIChatCompletionStreamResponse,
    OpenAIToolCall,
    TokenUsage,
)
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks


def discovery_runtime(tmp_path):
    home = AnomxHome(tmp_path / "home")
    home.save_config(
        {
            **home.load_config(),
            "provider": "desy",
            "model": "coding",
            "maximum_context_tokens": 256_000,
        }
    )
    home.set_api_key("desy", "test-key")
    session = home.create_session(tmp_path, provider="desy", model="coding")
    home.append_session_event(
        session.path,
        "user_message",
        {
            "message_id": "request",
            "message": "Discover a channel and its system hierarchy.",
        },
    )
    return AgentRuntime(home, tmp_path, mode=AgentMode.AUTONOMOUS), session.path


@pytest.mark.parametrize(
    "tool,size",
    [
        ("search_anomx_data_channels", 326_808),
        ("use_anomx_api", 1_526_957),
        ("read", 5_456_893),
    ],
)
def test_discovery_continues_after_production_sized_results(tmp_path, monkeypatch, tool, size):
    runtime, path = discovery_runtime(tmp_path)
    backend = _DesyReasoningBackend(runtime)
    raw = json.dumps({"ok": True, "values": "x" * (size - 26)})
    monkeypatch.setattr(
        runtime, "_tool_for_call", lambda _: SimpleNamespace(execute=lambda *_: raw)
    )
    runtime.context_optimizer = lambda *_: pytest.fail("Large results must not need a model")
    runtime.context_summarizer = lambda *_: pytest.fail("This short history must still fit")
    call = {"type": "function", "id": "discovery-1", "function": {"name": tool, "arguments": "{}"}}
    responses = iter(
        [
            OpenAIChatCompletionStreamResponse(
                "",
                (OpenAIToolCall(tool, "discovery-1", "{}"),),
                {"role": "assistant", "tool_calls": [call]},
                usage=TokenUsage(input_tokens=48_072, output_tokens=9_439),
            ),
            OpenAIChatCompletionStreamResponse(
                "Done.", (), {"role": "assistant", "content": "Done."}
            ),
        ]
    )
    requests = []
    monkeypatch.setattr(
        backend,
        "_stream_chat_completion",
        lambda _key, payload, *_: requests.append(deepcopy(payload)) or next(responses),
    )

    assert backend.generate(path, "coding", RuntimeCallbacks()) == "Done."
    outputs = [item for item in requests[1]["messages"] if item["role"] == "tool"]
    assert len(outputs) == 1 and outputs[0]["tool_call_id"] == "discovery-1"
    compact = outputs[0]["content"]
    assert len(compact) < 6_000
    assert Path(json.loads(compact)["result_path"]).read_text() == raw
    resumed = AgentRuntime(runtime.home, tmp_path).backend_conversation_entries(path)
    assert resumed[-1].payload["content"] == compact
    assert resumed[-1].payload["tool_call_id"] == "discovery-1"


def test_old_failed_discovery_can_reduce_below_limit_without_repeating_tools(tmp_path):
    runtime, path = discovery_runtime(tmp_path)
    # Only size/shape is reproduced; no private production results are copied.
    sizes = [
        31_730,
        4_852,
        5_120,
        348,
        22_899,
        341,
        4_710,
        10_392,
        10_291,
        22_490,
        3_305,
        2_048,
        326_808,
        2_561,
        2_412,
        2_645,
        2_234,
        2_357,
        2_722,
    ]
    for index, size in enumerate(sizes):
        runtime.home.append_session_event(
            path,
            "tool_execution",
            {
                "message_id": f"result-{index}",
                "tool_call_id": f"call-{index}",
                "tool": "discovery",
                "arguments": {},
                "result": "x" * size,
            },
        )
    original = runtime.conversation_messages(path)
    runtime.context_optimizer = lambda *_: None
    summaries = []
    runtime.context_summarizer = lambda _system, history: (
        summaries.append(history)
        or "The original discovery goal and verified outcomes remain recorded in the result files."
    )
    entries = runtime.backend_conversation_entries(path)
    assert len(entries) < 48

    optimized, changed = runtime.compress_in_turn_context(
        path,
        entries,
        current_context_tokens=296_901,
        status_callback=None,
    )

    assert changed
    assert runtime.estimate_session_context_tokens(path) < 256_000 * 0.8
    assert runtime.conversation_messages(path) == original
    assert optimized[-1].payload["tool_call_id"] == "call-18"
    assert summaries == []
