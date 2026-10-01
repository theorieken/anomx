from types import SimpleNamespace

import pytest

from anomx.agent import AnomxHome
from anomx.agent.app import AnomxCliApp
from anomx.agent.context_management import (
    ContextMessage,
    adaptive_context_target,
    context_evaluation_band,
    effective_context_limit,
    tool_context_groups,
)
from anomx.agent.exceptions import AgentBackendError
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks


def runtime_session(tmp_path, **kwargs):
    home = AnomxHome(tmp_path / "home")
    home.save_config({**home.load_config(), "maximum_context_tokens": 32_000})
    session = home.create_session(tmp_path, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, tmp_path, **kwargs)
    return runtime, session


def append(runtime, session, kind, content, message_id):
    runtime.home.append_session_event(
        session.path,
        kind,
        {
            "message": content,
            "result": content,
            "tool": "read_file",
            "message_id": message_id,
        },
    )


def test_legacy_percentage_is_discarded(tmp_path):
    runtime, _ = runtime_session(tmp_path)
    runtime.home.save_config(
        {**runtime.home.load_config(), "context_compression_target_percent": 75}
    )
    assert "context_compression_target_percent" not in runtime.home.load_config()
    assert "context_compression_target_percent" not in runtime.home.config_path.read_text()


def test_limit_and_adaptive_headroom():
    assert effective_context_limit(32_000, "unknown") == 32_000
    assert effective_context_limit(1_000_000, "alias-code") == 262_144 - 32_768
    assert effective_context_limit(32_000, "alias-code") == 32_000
    small = [ContextMessage("", {"role": "user", "content": "hello"})]
    large = [ContextMessage("", {"role": "user", "content": "data" * 10_000})]
    assert adaptive_context_target(32_000, small) == 20_800
    assert adaptive_context_target(32_000, large) == 16_000


@pytest.mark.parametrize(
    "percent,band", [(49, 0), (50, 50), (65, 65), (80, 80), (90, 90), (99, 99), (110, 99)]
)
def test_evaluation_thresholds(percent, band):
    assert context_evaluation_band(percent * 320, 32_000) == band


def test_tool_blocks_do_not_cross_user_or_agent_messages():
    def tool(id):
        return ContextMessage(id, {"role": "assistant", "content": "data", "context_kind": "tool"})

    entries = [
        tool("1"),
        tool("2"),
        ContextMessage("3", {"role": "user", "content": "Keep x"}),
        tool("4"),
        ContextMessage("5", {"role": "assistant", "content": "Next step"}),
        tool("6"),
    ]
    assert [[entry.message_id for entry in group] for group in tool_context_groups(entries)] == [
        ["1", "2"],
        ["4"],
        ["6"],
    ]


def test_followup_optimizes_tool_blocks_and_restores_them_without_losing_transcript(tmp_path):
    calls = []
    runtime, session = runtime_session(
        tmp_path,
        context_optimizer=lambda system, user: (
            calls.append((system, user)) or "Measurement x=3 at /data/result.csv; read completed."
        ),
    )
    append(runtime, session, "user_message", "Inspect the data", "u1")
    append(runtime, session, "tool_execution", "data " * 2_000, "t1")
    append(runtime, session, "tool_execution", "other data " * 1_000, "t2")
    append(runtime, session, "agent_message", "I found a measurement", "a1")
    append(runtime, session, "user_message", "Explain it", "u2")
    full_transcript = runtime.conversation_messages(session.path)
    activities = []
    runtime._prepare_context_compression(
        session.path, RuntimeCallbacks(context_activity=activities.append)
    )
    assert len(calls) == 1
    assert "untrusted data" in calls[0][0]
    assert runtime.conversation_messages(session.path) == full_transcript
    restored = AgentRuntime(runtime.home, tmp_path).backend_conversation_entries(session.path)
    assert [entry.message_id for entry in restored] == ["u1", "t1", "a1", "u2"]
    assert "/data/result.csv" in restored[1].payload["content"]
    assert [activity["status"] for activity in activities] == ["running", "completed"]
    assert activities[-1]["changed"]
    assert runtime.context_compression_state(session.path) is None


@pytest.mark.parametrize("answer", ["KEEP", "", "data " * 10_000])
def test_optimizer_never_replaces_with_empty_or_larger_output(tmp_path, answer):
    runtime, session = runtime_session(tmp_path, context_optimizer=lambda *_: answer)
    append(runtime, session, "tool_execution", "data " * 2_000, "t1")
    before = runtime.backend_conversation_entries(session.path)
    runtime._prepare_context_compression(session.path, RuntimeCallbacks())
    assert runtime.backend_conversation_entries(session.path) == before


def test_large_tool_result_uses_medium_model_and_keeps_original_evidence(tmp_path, monkeypatch):
    runtime, session = runtime_session(tmp_path)
    selection = []

    class Backend:
        def summarize_conversation(self, messages, previous, model, *, system_prompt):
            assert model == "medium-model"
            assert "Preserve exact identifiers" in system_prompt
            return "Read succeeded. x=3, file /data/result.csv."

    monkeypatch.setattr(
        runtime,
        "_background_work_backend",
        lambda key: selection.append(key) or (Backend(), "medium-model"),
    )
    result = "measurement " * 4_000
    monkeypatch.setattr(
        runtime, "_tool_for_call", lambda _: SimpleNamespace(execute=lambda *_: result)
    )
    optimized = runtime._execute_tool("read_file", {}, RuntimeCallbacks(), session.path)
    assert "context_optimized" in optimized and "result_reference" in optimized
    assert selection == ["background_medium_work_model"]
    assert result in runtime.conversation_messages(session.path)[0]["content"]
    assert result not in runtime.backend_conversation_messages(session.path)[0]["content"]


def test_summary_uses_easy_model(tmp_path, monkeypatch):
    runtime, session = runtime_session(tmp_path)
    selections = []

    class Backend:
        def summarize_conversation(self, *args):
            return "Goals and measurements retained."

    monkeypatch.setattr(
        runtime,
        "_background_work_backend",
        lambda key: selections.append(key) or (Backend(), "easy-model"),
    )
    append(runtime, session, "user_message", "Inspect all measurements", "u1")
    _, changed = runtime.compress_in_turn_context(
        session.path,
        runtime.backend_conversation_entries(session.path),
        current_context_tokens=26_000,
        status_callback=None,
    )
    assert changed
    assert selections == ["background_easy_work_model"]


def test_eighty_percent_compresses_and_failed_optional_summary_waits_for_next_band(tmp_path):
    calls = []
    runtime, session = runtime_session(
        tmp_path, context_summarizer=lambda *_: calls.append(True) or None
    )
    append(runtime, session, "user_message", "Inspect all measurements", "u1")
    entries = runtime.backend_conversation_entries(session.path)
    for count in (25_600, 26_000, 28_800):
        result, changed = runtime.compress_in_turn_context(
            session.path, entries, current_context_tokens=count, status_callback=None
        )
        assert not changed and result == entries
    assert len(calls) == 2
    with pytest.raises(AgentBackendError, match="no summary"):
        runtime.compress_in_turn_context(
            session.path, entries, current_context_tokens=31_680, status_callback=None
        )
    assert runtime.context_compression_state(session.path) is None


def test_many_old_messages_trigger_compression_below_eighty_percent(tmp_path):
    runtime, session = runtime_session(
        tmp_path, context_summarizer=lambda *_: "Earlier constraints retained."
    )
    for index in range(50):
        append(
            runtime,
            session,
            "user_message" if index % 2 == 0 else "agent_message",
            "context " * 150,
            str(index),
        )
    tokens = runtime.estimate_session_context_tokens(session.path)
    assert 16_000 <= tokens < 25_600
    runtime._prepare_context_compression(session.path, RuntimeCallbacks())
    assert runtime.context_compression_state(session.path) is not None
    assert runtime.backend_conversation_entries(session.path)[-1].message_id == "49"
    assert len(runtime.conversation_messages(session.path)) == 50


def test_cli_context_activity_splits_tool_blocks_and_updates_in_place(tmp_path):
    runtime, session = runtime_session(tmp_path)
    for kind, payload in [
        ("work_message", {"message": "Read file", "role": "tool", "turn_id": "turn"}),
        ("context_activity", {"id": "opt", "kind": "optimization", "status": "running"}),
        (
            "context_activity",
            {
                "id": "opt",
                "kind": "optimization",
                "status": "completed",
                "changed": True,
                "tokens_before": 8_000,
                "tokens_after": 2_000,
            },
        ),
        ("work_message", {"message": "Continue analysis", "role": "tool", "turn_id": "turn"}),
    ]:
        runtime.home.append_session_event(session.path, kind, payload)
    app = AnomxCliApp(home=runtime.home, isolate_runtime=True)
    lines = app._read_message_lines(session.path)
    assert [line.role for line in lines] == ["work_summary", "context", "work_active"]
    assert "8,000 → 2,000" in lines[1].text


def test_in_turn_optimization_resets_openai_chain_without_summarizing_history(
    tmp_path, monkeypatch
):
    from anomx.agent.backends.openai import OpenAIBackend
    from anomx.agent.base.backends import OpenAIStreamResponse, OpenAIToolCall, TokenUsage

    def no_summary(*_):
        raise AssertionError("Tool optimization should leave history intact")

    runtime, session = runtime_session(
        tmp_path,
        context_optimizer=lambda *_: "Measurement x=3; completed read of /data/result.csv.",
        context_summarizer=no_summary,
    )
    runtime.home.set_api_key("openai", "test-key")
    append(runtime, session, "user_message", "Find x and keep the evidence.", "u1")
    responses = iter(
        [
            OpenAIStreamResponse(
                "r1", "", (OpenAIToolCall("read", "c1", "{}"),), TokenUsage(input_tokens=16_000)
            ),
            OpenAIStreamResponse("r2", "x=3", (), TokenUsage(input_tokens=2_000)),
        ]
    )
    payloads = []
    backend = OpenAIBackend(runtime)
    monkeypatch.setattr(
        backend,
        "_stream_openai_response",
        lambda key, payload, *_: payloads.append(payload) or next(responses),
    )
    monkeypatch.setattr(
        backend,
        "_execute_requested_tools",
        lambda response, *_: (
            [{"type": "function_call_output", "call_id": "c1", "output": "measurement " * 2_000}]
            if response.tool_calls
            else []
        ),
    )
    assert backend.generate(session.path, "gpt-5.5", RuntimeCallbacks()) == "x=3"
    assert "previous_response_id" not in payloads[1]
    assert any("Measurement x=3" in item["content"] for item in payloads[1]["input"])
    assert any("Find x and keep the evidence." in item["content"] for item in payloads[1]["input"])
    assert runtime.context_compression_state(session.path) is None


def test_cli_context_capacity_uses_configured_maximum_and_invalidates_on_change(tmp_path):
    runtime, session = runtime_session(tmp_path)
    append(runtime, session, "user_message", "Inspect the data", "u1")
    app = AnomxCliApp(home=runtime.home, isolate_runtime=True)
    assert "/32k" in app._context_status(session, "gpt-5.5")
    runtime.home.save_config({**runtime.home.load_config(), "maximum_context_tokens": 64_000})
    assert "/64k" in app._context_status(session, "gpt-5.5")


def test_cli_activity_uses_callback_order_when_runtime_writes_ahead(tmp_path):
    runtime, session = runtime_session(tmp_path)
    activity = {"id": "opt", "kind": "optimization", "status": "running"}
    runtime.home.append_session_event(session.path, "context_activity", activity)
    runtime.home.append_session_event(
        session.path,
        "work_message",
        {
            "message": "Read file",
            "role": "tool",
            "turn_id": "turn",
        },
    )
    runtime.home.append_session_event(session.path, "context_activity_display", activity)
    runtime.home.append_session_event(
        session.path,
        "context_activity",
        {
            **activity,
            "status": "completed",
            "changed": False,
        },
    )
    runtime.home.append_session_event(
        session.path,
        "context_activity_display",
        {
            **activity,
            "status": "completed",
            "changed": False,
        },
    )
    runtime.home.append_session_event(
        session.path,
        "work_message",
        {
            "message": "Continue analysis",
            "role": "tool",
            "turn_id": "turn",
        },
    )
    app = AnomxCliApp(home=runtime.home, isolate_runtime=True)
    lines = app._read_message_lines(session.path)
    assert [line.role for line in lines] == ["work_summary", "context", "work_active"]
    assert "context retained" in lines[1].text
