"""Context activity and durable reductions across provider request boundaries."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from anomx.agent import AnomxHome
from anomx.agent.backends.desy_assistant import DesyAssistantBackend
from anomx.agent.backends.ollama import OllamaBackend
from anomx.agent.backends.openai import OpenAIBackend
from anomx.agent.backends.openai_chat import OpenAICompatibleChatBackend
from anomx.agent.context_management import ContextMessage, ContextToolResult
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks


def make_runtime(tmp_path, **kwargs):
    home = AnomxHome(tmp_path / "home")
    home.save_config({**home.load_config(), "maximum_context_tokens": 32_000})
    session = home.create_session(tmp_path, provider="openai", model="gpt-5.5")
    home.append_session_event(
        session.path, "user_message", {"message": "Keep the constraints.", "message_id": "u1"}
    )
    return AgentRuntime(home, tmp_path, **kwargs), session


@pytest.mark.parametrize("reason", ["unavailable", "oversized", "cancelled"])
def test_preflight_does_not_emit_optimization_activity(tmp_path, monkeypatch, reason):
    calls, activities = [], []
    runtime, session = make_runtime(tmp_path)
    runtime._context_activity_callback = activities.append
    monkeypatch.setattr(runtime, "_background_work_backend", lambda _: None)
    if reason != "unavailable":
        runtime.context_optimizer = lambda *_: calls.append(True) or "Digest"
    if reason == "oversized":
        runtime.context_optimizer_context_window = 1_024
    if reason == "cancelled":
        runtime._turn_abort_event.set()
    entries = [
        ContextMessage(
            "t1",
            {
                "role": "assistant",
                "content": "measurement " * 6_000,
                "context_kind": "tool",
            },
        )
    ]
    assert runtime._optimize_tool_blocks(
        session.path,
        entries,
        maximum=32_000,
        current_context_tokens=20_000,
    ) == (entries, False)
    assert calls == activities == []


@pytest.mark.parametrize("answer", ["KEEP", "Measurement x=3."])
def test_activity_begins_at_model_call_and_preserves_provider_token_baseline(tmp_path, answer):
    activities = []

    def optimize(*_):
        assert [activity["status"] for activity in activities] == ["running"]
        return answer

    runtime, session = make_runtime(tmp_path, context_optimizer=optimize)
    runtime._context_activity_callback = activities.append
    entries = [
        ContextMessage(
            "t1",
            {
                "role": "assistant",
                "content": "measurement " * 1_000,
                "context_kind": "tool",
            },
        )
    ]
    result, changed = runtime._optimize_tool_blocks(
        session.path,
        entries,
        maximum=32_000,
        current_context_tokens=20_000,
    )
    saved = sum(entry.estimated_tokens for entry in entries) - sum(
        entry.estimated_tokens for entry in result
    )
    assert activities[-1]["context_tokens_after"] == 20_000 - saved
    assert activities[-1]["changed"] == changed == (answer != "KEEP")
    assert activities[-1]["model_requests"] == 1


def test_unavailable_summary_model_emits_no_running_activity(tmp_path, monkeypatch):
    runtime, session = make_runtime(tmp_path)
    activities = []
    runtime._context_activity_callback = activities.append
    monkeypatch.setattr(runtime, "_background_work_backend", lambda _: None)
    entries = runtime.backend_conversation_entries(session.path)
    assert runtime.compress_in_turn_context(
        session.path,
        entries,
        current_context_tokens=26_000,
        status_callback=None,
    ) == (entries, False)
    assert [(activity["kind"], activity["status"]) for activity in activities] == [
        ("check", "completed"),
    ]
    assert activities[0]["model_requests"] == 0


def test_failed_summary_preserves_provider_context_count(tmp_path):
    runtime, session = make_runtime(tmp_path, context_summarizer=lambda *_: None)
    activities = []
    runtime._context_activity_callback = activities.append
    entries = runtime.backend_conversation_entries(session.path)
    assert runtime.compress_in_turn_context(
        session.path, entries, current_context_tokens=26_000, status_callback=None,
    ) == (entries, False)
    assert [(activity["kind"], activity["status"]) for activity in activities] == [
        ("check", "completed"), ("compression", "running"), ("compression", "failed"),
    ]
    assert activities[-1]["context_tokens_after"] == 26_000
    assert runtime.context_compression_state(session.path) is None


def test_policy_checks_only_emit_at_evaluation_checkpoints(tmp_path):
    calls, activities = [], []
    runtime, session = make_runtime(
        tmp_path, context_optimizer=lambda *_: calls.append(True) or "Digest",
    )
    runtime._context_activity_callback = activities.append
    entries = runtime.backend_conversation_entries(session.path)
    for tokens in (16_000, 17_000, 20_800, 21_000):
        assert runtime.compress_in_turn_context(
            session.path, entries, current_context_tokens=tokens, status_callback=None,
        ) == (entries, False)
    assert calls == []
    assert [activity["context_tokens_before"] for activity in activities] == [16_000, 20_800]
    assert all(activity["kind"] == "check" for activity in activities)
    assert all(activity["model_requests"] == 0 for activity in activities)


@pytest.mark.parametrize(
    "backend_type,method,content_key,reference_key",
    [
        (OpenAIBackend, "_openai_context_entries", "output", "call_id"),
        (OpenAICompatibleChatBackend, "_chat_context_entries", "content", "tool_call_id"),
        (DesyAssistantBackend, "_anthropic_context_entries", "content", "tool_use_id"),
        (OllamaBackend, "_ollama_context_entries", "content", "tool_name"),
    ],
)
def test_provider_tool_block_reduction_survives_resume_and_history_compression(
    tmp_path,
    monkeypatch,
    backend_type,
    method,
    content_key,
    reference_key,
):
    summaries = []
    runtime, session = make_runtime(
        tmp_path,
        context_optimizer=lambda *_: "Measurements x=3 and y=4; both reads completed.",
        context_summarizer=lambda _, prompt: (
            summaries.append(prompt) or "Constraints and x=3, y=4."
        ),
    )
    results = []
    for index in range(2):
        raw = f"raw-evidence-{index}:" + "x" * 10_000
        tool = SimpleNamespace(execute=lambda *_, value=raw: value)
        monkeypatch.setattr(runtime, "_tool_for_call", lambda _, selected=tool: selected)
        results.append(runtime._execute_tool("read", {}, RuntimeCallbacks(), session.path))
    assert all(isinstance(result, ContextToolResult) for result in results)
    outputs = [
        {content_key: result, reference_key: f"call-{i}"} for i, result in enumerate(results)
    ]
    wire = json.dumps(outputs)
    assert all(result.message_id not in wire for result in results)
    assert deepcopy(results)[-1].message_id == results[-1].message_id
    entries = [
        runtime.backend_conversation_entries(session.path)[0],
        *getattr(backend_type(runtime), method)(
            SimpleNamespace(text="Review evidence.", tool_calls=()), outputs
        ),
    ]
    optimized, changed = runtime.compress_in_turn_context(
        session.path,
        entries,
        current_context_tokens=20_000,
        status_callback=None,
    )
    assert changed
    resumed = AgentRuntime(runtime.home, tmp_path).backend_conversation_entries(session.path)
    assert len(resumed) == 2
    assert resumed[0].message_id == "u1"
    assert "Measurements x=3 and y=4" in resumed[1].payload["content"]
    assert "raw-evidence" not in str([entry.payload for entry in resumed])
    assert len(runtime.conversation_messages(session.path)) == 3

    _, compressed = runtime._compress_context_entries(
        session.path,
        optimized,
        current_context_tokens=29_000,
        status_callback=None,
        minimum_retained_messages=0,
        compress_all=True,
    )
    assert compressed
    state = runtime.context_compression_state(session.path)
    assert state.last_message_id == results[-1].message_id
    assert state.compressed_message_count == 3
    assert len(summaries) == 1
    assert "raw-evidence" not in summaries[0]
    assert "Measurements x=3 and y=4" in summaries[0]
    assert runtime.backend_conversation_entries(session.path) == []
