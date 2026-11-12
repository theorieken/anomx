"""Pressure policy and recovery of long-running recurring conversations."""

import json

import pytest

from anomx.agent.base.tools import ToolExecutionContext
from anomx.agent.context_management import (
    ContextMessage,
    ContextPressure,
    context_summary_batches,
    retained_history,
)
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.store import AnomxHome
from anomx.agent.tools.compress_context import CompressContextTool


def make_runtime(tmp_path, summarizer):
    home = AnomxHome(tmp_path / "home")
    home.save_config({**home.load_config(), "maximum_context_tokens": 256_000})
    session = home.create_session(tmp_path, provider="desy", model="coding")
    return AgentRuntime(home, tmp_path, context_summarizer=summarizer), session.path


def test_repeated_scheduled_prompt_no_longer_makes_compression_impossible(tmp_path):
    summaries = []
    runtime, path = make_runtime(
        tmp_path,
        lambda system, user: summaries.append(user) or "119 checks completed. Keep the checkpoint.",
    )
    repeated = "Recurring task: verify the channel catalog and maintain the checkpoint. " * 150
    for i in range(700):
        runtime.home.append_session_event(
            path, "user_message", {"message": repeated, "message_id": f"u{i}"}
        )
        if i == 350:
            runtime.home.append_session_event(
                path,
                "user_message",
                {
                    "message": "Never change human-curated descriptions.",
                    "message_id": "steering",
                },
            )
        runtime.home.append_session_event(
            path,
            "agent_message",
            {
                "message": f"Run {i}: verified catalog, no changes needed.",
                "message_id": f"a{i}",
            },
        )
    original = runtime.conversation_messages(path)
    runtime._prepare_context_compression(path, RuntimeCallbacks())
    state = runtime.context_compression_state(path)
    assert state and state.context_tokens_after < 256_000 * 0.5
    assert len(summaries) < 4
    assert runtime.conversation_messages(path) == original
    restarted = AgentRuntime(runtime.home, tmp_path)
    remaining = restarted.backend_conversation_entries(path)
    assert any(entry.message_id == "steering" for entry in remaining)
    assert remaining[-1].message_id == "a699"
    assert any(entry.payload.get("content") == repeated.strip() for entry in remaining)


def test_duplicate_handling_keeps_changed_instructions_and_images():
    entries = [
        ContextMessage("a", {"role": "user", "content": "Inspect", "images": ["image-a"]}),
        ContextMessage("b", {"role": "user", "content": "Inspect", "images": ["image-b"]}),
        ContextMessage("c", {"role": "user", "content": "Inspect", "images": ["image-a"]}),
        ContextMessage("d", {"role": "user", "content": "Do not write"}),
    ]
    assert [entry.message_id for entry in retained_history(entries, 3)] == ["b", "c", "d"]
    assert len(context_summary_batches(entries, maximum_batch_tokens=1000)[0]) == 3


@pytest.mark.parametrize(
    "tokens,available,urgent",
    [(127_999, False, False), (128_000, True, False), (204_800, True, True)],
)
def test_tools_and_instructions_follow_pressure(tmp_path, tokens, available, urgent):
    runtime, path = make_runtime(tmp_path, lambda *_: "Summary")
    runtime._context_session_path = path
    runtime._context_pressure[path] = ContextPressure(tokens=tokens)
    tools = {tool["name"] for tool in runtime._tool_definitions()}
    instructions = runtime._instructions(path)
    assert ("compress_context" in tools) is available
    assert ("really important that you compress soon" in instructions) is urgent
    assert instructions.count("## Context capacity") <= 1


def test_agent_requested_checkpoint_is_deferred_until_tool_batch_settles(tmp_path):
    summaries = []
    runtime, path = make_runtime(
        tmp_path, lambda *_: summaries.append(True) or "Verified earlier results."
    )
    for index in range(50):
        runtime.home.append_session_event(
            path,
            "agent_message",
            {
                "message": "Verified measurements. " * 300,
                "message_id": str(index),
            },
        )
    entries = runtime.backend_conversation_entries(path)
    runtime.compress_in_turn_context(
        path, entries, current_context_tokens=140_000, status_callback=None
    )
    assert summaries == []
    result = CompressContextTool().execute(
        {}, ToolExecutionContext(runtime, RuntimeCallbacks(), path)
    )
    assert json.loads(result)["scheduled"]
    assert summaries == []
    remaining, changed = runtime.compress_in_turn_context(
        path, entries, current_context_tokens=140_000, status_callback=None
    )
    assert changed and summaries == [True]
    assert remaining[-1] == entries[-1]
    assert not runtime._context_compression_available(path)
    # The next optional checkpoint becomes available after new work, without
    # having to exceed the pre-compression high-water mark.
    pressure = runtime._context_pressure[path]
    pressure.checks += 8
    pressure.tokens = 128_000
    assert runtime._context_compression_available(path)


def test_hard_limit_bypasses_optional_cooldown(tmp_path):
    runtime, path = make_runtime(tmp_path, lambda *_: "Keep the last measurement.")
    for index in range(50):
        runtime.home.append_session_event(
            path,
            "agent_message",
            {
                "message": "Verified measurements. " * 300,
                "message_id": str(index),
            },
        )
    runtime._context_pressure[path] = ContextPressure(reviewed_at=100, reviewed_tokens=200_000)
    _, changed = runtime.compress_in_turn_context(
        path,
        runtime.backend_conversation_entries(path),
        current_context_tokens=243_200,
        status_callback=None,
    )
    assert changed


def test_optional_summary_failure_preserves_transcript_and_continues(tmp_path):
    runtime, path = make_runtime(tmp_path, lambda *_: None)
    for index in range(50):
        runtime.home.append_session_event(
            path,
            "agent_message",
            {
                "message": "Verified measurements. " * 300,
                "message_id": str(index),
            },
        )
    entries = runtime.backend_conversation_entries(path)
    runtime._context_pressure[path] = ContextPressure(tokens=140_000, requested=True)
    assert runtime.compress_in_turn_context(
        path, entries, current_context_tokens=140_000, status_callback=None
    ) == (entries, False)
    assert runtime.context_compression_state(path) is None
    assert runtime.backend_conversation_entries(path) == entries
