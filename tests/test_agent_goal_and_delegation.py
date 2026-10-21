import json
from pathlib import Path

import pytest

from anomx.agent.backends.openai import OpenAIBackend
from anomx.agent.backends.openai_chat import OpenAICompatibleChatBackend
from anomx.agent.base.agents import AgentKind
from anomx.agent.base.backends import (
    OpenAIChatCompletionStreamResponse,
    OpenAIStreamResponse,
    OpenAIToolCall,
)
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.store import AnomxHome


def runtime_session(tmp_path, **kwargs):
    home = AnomxHome(tmp_path / "home")
    task = tmp_path / "task"
    task.mkdir()
    session = home.create_session(task, provider="openai", model="gpt-5.5")
    return AgentRuntime(home, task, **kwargs), session


def test_features_require_explicit_activation_and_persist(tmp_path):
    runtime, session = runtime_session(tmp_path)
    assert "start_subagent" not in {tool.name for tool in runtime._available_tools()}
    assert "finish_goal" not in {tool.name for tool in runtime._available_tools()}
    runtime.configure_session_features(session.path, subagents_enabled=True, goal_enabled=True)

    resumed = AgentRuntime(runtime.home, runtime.cwd)
    resumed._prepare_session_features(session.path)
    assert {"start_subagent", "finish_goal"} <= {tool.name for tool in resumed._available_tools()}
    assert resumed.home.tmp_dir == session.path.parent / "tmp"
    assert (resumed.cwd / "GOAL.md").is_file()


def test_goal_checks_remain_hidden_until_explicit_finish(tmp_path):
    runtime, session = runtime_session(tmp_path)
    runtime.home.append_session_event(session.path, "user_message", {"message": "Fix the report"})
    runtime.configure_session_features(session.path, goal_enabled=True)
    runtime._prepare_session_features(session.path)
    visible = []
    callbacks = RuntimeCallbacks(message=visible.append, status=visible.append)
    initial_events = runtime.home.read_session_events(session.path)
    for _ in range(8):
        continuation, plan_guard = runtime._continuation_prompt_after_text(
            "Done",
            callbacks,
            session.path,
            100,
        )
        assert "GOAL.md" in continuation and "finish_goal" in continuation
        assert plan_guard is False
    assert not visible
    assert runtime.home.read_session_events(session.path) == initial_events
    assert "Fix the report" in (runtime.cwd / "GOAL.md").read_text()

    result = json.loads(
        runtime._execute_tool(
            "finish_goal",
            {"status": "completed", "summary": "Report corrected and checked."},
            callbacks,
            session.path,
        )
    )
    assert result["finished"]
    assert runtime._continuation_prompt_after_text("Done", callbacks, session.path, 0) == (
        None,
        False,
    )
    assert "Report corrected and checked." in (runtime.cwd / "GOAL.md").read_text()
    assert (
        AgentRuntime(runtime.home, runtime.cwd).session_features(session.path)["goal_enabled"]
        is False
    )


def test_goal_preserves_existing_document_and_blocks_premature_platform_output(tmp_path):
    runtime, session = runtime_session(tmp_path)
    goal = runtime.cwd / "GOAL.md"
    goal.write_text("# Existing objective\nDo the remaining work.\n")
    runtime.configure_session_features(session.path, goal_enabled=True)
    runtime._prepare_session_features(session.path)
    assert goal.read_text() == "# Existing objective\nDo the remaining work.\n"
    runtime.has_platform_connection = lambda: True
    runtime.can_output_response = lambda: True
    outputs = []
    result = json.loads(
        runtime._execute_tool(
            "produce_output",
            {"items": [{"kind": "text", "content": "Done"}]},
            RuntimeCallbacks(output_response=outputs.append),
            session.path,
        )
    )
    assert result["goal_active"] is True
    assert not outputs


def test_goal_finish_cannot_skip_running_workers_or_missing_reason(tmp_path, monkeypatch):
    runtime, session = runtime_session(tmp_path, goal_enabled=True)
    assert "error" in runtime.finish_goal(session.path, status="completed", summary="")
    monkeypatch.setattr(runtime, "_running_subagent_states", lambda: ("active",))
    assert "error" in runtime.finish_goal(session.path, status="completed", summary="Done")
    assert runtime.goal_enabled


@pytest.mark.parametrize("local_sandbox", [False, True])
def test_subagent_handoff_uses_one_folder_and_retriggers_after_restart(
    tmp_path,
    monkeypatch,
    local_sandbox,
):
    runtime, session = runtime_session(tmp_path, local_sandbox_enabled=local_sandbox)
    runtime.configure_session_features(session.path, subagents_enabled=True)
    invocations = []

    def respond(child, session_path, callbacks=None, **kwargs):
        assert child.agent_kind == AgentKind.SUB
        assert child.cwd == session_path.parent
        assert child.cwd != runtime.cwd
        if local_sandbox:
            assert child.local_sandbox_session.current_dir == child.cwd
        assert "Inspect the data" in (child.cwd / "AGENT.md").read_text()
        assert "INBOX.md" in child.home.read_session_events(session_path)[-1]["payload"]["message"]
        invocations.append((child.cwd / "INBOX.md").read_text())
        return "Checked the data; evidence is in result.md."

    monkeypatch.setattr(AgentRuntime, "backend_response", respond)
    started = json.loads(
        runtime._execute_tool(
            "start_subagent",
            {"statement": "Inspecting", "name": "Data audit", "prompt": "Inspect the data"},
            RuntimeCallbacks(),
            session.path,
        )
    )
    state = runtime._subagents[started["agent_id"]]
    state.worker.join(3)
    folder = Path(started["directory"])
    assert state.status == "ready", state.error
    assert {item.name for item in runtime.home.sub_agents_dir.iterdir()} == {folder.name}
    assert "Checked the data" in (folder / "OUTBOX.md").read_text()
    assert "Status: ready" in (folder / "STATUS.md").read_text()
    continuation, _ = runtime._continuation_prompt_after_text(
        "Done", RuntimeCallbacks(), session.path, 0
    )
    assert "OUTBOX.md" in continuation

    resumed = AgentRuntime(runtime.home, runtime.cwd, local_sandbox_enabled=local_sandbox)
    resumed._prepare_session_features(session.path)
    prompted = json.loads(
        resumed._execute_tool(
            "prompt_subagent",
            {
                "statement": "Check again",
                "agent_id": state.agent_id,
                "prompt": "Verify timestamps too",
            },
            RuntimeCallbacks(),
            session.path,
        )
    )
    restored = resumed._subagents[state.agent_id]
    restored.worker.join(3)
    assert prompted["prompted"]
    assert restored.status == "ready", restored.error
    assert "Inspect the data" in invocations[-1]
    assert "Verify timestamps too" in invocations[-1]
    assert (folder / "OUTBOX.md").read_text().count("Checked the data") == 2


def test_subagent_cannot_enable_nested_delegation(tmp_path):
    runtime, session = runtime_session(tmp_path, agent_kind=AgentKind.SUB)
    runtime.configure_session_features(session.path, subagents_enabled=True, goal_enabled=True)
    names = {tool.name for tool in runtime._available_tools()}
    assert "start_subagent" not in names
    assert "finish_goal" not in names


def test_session_features_do_not_leak_to_another_chat(tmp_path):
    runtime, session = runtime_session(tmp_path)
    runtime.configure_session_features(session.path, subagents_enabled=True, goal_enabled=True)
    other = runtime.home.create_session(runtime.cwd, provider="openai", model="gpt-5.5")
    assert runtime.session_features(other.path) == {
        "goal_enabled": False,
        "subagents_enabled": False,
    }


def test_goal_file_symlink_is_not_overwritten(tmp_path):
    runtime, session = runtime_session(tmp_path, goal_enabled=True)
    protected = tmp_path / "private.md"
    protected.write_text("keep")
    (runtime.cwd / "GOAL.md").symlink_to(protected)
    with pytest.raises(ValueError, match="symbolic link"):
        runtime.finish_goal(session.path, status="completed", summary="Done")
    assert protected.read_text() == "keep"


@pytest.mark.parametrize("provider", ["responses", "chat"])
def test_goal_backend_loop_only_delivers_final_after_finish_tool(tmp_path, monkeypatch, provider):
    runtime, session = runtime_session(tmp_path)
    runtime.configure_session_features(session.path, goal_enabled=True)
    runtime._prepare_session_features(session.path)
    backend = (
        OpenAIBackend(runtime) if provider == "responses" else OpenAICompatibleChatBackend(runtime)
    )
    monkeypatch.setattr(backend, "_api_key", lambda *_args: "test-key")
    arguments = json.dumps({"status": "completed", "summary": "All criteria verified."})
    call = OpenAIToolCall("finish_goal", "call-goal", arguments)
    if provider == "responses":
        responses = iter(
            [
                OpenAIStreamResponse("r1", "Provisional answer", ()),
                OpenAIStreamResponse("r2", "", (call,)),
                OpenAIStreamResponse("r3", "Verified final answer", ()),
            ]
        )
        method = "_stream_openai_response"
    else:
        responses = iter(
            [
                OpenAIChatCompletionStreamResponse(
                    "Provisional answer",
                    (),
                    {"role": "assistant", "content": "Provisional answer"},
                ),
                OpenAIChatCompletionStreamResponse(
                    "",
                    (call,),
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-goal",
                                "type": "function",
                                "function": {"name": "finish_goal", "arguments": arguments},
                            }
                        ],
                    },
                ),
                OpenAIChatCompletionStreamResponse(
                    "Verified final answer",
                    (),
                    {"role": "assistant", "content": "Verified final answer"},
                ),
            ]
        )
        method = "_stream_chat_completion"
    requests = []

    def respond(_key, payload, *_callbacks):
        requests.append(json.loads(json.dumps(payload)))
        return next(responses)

    monkeypatch.setattr(backend, method, respond)
    final = []
    result = backend.generate(session.path, "gpt-5.5", RuntimeCallbacks(finish=final.append))
    assert result == "Verified final answer"
    assert final == ["Verified final answer"]
    assert "Are you truly done?" in json.dumps(requests[1])
    assert runtime.session_features(session.path)["goal_enabled"] is False
    assert all(
        "Are you truly done?" not in str(event)
        for event in runtime.home.read_session_events(session.path)
    )
