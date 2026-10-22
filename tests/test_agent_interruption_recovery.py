from types import SimpleNamespace

import pytest

from anomx.agent import AnomxHome
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks


@pytest.mark.parametrize("interrupted", [False, True])
def test_tool_checkpoint_preserves_completed_effects_and_marks_unknown_outcomes(
    tmp_path, monkeypatch, interrupted,
):
    home = AnomxHome(tmp_path / "home")
    runtime = AgentRuntime(home, tmp_path)
    session = home.create_session(tmp_path, provider="missing", model="missing")
    effects = []

    def execute(arguments, context):
        started = [
            event["payload"] for event in home.read_session_events(session.path)
            if event.get("payload", {}).get("type") == "tool_execution_started"
        ]
        assert len(started) == 1
        assert "headers" not in started[0]["arguments"]
        effects.append("submitted")
        if interrupted:
            raise ConnectionResetError("Worker stopped after external submission")
        return '{"job_id":"existing-job"}'

    monkeypatch.setattr(runtime, "_tool_for_call", lambda _: SimpleNamespace(execute=execute))
    if interrupted:
        with pytest.raises(ConnectionResetError):
            runtime._execute_tool(
                "submit", {"headers": {"Authorization": "secret"}},
                RuntimeCallbacks(), session.path,
            )
    else:
        runtime._execute_tool("submit", {}, RuntimeCallbacks(), session.path)

    resumed = AgentRuntime(home, tmp_path)
    messages = resumed.conversation_messages(session.path)
    assert effects == ["submitted"]
    if interrupted:
        assert len(messages) == 1
        assert "outcome is unknown" in messages[0]["content"]
        assert "Inspect the current state" in messages[0]["content"]
        assert "secret" not in messages[0]["content"]
    else:
        assert len(messages) == 2
        assert messages[-1]["content"] == '{"job_id":"existing-job"}'
        assert "Runtime recovery" not in str(messages)


def test_resume_reuses_saved_user_message_without_readding_request(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    runtime = AgentRuntime(home, tmp_path)
    session = home.create_session(tmp_path, provider="missing", model="missing")
    home.append_session_event(session.path, "user_message", {
        "message": "Submit one job", "message_id": "input-turn",
    })
    home.append_session_event(session.path, "tool_execution", {
        "tool": "submit", "arguments": {}, "result": '{"job_id":"existing-job"}',
    })
    conversations = []
    monkeypatch.setattr(runtime, "backend_response", lambda path, **_: conversations.append(
        runtime.conversation_messages(path),
    ) or "Continuing")

    assert runtime.backend_response_for_prompt(
        "Submit one job", parent_session_path=session.path,
        prompt_message_id="input-turn", resume=True,
    ) == "Continuing"
    assert [entry["content"] for entry in conversations[0] if entry["role"] == "user"] == [
        "Submit one job",
    ]
    assert conversations[0][-1]["content"] == '{"job_id":"existing-job"}'
