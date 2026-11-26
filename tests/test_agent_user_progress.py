import pytest

from anomx.agent import AgentKind, AgentMode
from anomx.agent.helpers.user_progress import UserProgress
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.store import AnomxHome


def test_silent_time_and_tool_batches_remind_without_extra_model_calls():
    now = [100.0]
    progress = UserProgress(clock=lambda: now[0])
    now[0] += 44
    assert not progress.reminder()
    now[0] += 1
    assert "User update due" in progress.reminder()
    progress.text_received(" ")
    assert progress.reminder()
    progress.text_received("Two jobs are active; one reports stale health.")
    assert not progress.reminder()
    for _ in range(4):
        progress.tool_completed()
    assert not progress.reminder()
    progress.tool_completed()
    assert "deliver the final result now" in progress.reminder()
    progress.text_received("The status check is complete.")
    assert not progress.reminder()


@pytest.mark.parametrize("cloud", [False, True])
def test_runtime_tracks_visible_text_but_not_thoughts_or_tool_labels(tmp_path, monkeypatch, cloud):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path,
                           platform_chat_id="chat-id" if cloud else "")
    if cloud:
        monkeypatch.setattr(runtime, "has_platform_connection", lambda: True)
    visible, thoughts = [], []
    callbacks = runtime._with_usage_tracking(RuntimeCallbacks(
        delta=visible.append, thought=thoughts.append,
    ))
    monkeypatch.setattr(runtime, "_execute_with_approval", lambda *_: '{}')
    for _ in range(5):
        runtime._execute_tool("read", {"path": "example"}, callbacks)
    assert "User update due" in runtime._instructions()
    callbacks.thought("Internal reasoning")
    assert "User update due" in runtime._instructions()
    callbacks.delta("The job is running; its health timestamp is stale.")
    assert "User update due" not in runtime._instructions()
    assert visible == ["The job is running; its health timestamp is stale."]
    assert thoughts == ["Internal reasoning"]
    runtime._with_usage_tracking(RuntimeCallbacks())
    assert not runtime._user_progress.reminder()


@pytest.mark.parametrize("options", [
    {"mode": AgentMode.BACKGROUND}, {"agent_kind": AgentKind.SUB},
    {"task_instructions": "Import the assigned dataset only."},
])
def test_noninteractive_agents_do_not_get_user_update_reminders(tmp_path, options):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path, **options)
    runtime._with_usage_tracking(RuntimeCallbacks())
    assert runtime._user_progress is None
    assert "User update due" not in runtime._instructions()
