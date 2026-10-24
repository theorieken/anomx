"""Live activity timing and tool execution must not alter transcript history."""

import threading

import pytest

from anomx.agent.app import AnomxCliApp
from anomx.agent.base.tools import BaseTool
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.runtime_process import RuntimeProcessClient
from anomx.agent.store import AnomxHome
from anomx.agent.ui.activity import ActivityPresentation


def test_default_thinking_and_each_thought_gets_one_second():
    activity = ActivityPresentation()
    assert activity.display(0) == "Thinking"
    activity.push("Created a thought", thought=True)
    activity.push("Reading config")
    activity.push("Created a thought", thought=True)
    activity.phase = "Thinking"
    assert activity.display(0.1) == "Created a thought"
    assert activity.display(1.09) == "Created a thought"
    assert activity.display(1.1) == "Reading config"
    assert activity.display(2.09) == "Reading config"
    assert activity.display(2.1) == "Created a thought"
    assert activity.display(3.2) == "Thinking"


def test_specific_statement_survives_model_requests_and_writing_tools_are_visible():
    activity = ActivityPresentation()
    activity.push("Reading config")
    assert activity.display(0) == "Reading config"
    activity.phase = "Writing"
    assert activity.display(3) == "Reading config"
    activity.phase = "Thinking"
    assert activity.display(4) == "Reading config"
    activity.push("Writing configuration file")
    activity.running["write-1"] = "Writing configuration file"
    assert activity.display(5) == "Writing configuration file"
    activity.push("Created a thought", thought=True)
    assert activity.display(6.1) == "Created a thought"
    assert activity.display(7.2) == "Writing configuration file"
    activity.running.clear()
    assert activity.display(8.3) == "Thinking"


@pytest.mark.parametrize("fail", [False, True])
def test_tool_lifecycle_surrounds_execution_and_finishes_on_error(tmp_path, monkeypatch, fail):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    monkeypatch.setattr(runtime, "init_sandbox", lambda **kwargs: None)
    monkeypatch.setattr(runtime, "refresh_mode", lambda: None)
    monkeypatch.setattr(runtime, "_tool_for_call", lambda name: BaseTool(name, "test", {}))
    observed = []

    def execute(*args):
        assert observed[-1]["status"] == "running"
        assert observed[-1]["statement"] == "Inspecting a config file"
        if fail:
            raise RuntimeError("write failed")
        return "{}"

    monkeypatch.setattr(runtime, "_execute_with_approval", execute)
    callbacks = RuntimeCallbacks(tool_activity=observed.append)
    if fail:
        with pytest.raises(RuntimeError, match="write failed"):
            runtime._execute_tool("run_command", {"statement": "Inspecting a config file", "command": "write"}, callbacks)
    else:
        assert runtime._execute_tool("run_command", {"statement": "Inspecting a config file", "command": "write"}, callbacks) == "{}"
    assert [entry["status"] for entry in observed] == ["running", "completed"]
    assert observed[0]["id"] == observed[1]["id"]
    assert RuntimeProcessClient._callback_map(callbacks)["tool_activity"] == observed.append


def test_cli_renders_running_tool_before_completion_without_duplicate_log(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="missing", model="missing")
    home.append_session_event(session.path, "user_message", {"message": "Write a file"})
    app = AnomxCliApp(home=home, cwd=tmp_path, use_color=False)
    runtime = app.runtime
    started = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(runtime, "init_sandbox", lambda **kwargs: None)

    def response(path, *, callbacks):
        payload = {"id": "write-1", "tool": "run_command", "statement": "Writing configuration", "command": "write"}
        callbacks.tool_activity({**payload, "status": "running"})
        started.set()
        release.wait(3)
        callbacks.command("Writing configuration", "write", "done")
        callbacks.tool_activity({**payload, "status": "completed"})
        callbacks.status("Thinking")
        return "Done"

    monkeypatch.setattr(runtime, "backend_response", response)
    turn = app._start_session_turn(session, runtime)
    try:
        assert started.wait(3)
        app._drain_session_turn_events(object(), turn, render_events=False)
        assert turn.working_text == "Writing configuration"
        lines = app._messages_with_working_status(app._read_message_lines(session.path), turn.working_text)
        assert [line.text for line in lines if line.activity_wave] == ["Writing configuration"]
    finally:
        release.set()
        turn.worker.join(3)
    assert turn.result == {"response": "Done"}
    app._drain_session_turn_events(object(), turn, render_events=False)
    work = [event for event in home.read_session_events(session.path) if event.get("payload", {}).get("type") == "work_message"]
    assert len(work) == 1
    assert work[0]["payload"]["message"] == "Writing configuration"
    assert turn.working_text == "Writing configuration"
