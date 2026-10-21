"""CLI execution onboarding, explicit chat modes, and readable output pacing."""

from __future__ import annotations

import curses
import queue
import threading

import pytest

from anomx.agent.app import AnomxCliApp
from anomx.agent.runtime_process import RuntimeProcessClient
from anomx.agent.store import AnomxHome, provider_by_key
from anomx.agent.ui.features import (
    completed_response_lines,
    parse_feature_commands,
    response_reveal_frames,
)
from anomx.agent.ui.models import ActiveSessionTurn, MenuChoice, MessageLine, RuntimeUiEvent


class Window:
    def __init__(self):
        self.writes = []

    def erase(self):
        self.writes.clear()

    def getmaxyx(self):
        return 28, 100

    def addnstr(self, y, x, text, n, attr=0):
        self.writes.append((y, x, text[:n], attr))

    def refresh(self):
        pass

    def nodelay(self, _enabled):
        pass


@pytest.fixture
def chat(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "anomx.agent.file_reference_search.FileReferenceSearchClient.start", lambda _self: None,
    )
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="blablador", model="coding")
    app = AnomxCliApp(home=home, cwd=tmp_path, use_color=False)
    return home, session, app


def test_commands_are_explicit_and_do_not_activate_from_paths_or_code():
    cleaned, features = parse_feature_commands("/subagents /goal Inspect this project")
    assert cleaned == "Inspect this project"
    assert features == {"subagents_enabled": True, "goal_enabled": True}
    for message in (
        "@goal", "@subagents", "mail/goal", "https://example.com/goal",
        "/goal/file.md", "`/goal`", "```\n/subagents\n```", "/goals",
    ):
        assert parse_feature_commands(message) == (message, {})
    assert parse_feature_commands("/goal:off Pause")[1] == {"goal_enabled": False}


def test_modes_are_slash_commands_and_never_file_references(chat, monkeypatch):
    _home, _session, app = chat
    monkeypatch.setattr(app._file_reference_search, "suggestions", lambda _query: [])
    assert app._background_file_reference_suggestions("go") == []
    assert [choice.command for choice in app._filtered_commands("/go")] == ["/goal", "/goal:off"]
    assert "/subagents" in {choice.command for choice in app._project_command_specs()}
    text, cursor = app._insert_feature_command("Inspect /go next", (8, 11, "/go"), "/goal", [])
    assert text == "Inspect /goal next"
    assert cursor == len("Inspect /goal")
    assert app._active_command_token("https://example.com/goal", 24) is None
    assert app._is_feature_prompt("/goal /subagents Inspect the source")
    assert not app._is_feature_prompt("/goal")
    assert not app._is_feature_prompt("/config")
    files = [MenuChoice("goal.txt", "goal.txt")]
    assert app._file_reference_bottom_panel(files, 0).title == "Files"


def test_feature_command_without_task_updates_mode_without_running_a_turn(chat, monkeypatch):
    _home, session, app = chat
    changes = []
    monkeypatch.setattr(
        app.runtime, "configure_session_features",
        lambda path, **features: changes.append((path, features)),
    )
    assert app._handle_command(None, "/goal", session, "/goal /subagents") is None
    assert changes == [(session.path, {"goal_enabled": True, "subagents_enabled": True})]
    app._handle_command(None, "/goal:off", session, "/goal:off")
    assert changes[-1] == (session.path, {"goal_enabled": False})


def test_runtime_choice_disables_missing_engines(chat, monkeypatch):
    _home, _session, app = chat
    monkeypatch.setattr("shutil.which", lambda name: "/bin/docker" if name == "docker" else None)
    choices = app._onboarding_execution_choices()
    assert [(choice.value, choice.selectable) for choice in choices] == [
        ("host", True), ("docker", True), ("podman", False),
    ]
    assert "not installed" in choices[-1].label


def test_onboarding_renders_available_and_disabled_engines_with_wrapped_help(chat, monkeypatch):
    _home, _session, app = chat
    monkeypatch.setattr("shutil.which", lambda name: "/bin/docker" if name == "docker" else None)
    app._colors.update(muted=curses.A_DIM)
    window = Window()
    window.get_wch = lambda: "\n"
    assert app._select_onboarding_execution(window) == "host"
    assert any("Docker" in text and attr == 0 for _, _, text, attr in window.writes)
    assert any("Podman" in text and attr == curses.A_DIM for _, _, text, attr in window.writes)
    assert any("Manage Sandbox" in text for _, _, text, _ in window.writes)


@pytest.mark.parametrize("execution", ["host", "docker", "podman"])
def test_onboarding_saves_execution_default(chat, monkeypatch, execution):
    home, _session, app = chat
    monkeypatch.setattr(app, "_select_onboarding_execution", lambda _screen: execution)
    monkeypatch.setattr(app, "_select_provider", lambda _screen: provider_by_key("openai"))
    monkeypatch.setattr(app, "_select_model", lambda *_args: "gpt-5.5")
    monkeypatch.setattr(app, "_select_thinking_intensity", lambda *_args: "auto")
    answers = iter(["", "Theo"])
    monkeypatch.setattr(app, "_prompt_text", lambda *_args, **_kwargs: next(answers))
    assert app._run_onboarding(Window())
    config = home.load_config()
    assert config["sandbox_enabled"] is (execution != "host")
    if execution != "host":
        assert config["sandbox_system"] == execution
        assert config["sandbox_method"] == "mount"
    assert config["onboarding_complete"]
    assert config["user_name"] == "Theo"


def test_runtime_header_only_replaces_location_for_containers(chat):
    home, session, app = chat
    for system, expected in (("python", str(session.cwd)), ("docker", "Secure Agent Runtime")):
        config = home.load_config()
        config.update(sandbox_enabled=True, sandbox_system=system)
        home.save_config(config)
        app._config_cache_at = 0
        assert app._session_location_line(session) == expected


def test_session_feature_notes_use_events_and_draft_tokens(chat):
    home, session, app = chat
    home.append_session_event(session.path, "agent_features", {
        "subagents_enabled": True, "goal_enabled": True,
    })
    assert app._feature_notes_for_prompt("", session.path) == (
        "Sub agents active", "Goal mode active · GOAL.md",
    )
    window = Window()
    app._draw_session(window, session, [], "", 0, 0)
    assert any(y == 27 and "Sub agents active" in text for y, _, text, _ in window.writes)
    assert app._prompt_layout(window).hint_line == 26
    home.append_session_event(session.path, "agent_features", {
        "subagents_enabled": True, "goal_enabled": False,
    })
    assert app._feature_notes_for_prompt("", session.path) == ("Sub agents active",)
    assert app._feature_notes_for_prompt("/subagents:off", session.path) == ()


def test_line_reveal_preserves_source_and_bounds_long_answer_repaints():
    text = "First line\nSecond line\nFinal"
    assert response_reveal_frames(text, 80) == ("First line\n", "First line\nSecond line\n", text)
    assert completed_response_lines(text, 80) == "First line\nSecond line\n"
    assert completed_response_lines("1234567890partial", 10) == "1234567890"
    long_text = "Some output\n" * 5000
    frames = response_reveal_frames(long_text, 80)
    assert len(frames) <= 61
    assert frames[-1] == long_text


def test_line_reveal_runs_with_isolated_runtime(chat, monkeypatch):
    _home, session, app = chat
    app.isolate_runtime = True
    shown = []
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        app, "_draw_session", lambda _w, _s, lines, *_a, **_k: shown.append(lines[-1].text),
    )
    app._fake_type_message(Window(), session, "One\nTwo\nThree")
    assert shown == ["One\n", "One\nTwo\n", "One\nTwo\nThree"]


def test_live_stream_buffers_partial_line_and_completion_avoids_replay(chat, monkeypatch):
    _home, session, app = chat
    worker = threading.Thread(target=lambda: None)
    worker.start()
    worker.join()
    turn = ActiveSessionTurn(
        session, app.runtime, queue.SimpleQueue(), {}, "turn", 0, worker, app.agent_mode,
        final_text="First line\nPartial",
    )
    assert app._streaming_response_text(Window(), turn) == "First line\n"
    assert not turn.streamed_final_displayed
    shown = []
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        app, "_draw_session", lambda _w, _s, lines, *_a, **_k: shown.append(lines[-1].text),
    )
    app._fake_type_message(
        Window(), session, "First line\nPartial", revealed_prefix=turn.displayed_final_text,
    )
    assert shown == ["First line\nPartial"]


def test_intermediate_messages_reveal_in_collapsed_view(chat, monkeypatch):
    _home, session, app = chat
    app.work_visualization = "collapsed"
    rendered = []
    monkeypatch.setattr(app, "_fake_type_message", lambda *_a, **kw: rendered.append(kw))
    events = queue.SimpleQueue()
    events.put(RuntimeUiEvent("message", "I am inspecting the source."))
    app._process_runtime_events(Window(), session, events, "Thinking", None, "", "turn", 0, None)
    assert len(rendered) == 1
    assert rendered[0]["animate"] is True
    assert MessageLine(
        "agent_intermediate", "I am inspecting the source.", "turn",
    ) in app._read_message_lines(session.path)


def test_podman_selection_is_preserved_when_docker_is_also_installed(chat, monkeypatch):
    home, _session, app = chat
    config = home.load_config()
    config.update(sandbox_enabled=True, sandbox_system="podman", sandbox_method="mount")
    home.save_config(config)
    monkeypatch.setattr("shutil.which", lambda name: f"/bin/{name}")
    assert app._handle_sandbox_check(Window(), None)
    assert home.load_config()["sandbox_system"] == "podman"


def test_typing_skips_animation_without_losing_input(chat, monkeypatch):
    _home, session, app = chat
    window = Window()
    window.get_wch = lambda: "x"
    shown = []
    monkeypatch.setattr(
        app, "_draw_session", lambda _w, _s, lines, *_a, **_k: shown.append(lines[-1].text),
    )
    app._fake_type_message(window, session, "First\nSecond\nThird")
    assert shown == ["First\nSecond\nThird"]
    assert app._read_prompt_key(window) == "x"
    assert app._deferred_prompt_keys == []


def test_mode_mentions_persist_and_are_removed_from_model_request(chat):
    _home, session, app = chat
    cleaned = app._apply_feature_commands(session.path, "/goal /subagents Inspect the source")
    assert cleaned == "Inspect the source"
    assert app.runtime.session_features(session.path) == {
        "subagents_enabled": True, "goal_enabled": True,
    }
    app._apply_feature_commands(session.path, "/goal:off Pause the goal")
    assert app.runtime.session_features(session.path) == {
        "subagents_enabled": True, "goal_enabled": False,
    }


def test_subprocess_goal_completion_refreshes_cli_note(chat):
    home, session, app = chat
    runtime = RuntimeProcessClient(home, app.cwd, workspace_root=app.workspace_root)
    try:
        runtime.configure_session_features(session.path, goal_enabled=True, subagents_enabled=True)
        assert app._feature_notes_for_prompt("", session.path) == (
            "Sub agents active", "Goal mode active · GOAL.md",
        )
        result = runtime._call(
            "finish_goal", session.path, status="completed", summary="Verified the test objective.",
        )
        assert result["finished"]
        assert app._feature_notes_for_prompt("", session.path) == ("Sub agents active",)
    finally:
        runtime.close()


@pytest.mark.parametrize("width", [60, 100, 160])
def test_setup_card_separates_details_and_labels_inputs(chat, width):
    _home, _session, app = chat
    window = Window()
    window.getmaxyx = lambda: (40, width)
    app._onboarding_step = 0
    app._draw_overlay(
        window, "Where would you like to work?", "Choose your default.",
        choices=(MenuChoice("Docker runtime", "docker", "Ready · isolated tools."),),
    )
    label = next((y, x, text) for y, x, text, _ in window.writes if text == "Docker runtime")
    detail = next((y, x, text) for y, x, text, _ in window.writes if text.startswith("Ready"))
    assert detail[0] == label[0] + 1
    assert detail[1] == label[1]
    assert any("1 of 4" in text for _, _, text, _ in window.writes)
    app._onboarding_step = 1
    app._draw_overlay(window, "Connect to DESY", "API key", show_input_cursor=True,
                      input_value="*****", input_cursor=5)
    label_y = next(y for y, _, text, _ in window.writes if text == "API key")
    value_y = next(y for y, _, text, _ in window.writes if text == "*****")
    assert value_y == label_y + 2
    assert any("Esc Back" in text for _, _, text, _ in window.writes)


def test_setup_back_navigation_returns_to_previous_step(chat, monkeypatch):
    home, _session, app = chat
    monkeypatch.setattr(app, "_select_onboarding_execution", lambda _: "host")
    provider_calls = []
    monkeypatch.setattr(app, "_select_provider", lambda _: (
        provider_calls.append(True) or provider_by_key("openai")
    ))
    answers = iter([None, "", "Theo"])
    monkeypatch.setattr(app, "_prompt_text", lambda *a, **kw: next(answers))
    monkeypatch.setattr(app, "_select_model", lambda *a: "gpt-5.5")
    monkeypatch.setattr(app, "_select_thinking_intensity", lambda *a: "auto")
    monkeypatch.setattr(app, "_animate_onboarding_complete", lambda *a: None)
    assert app._run_onboarding(Window())
    assert len(provider_calls) == 2
    assert home.load_config()["onboarding_complete"]
    assert app._onboarding_step is None
