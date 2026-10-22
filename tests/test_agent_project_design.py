"""Project focus, header controls, and persisted design preferences."""

from __future__ import annotations

import curses
from dataclasses import replace
from types import SimpleNamespace

import pytest

from anomx.agent.app import AnomxCliApp
from anomx.agent.store import AnomxHome
from anomx.agent.ui.models import PromptPasteEvent


class Window:
    def __init__(self, width=100):
        self.width = width
        self.writes = []
        self.frames = []

    def erase(self):
        self.writes.clear()

    def getmaxyx(self):
        return 28, self.width

    def addnstr(self, y, x, text, n, attr=0):
        self.writes.append((y, x, text[:n], attr))

    def refresh(self):
        self.frames.append(list(self.writes))

    def nodelay(self, enabled):
        pass


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "anomx.agent.file_reference_search.FileReferenceSearchClient.start", lambda _self: None,
    )
    home = AnomxHome(tmp_path / "home")
    home.save_config({"provider": "desy", "model": "coding"})
    record = home.save_project(tmp_path, "Signal Lab")
    sessions = [home.create_session(tmp_path, provider="desy", model="coding") for _ in range(2)]
    app = AnomxCliApp(home=home, cwd=tmp_path, use_color=False)
    app._colors.update(
        light=curses.A_DIM, cursor=curses.A_REVERSE, accent=curses.A_BOLD, bold=curses.A_BOLD,
    )
    return app, record, sessions


def dots(frame):
    return [(y, x, attr) for y, x, text, attr in frame if text == "●"]


@pytest.mark.parametrize("typed", ["a", PromptPasteEvent("pasted text")])
def test_project_keyboard_focus_moves_to_list_then_back_to_prompt(project, monkeypatch, typed):
    app, record, sessions = project
    window = Window()
    keys = iter([
        "\n", curses.KEY_UP, curses.KEY_DOWN, curses.KEY_DOWN,
        curses.KEY_UP, typed, "\x03", "\x03", "\x03",
    ])
    monkeypatch.setattr(app, "_read_prompt_key", lambda _screen: next(keys))
    monkeypatch.setattr(app, "_open_project_session", lambda *_: pytest.fail("unexpected open"))
    assert app._run_project(window, record) == 0
    top = app._session_body_top()
    assert dots(window.frames[0]) == []  # Initial focus is the prompt.
    assert dots(window.frames[1]) == []  # Empty Enter does not open a chat.
    assert dots(window.frames[2]) == [(top, 2, 0)]
    assert dots(window.frames[3]) == [(top + 1, 2, 0)]
    assert dots(window.frames[4]) == []  # Down past the final chat returns to the prompt.
    assert dots(window.frames[5]) == [(top, 2, 0)]
    assert dots(window.frames[6]) == []
    text = typed.text if isinstance(typed, PromptPasteEvent) else typed
    assert any(value == text for _, _, value, _ in window.frames[6])
    for frame in window.frames:
        prompt_cursor = [
            row for row in frame
            if row[0] == app._prompt_layout(window).prompt_line and row[3] == curses.A_REVERSE
        ]
        assert bool(prompt_cursor) is not bool(dots(frame))


def test_enter_opens_focused_chat_and_delete_requires_list_focus(project, monkeypatch):
    app, record, sessions = project
    keys = iter(["\x04", "\n", curses.KEY_UP, "\n", "\x03", "\x03"])
    opened = []
    monkeypatch.setattr(app, "_read_prompt_key", lambda _screen: next(keys))
    monkeypatch.setattr(app, "_open_project_session", lambda _, session: opened.append(session))
    monkeypatch.setattr(app, "_delete_project_session", lambda *_: pytest.fail("unexpected delete"))
    assert app._run_project(Window(), record) == 0
    assert opened == [app._project_sessions(record.path)[0]]


def test_project_title_wave_and_scroll_do_not_depend_on_keyboard_focus(project, monkeypatch):
    app, record, sessions = project
    monkeypatch.setattr(app, "_session_is_running", lambda _: True)
    waves = []
    monkeypatch.setattr(app, "_draw_activity_wave", lambda *args, **kw: waves.append(args[3]))
    window = Window()
    scroll = app._draw_project(window, record, sessions, 0, 1, "", 0, "", "light", 0)
    assert scroll == 1
    assert waves == [sessions[1].title]
    assert app._project_animation_active(sessions)
    assert not dots(window.writes)


@pytest.mark.parametrize(
    "selected,running", [(False, False), (True, False), (False, True), (True, True)],
)
def test_selected_and_running_titles_use_primary_text(project, monkeypatch, selected, running):
    app, _, sessions = project
    monkeypatch.setattr(app, "_session_is_running", lambda _: running)
    window = Window()
    app._draw_project_session_row(
        window, 6, 100, sessions[0], selected=selected, index=0, body_bottom=20,
    )
    title = next(row for row in window.writes if row[1] == 4 and row[2] == sessions[0].title)
    assert title[3] == (curses.A_NORMAL if selected or running else curses.A_DIM)


@pytest.mark.parametrize("width", [32, 80, 160])
def test_header_uses_model_name_and_gray_divider_without_box_or_tagline(project, width):
    app, record, sessions = project
    window = Window(width)
    app._draw_project(window, record, sessions, 0, 0, "", 0, "", "light", 0)
    text = "\n".join(value for _, _, value, _ in window.writes)
    if width >= 80:
        assert "DeepSeek V4" in text
    assert "desy/coding" not in text
    assert "Data Analysis and Anomaly Detection Agent" not in text
    assert not any(char in text for char in "╭╮╰╯│")
    assert (4, 2, "─" * (width - 4), curses.A_DIM) in window.writes
    assert all(0 <= x < width for _, x, _, _ in window.writes)


@pytest.mark.parametrize("width", [32, 80, 160])
@pytest.mark.parametrize("raw_mouse", [False, True])
def test_context_click_switches_only_the_unit_and_persists(project, width, raw_mouse, monkeypatch):
    app, _, sessions = project
    session = sessions[0]
    app.home.append_session_event(session.path, "user_message", {"message": "Inspect"})
    app._session_usage_snapshots[session.path] = SimpleNamespace(context_tokens=5700)
    window = Window(width)
    for before, after in (("2%", "5.7k"), ("5.7k", "2%")):
        app._draw_session(window, session, [], "", 0, 0)
        assert app._context_status(session, "coding") == before
        targets = [
            (y, action) for y, actions in app._click_targets.items() for action in actions
            if action.kind == "toggle_context_unit"
        ]
        assert len(targets) == 1
        y, action = targets[0]
        assert action.x_end - action.x_start == len(before)
        assert app._click_target_action_at(action.x_end, y, curses.BUTTON1_PRESSED) is None
        if raw_mouse:
            result = app._raw_mouse_action(
                f"\x1b[<0;{action.x_start + 1};{y + 1}M", window, "", [], 0,
            )
        else:
            monkeypatch.setattr(
                curses, "getmouse",
                lambda target=action, row=y: (0, target.x_start, row, 0, curses.BUTTON1_PRESSED),
            )
            result = app._session_mouse_action(window, "", [], 0)
        assert result.kind == "toggle_context_unit"
        assert app._context_status(session, "coding") == after
        restored = AnomxCliApp(home=app.home, cwd=app.cwd, use_color=False)
        assert restored.home.load_config()["context_unit"] == (
            "tokens" if after.endswith("k") else "percent"
        )


def test_design_settings_change_context_unit_and_cancel_preserves_it(project, monkeypatch):
    app, _, _ = project
    answers = iter(["context_unit", "tokens", "context_unit", None, None])
    monkeypatch.setattr(app, "_menu", lambda *args, **kwargs: next(answers))
    app._run_manage_settings_panel(Window())
    assert app.home.load_config()["context_unit"] == "tokens"
    choices = app._manage_settings_choices()
    assert choices[0].label == "Design" and not choices[0].selectable
    assert choices[1].label == "Visualization: Default"
    assert choices[2].label == "Context Unit: k"


def test_invalid_or_missing_context_unit_defaults_to_percent(project):
    app, _, _ = project
    app.home.save_config({"context_unit": "invalid"})
    assert app.home.load_config()["context_unit"] == "percent"
    app.home.config_path.write_text('context_unit = "invalid"\n')
    assert app.home.load_config()["context_unit"] == "percent"
    app.home.config_path.write_text("")
    assert app.home.load_config()["context_unit"] == "percent"


def test_model_name_tracks_current_selection_with_custom_fallback(project):
    app, record, sessions = project
    app.home.save_config({"provider": "desy", "model": "desy-assistant"})
    window = Window()
    app._draw_project(window, record, sessions, 0, 0, "", 0, "", "light", 0)
    assert any("Mistral Medium 3.5" in text for _, _, text, _ in window.writes)
    assert "Mistral Medium 3.5" in app._session_header_meta(sessions[0], "desy", "desy-assistant")
    assert app._model_header_label("ollama", "my-local-model") == "my-local-model"


@pytest.mark.parametrize("system", ["host", "python", "docker", "podman"])
@pytest.mark.parametrize("view", ["project", "session"])
def test_two_line_header_keeps_path_and_shows_sandbox_by_brand(project, system, view):
    app, record, sessions = project
    app.home.save_config({
        **app.home.load_config(), "sandbox_enabled": system != "host", "sandbox_system": system,
    })
    window = Window(200)
    if view == "project":
        app._draw_project(window, record, sessions, 0, 0, "", 0, "", "light", 0)
    else:
        app._draw_session(window, sessions[0], [], "", 0, 0)
    path = str(record.path)
    assert (3, 200 - len(path) - 5, path, curses.A_BOLD) in window.writes
    assert (4, 2, "─" * 196, curses.A_DIM) in window.writes
    assert not any(text in {"Back to Project", "esc", "Secure Agent Runtime"}
                   for _, _, text, _ in window.writes)
    sandbox = [(y, x) for y, x, text, _ in window.writes if text == "Sandbox"]
    assert sandbox == ([(2, 12)] if system != "host" else [])
    assert app._session_body_top() == 6


@pytest.mark.parametrize("width", [32, 80, 160])
def test_long_path_does_not_overlap_title_or_its_click_target(project, width):
    app, _, sessions = project
    session = replace(sessions[0], cwd="/very/long/launch/location/" * 10 + "project")
    window = Window(width)
    app._draw_session(window, session, [], "", 0, 0)
    path_write = next(row for row in window.writes if row[0] == 3 and row[2].endswith("project"))
    y, x, text, attr = path_write
    assert text.startswith("…")
    assert attr == curses.A_BOLD
    assert x + len(text) == width - 5
    title_write = next(row for row in window.writes if row[0] == 3 and row[1] == 4)
    assert 4 + len(title_write[2]) < x
    action = app._click_target_action_at(x, y, curses.BUTTON1_PRESSED)
    assert action.kind == "back_project"
    assert action.x_end == x + len(text)


def test_project_titles_align_with_brand_and_cursor_is_in_left_margin(project):
    app, record, sessions = project
    window = Window()
    app._draw_project(
        window, record, sessions, 0, 0, "", 0, "", "light", 0, sessions_focused=True,
    )
    top = app._session_body_top()
    assert dots(window.writes) == [(top, 2, 0)]
    assert all(x == 4 for y, x, text, _ in window.writes
               if y >= top and text == sessions[0].title)
    action = app._click_target_action_at(2, top, curses.BUTTON1_PRESSED)
    assert action.kind == "open_project_session"


@pytest.mark.parametrize("raw_mouse", [False, True])
def test_subagent_location_click_returns_to_parent_session(project, monkeypatch, raw_mouse):
    app, _, sessions = project
    reads = []

    def read_key(_screen):
        reads.append(True)
        assert len(reads) == 1, "Clicking the path must close the subagent view"
        action = next(a for a in app._click_targets[3] if a.kind == "back_project")
        monkeypatch.setattr(
            curses, "getmouse", lambda: (0, action.x_start, 3, 0, curses.BUTTON1_PRESSED),
        )
        return f"\x1b[<0;{action.x_start + 1};4M" if raw_mouse else curses.KEY_MOUSE

    monkeypatch.setattr(app, "_read_nonblocking_key", read_key)
    app._open_subagent_session(Window(), sessions[0], "starting-agent")
    assert len(reads) == 1
