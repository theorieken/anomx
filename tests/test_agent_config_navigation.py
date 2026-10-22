"""Config hierarchy, inline settings, and navigation back from child pages."""

import curses

import pytest

from anomx.agent.app import AnomxCliApp
from anomx.agent.store import AnomxHome
from anomx.agent.ui.models import MenuChoice


class Window:
    def __init__(self, keys=(), width=160, height=40):
        self.keys = iter(keys)
        self.width, self.height = width, height
        self.writes, self.frames = [], []

    def getmaxyx(self):
        return self.height, self.width

    def erase(self):
        self.writes.clear()

    def addnstr(self, y, x, text, n, attr=0):
        self.writes.append((y, x, text[:n], attr))

    def refresh(self):
        self.frames.append(list(self.writes))

    def get_wch(self):
        return next(self.keys)


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "anomx.agent.file_reference_search.FileReferenceSearchClient.start", lambda _self: None,
    )
    return AnomxCliApp(home=AnomxHome(tmp_path / "home"), cwd=tmp_path, use_color=False)


@pytest.mark.parametrize("width,height", [(160, 40), (80, 24)])
def test_commands_is_a_settings_child_and_escape_restores_selection(app, width, height):
    config_values = [c.value for c in app._config_menu_choices() if c.selectable]
    settings_values = [c.value for c in app._manage_settings_choices() if c.selectable]
    window = Window([
        *([curses.KEY_DOWN] * config_values.index("settings")), "\n",
        *([curses.KEY_DOWN] * settings_values.index("commands")), "\n",
        "\x1b", "\x1b", "\x1b",
    ], width, height)
    app._run_config_panel(window, None)

    for frame in window.frames:
        assert any(y == 4 and text.startswith("─") for y, _, text, _ in frame)
        assert not any(y == 4 and text.strip() and not text.startswith("─")
                       for y, _, text, _ in frame)
    command_frame = window.frames[-3]
    assert (3, 4, "Config › Manage Settings › Manage Commands", 0) in command_frame
    assert any(y > 4 and text == "No commands saved yet" for y, _, text, _ in command_frame)
    for frame, selected in [(window.frames[-2], "Manage Commands"),
                            (window.frames[-1], "Manage Settings")]:
        selected_y = next(y for y, _, text, _ in frame if text == "›")
        assert any(y == selected_y and text == selected for y, _, text, _ in frame)
    assert app._config_breadcrumbs == ()


def test_inline_sandbox_and_debug_controls_persist_and_show_their_section(app, monkeypatch):
    answers = iter([
        "settings", "debug_mode", "full_session_logs", "full_session_logs_path",
        "sandbox_system", "python", "sandbox_cpu", "sandbox_ram", "sandbox_hd", None, None,
    ])
    headers = []

    def menu(_screen, title, *_args, **_kwargs):
        headers.append(app._config_header_title(title, 160))
        return next(answers)

    def text(_screen, title, *_args, **_kwargs):
        headers.append(app._config_header_title(title, 160))
        return {"Debug Location": str(app.home.root / "logs"),
                "CPU Limit": "3", "RAM Limit": "8g", "Disk Limit": "20g"}[title]

    monkeypatch.setattr(app, "_menu", menu)
    monkeypatch.setattr(app, "_prompt_text", text)
    monkeypatch.setattr(app, "_run_overlay_text", text)
    app._run_config_panel(Window(), None)
    config = app.home.load_config()
    assert config["debug_mode"] and config["debug_full_session_logs"]
    assert config["debug_full_session_logs_path"] == str(app.home.root / "logs")
    assert config["sandbox_system"] == "python"
    limits = [config[f"sandbox_{key}_limit"] for key in ("cpu", "ram", "hd")]
    assert limits == ["3", "8g", "20g"]
    assert "Config › Manage Settings › Debug Mode › Debug Location" in headers
    assert "Config › Manage Settings › Sandbox › CPU Limit" in headers
    assert "Config › Manage Settings › Sandbox › Sandbox Runtime" in headers
    assert app._config_breadcrumbs == ()


def test_cancelled_sandbox_disable_keeps_config_and_containers(app, monkeypatch):
    app.home.save_config({"sandbox_enabled": True})
    answers = iter(["settings", "sandbox_toggle", "cancel", "sandbox_cpu", None, None, None])
    monkeypatch.setattr(app, "_menu", lambda *args, **kwargs: next(answers))
    monkeypatch.setattr(app, "_run_overlay_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        app, "_remove_all_sandbox_containers",
        lambda: pytest.fail("Cancelled disable removed runtime"),
    )
    original = app.home.load_config()
    app._run_config_panel(Window(), None)
    assert app.home.load_config() == original
    assert app._config_breadcrumbs == ()


def test_confirmation_dialog_does_not_remember_a_previous_confirm_selection(app):
    choices = (MenuChoice("Cancel", "cancel"), MenuChoice("Disable", "confirm"))
    with app._config_page("Config"), app._config_page("Manage Settings"):
        assert app._menu(
            Window([curses.KEY_DOWN, "\n"]), "Disable Sandbox", "", choices,
        ) == "confirm"
        assert app._menu(Window(["\n"]), "Disable Sandbox", "", choices) == "cancel"


@pytest.mark.parametrize("width", [32, 80, 160])
def test_nested_config_headers_keep_current_page_visible_and_help_below_divider(app, width):
    window = Window(width=width)
    with (
        app._config_page("Config"),
        app._config_page("Manage Settings"),
        app._config_page("Sandbox"),
    ):
        app._draw_overlay(
            window, "CPU Limit", "Number of CPU cores", input_value="2", show_input_cursor=True,
        )
        title = next(text for y, x, text, _ in window.writes if y == 3 and x == 4)
        assert title.startswith("Config › ") and title.endswith("CPU Limit")
        assert len(title) <= width - 9
        assert any(y > 4 and text == "Number of CPU cores" for y, _, text, _ in window.writes)
    assert app._config_breadcrumbs == ()
    # A config breadcrumb must not leak into a later, unrelated overlay.
    app._draw_overlay(window, "Unrelated", choices=(MenuChoice("Item", "item"),))
    assert (3, 4, "Unrelated", 0) in window.writes
