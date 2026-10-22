from __future__ import annotations

import curses
import io
import json
import threading
import time
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from anomx.agent import startup
from anomx.agent.app import AnomxCliApp
from anomx.agent.store import AnomxHome
from anomx.agent.ui.views import approve_update


@pytest.fixture
def app(tmp_path):
    instance = AnomxCliApp(home=AnomxHome(tmp_path / "home"), isolate_runtime=True)
    yield instance
    instance.runtime.close()


@pytest.mark.parametrize(
    "latest, expected", [("0.2.87", "0.2.87"), ("0.2.86", None), ("0.2.85", None)]
)
def test_version_check_returns_only_newer_releases(monkeypatch, latest, expected):
    monkeypatch.setattr(
        startup,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(json.dumps({"info": {"version": latest}}).encode()),
    )
    assert startup.check_for_update("0.2.86") == startup.UpdateCheck(True, expected)


def test_offline_check_is_not_reported_as_up_to_date(monkeypatch):
    def offline(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr(startup, "urlopen", offline)
    assert startup.check_for_update("0.2.86") == startup.UpdateCheck(False)


def test_version_check_has_a_wall_clock_deadline_even_when_dns_stalls(monkeypatch):
    release = threading.Event()

    def blocked(*args, **kwargs):
        release.wait(2)
        raise URLError("offline")

    monkeypatch.setattr(startup, "urlopen", blocked)
    monkeypatch.setattr(startup, "VERSION_CHECK_TIMEOUT_SECONDS", 0.02)
    started = time.monotonic()
    try:
        assert startup.check_for_update("0.2.86") == startup.UpdateCheck(False)
        assert time.monotonic() - started < 0.5
    finally:
        release.set()


def test_startup_prompts_once_when_background_result_arrives(app, monkeypatch):
    app._startup_tasks = SimpleNamespace(version_ready=False, latest_version=None)
    calls = []
    monkeypatch.setattr(app, "_run_version_check", lambda screen: calls.append(screen))
    assert app._startup_update_pending()
    app._maybe_prompt_startup_update(None, allow_prompt=True)
    assert not calls
    app._startup_tasks.version_ready = True
    app._startup_tasks.latest_version = "9.9.9"
    app._maybe_prompt_startup_update(None, allow_prompt=False)
    assert not calls
    app._maybe_prompt_startup_update(None, allow_prompt=True)
    app._maybe_prompt_startup_update(None, allow_prompt=True)
    assert calls == [None]
    assert not app._startup_update_pending()


def test_startup_skips_failed_check_without_prompt(app, monkeypatch):
    app._startup_tasks = SimpleNamespace(version_ready=True, latest_version=None)
    monkeypatch.setattr(app, "_run_version_check", lambda screen: pytest.fail("unexpected prompt"))
    app._maybe_prompt_startup_update(None, allow_prompt=True)
    assert app._startup_version_checked


def test_update_later_is_offered_again_on_next_launch_despite_old_skip_setting(app, monkeypatch):
    app._startup_tasks = SimpleNamespace(version_ready=True, latest_version="9.9.9")
    app.home.save_config({**app.home.load_config(), "skipped_version": "9.9.9"})
    draws = []
    keys = iter([curses.KEY_DOWN, "\n"])
    window = SimpleNamespace(nodelay=lambda value: None, get_wch=lambda: next(keys))
    monkeypatch.setattr(
        app,
        "_draw_version_check",
        lambda screen, selected, latest: draws.append((selected, latest)),
    )
    monkeypatch.setattr(app, "_install_update", lambda *args: pytest.fail("unexpected install"))
    app._maybe_prompt_startup_update(window, allow_prompt=True)
    assert draws == [(0, "9.9.9"), (1, "9.9.9")]


@pytest.mark.parametrize(
    "result, expected_message, expected_install",
    [
        (startup.UpdateCheck(True, "9.9.9"), None, "9.9.9"),
        (startup.UpdateCheck(True), "No updates available", None),
        (startup.UpdateCheck(False), "Could not check for updates", None),
    ],
)
def test_manual_update_is_fresh_and_reports_outcome(
    app, monkeypatch, result, expected_message, expected_install
):
    messages, installs = [], []
    app._startup_tasks = SimpleNamespace(version_ready=True, latest_version="old-cached-version")
    monkeypatch.setattr(approve_update, "check_for_update", lambda version: result)
    monkeypatch.setattr(app, "_draw_overlay", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "_message", lambda screen, title, message: messages.append(message))
    monkeypatch.setattr(app, "_install_update", lambda screen, latest: installs.append(latest))
    app._run_manual_update(None)
    assert installs == ([expected_install] if expected_install else [])
    assert (expected_message in messages[0]) if expected_message else not messages


def test_update_is_available_in_project_and_session_without_creating_chat(app, monkeypatch):
    calls = []
    monkeypatch.setattr(app, "_run_manual_update", lambda screen: calls.append(screen))
    monkeypatch.setattr(app, "_create_session", lambda: pytest.fail("unexpected new chat"))
    assert "/update" in {spec.command for spec in app._project_command_specs()}
    assert "/update" in {spec.command for spec in app._command_specs()}
    app._handle_project_command(None, "/update", "/update", None, [], 0)
    app._handle_command(None, "/update", None)
    assert calls == [None, None]


def test_update_does_not_restart_running_background_sessions(app, monkeypatch):
    app._active_session_turns = {
        "test": SimpleNamespace(worker=SimpleNamespace(is_alive=lambda: True))
    }
    messages = []
    monkeypatch.setattr(app, "_message", lambda screen, title, message: messages.append(message))
    monkeypatch.setattr(
        approve_update, "check_for_update", lambda version: pytest.fail("unexpected request")
    )
    app._run_manual_update(None)
    assert "running sessions" in messages[0]


def test_installer_pins_checked_version_and_restarts_same_cli_arguments(app, monkeypatch):
    calls, restarts = [], []
    window = SimpleNamespace(erase=lambda: None, refresh=lambda: None)
    monkeypatch.setattr(approve_update.curses, "endwin", lambda: None)
    monkeypatch.setattr(approve_update.subprocess, "check_call", lambda argv: calls.append(argv))
    monkeypatch.setattr(
        approve_update.os,
        "execve",
        lambda executable, argv, env: restarts.append((executable, argv)),
    )
    monkeypatch.setattr(
        approve_update.sys, "argv", ["anomx", "--home", "/tmp/example-home", "--no-color"]
    )
    app._install_update(window, "9.9.9")
    assert calls == [
        [approve_update.sys.executable, "-m", "pip", "install", "--upgrade", "anomx==9.9.9"]
    ]
    assert restarts == [
        (
            approve_update.sys.executable,
            [
                approve_update.sys.executable,
                "-m",
                "anomx",
                "--home",
                "/tmp/example-home",
                "--no-color",
            ],
        )
    ]
