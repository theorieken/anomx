import curses
import json
import queue
import stat
import sys
import tomllib
from pathlib import Path
from urllib.error import HTTPError

import anomx.agent.platform_client as platform_client_module
import anomx.agent.runtime as runtime_module
import anomx.agent.tool_manager as tool_manager_module
from anomx import __version__
from anomx.agent import AnomxHome
from anomx.agent.mode import AgentMode
from anomx.agent.platform_client import (
    connect_platform,
    heartbeat_platform_connection,
    normalize_platform_url,
    resolve_platform_api_url,
)
from anomx.agent.runtime import (
    AgentRole,
    AgentRuntime,
    QuestionRequest,
    QuestionResponse,
    RuntimeCallbacks,
)
from anomx.agent.skills import Skill, load_builtin_skills, load_user_skills, write_user_skill
from anomx.agent.state import (
    PlanStep,
    event_payload,
    event_payload_type,
    latest_plan_steps,
    running_process_snapshots,
    running_worker_snapshots,
    worker_snapshots,
)
from anomx.agent.store import (
    AI_PROVIDER_KEYS,
    model_context_window,
    model_detail,
    provider_by_key,
    resolve_anomx_home,
)
from anomx.agent.terminal import (
    markdown_to_terminal_lines,
    markdown_to_terminal_rendered_lines,
)
from anomx.agent.tool_manager import (
    ApprovalChoice,
    CliToolManager,
    CommandApprovalRequest,
    CommandSafety,
    discover_workspace_root,
)
from anomx.agent.ui import (
    MANUAL_INTERRUPT_MESSAGE,
    AgentState,
    AnomxCliApp,
    BackendTurnResult,
    BottomPanel,
    InfoRow,
    MenuChoice,
    MessageLine,
    PlatformConnectionDraft,
    RuntimeUiEvent,
    SkillFormDraft,
    SessionMouseAction,
)
from anomx.cli import _startup_model, _startup_provider


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_anomx_home_uses_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMX_HOME", str(tmp_path / "home"))

    assert resolve_anomx_home() == tmp_path / "home"


def test_trusted_repo_round_trips(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()

    assert not home.is_repo_trusted(repo)

    home.trust_repo(repo)

    assert home.is_repo_trusted(repo)
    trusted = tomllib.loads(home.config_path.read_text(encoding="utf-8"))
    assert trusted["projects"][str(repo.resolve())]["trust_level"] == "trusted"


def test_require_trusted_repo_config_is_always_true(tmp_path):
    home = AnomxHome(tmp_path / "home")
    home.ensure()
    home.config_path.write_text("require_trusted_repo = false\n", encoding="utf-8")

    config = home.load_config()
    assert config["require_trusted_repo"] is True

    config["require_trusted_repo"] = False

    home.save_config(config)

    assert home.load_config()["require_trusted_repo"] is True
    stored_config = tomllib.loads(home.config_path.read_text(encoding="utf-8"))
    assert "require_trusted_repo" not in stored_config


def test_agent_mode_config_defaults_and_normalizes(tmp_path):
    home = AnomxHome(tmp_path / "home")
    assert home.load_config()["agent_mode"] == AgentMode.CONFIRM.value

    home.ensure()
    home.config_path.write_text('agent_mode = "invalid"\n', encoding="utf-8")

    assert home.load_config()["agent_mode"] == AgentMode.CONFIRM.value


def test_discover_workspace_root_prefers_vcs_root(tmp_path):
    repo = tmp_path / "repo"
    nested = repo / "src" / "anomx" / "agent"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()
    (nested / "pyproject.toml").write_text("[project]\nname = 'nested'\n", encoding="utf-8")

    assert discover_workspace_root(nested) == repo


def test_session_storage_writes_metadata_and_events(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()

    session = home.create_session(repo, provider="openai", model="gpt-5")
    home.append_session_event(session.path, "user_message", {"message": "inspect the data"})

    events = _read_jsonl(session.path)
    assert session.path.name.startswith("rollout-")
    assert events[0]["type"] == "session_meta"
    assert events[0]["payload"]["model_provider"] == "openai"
    assert events[1]["type"] == "event_msg"
    assert events[1]["payload"]["type"] == "user_message"
    assert home.load_config()["last_session_id"] == session.session_id
    assert home.list_sessions()[0].title == "inspect the data"
    assert _read_jsonl(home.session_index_path)[0]["payload"]["session_id"] == session.session_id


def test_session_title_can_be_updated(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")

    home.update_session_title(session.path, "Repository Anomaly Scan")

    assert home.list_sessions()[0].title == "Repository Anomaly Scan"


def test_clear_sessions_keeps_current_session_and_resets_index(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    keep_session = home.create_session(repo, provider="openai", model="gpt-5.5")
    drop_session = home.create_session(repo, provider="openai", model="gpt-5.5")

    home.clear_sessions(keep_session_path=keep_session.path)

    sessions = home.list_sessions(limit=None)
    assert [session.session_id for session in sessions] == [keep_session.session_id]
    assert keep_session.path.exists()
    assert not drop_session.path.exists()
    assert home.load_config()["last_session_id"] == keep_session.session_id
    assert (
        _read_jsonl(home.session_index_path)[0]["payload"]["session_id"]
        == keep_session.session_id
    )


def test_model_metadata_tracks_context_windows():
    assert model_context_window("gpt-5.5") == 1_000_000
    assert model_context_window("gpt-5.4-mini") == 400_000
    assert model_context_window("claude-haiku-4-5-20251001") == 200_000
    assert model_detail("claude-sonnet-4-6") == "1M context · 64K max output"


def test_provider_catalog_includes_desy_assistant():
    assert AI_PROVIDER_KEYS == ("openai", "anthropic", "desy", "ollama")


def test_api_key_is_written_to_owner_only_auth_file(tmp_path):
    home = AnomxHome(tmp_path / "home")

    home.set_api_key("openai", "test-key")

    assert home.has_api_key("openai")
    mode = stat.S_IMODE(home.auth_path.stat().st_mode)
    assert mode == 0o600


def test_platform_connection_is_written_to_owner_only_auth_file(tmp_path):
    home = AnomxHome(tmp_path / "home")
    home.set_platform_form_defaults(
        url="https://anomalies.msktools.desy.de",
        email="ada@example.com",
    )

    home.set_platform_connection(
        url="http://localhost:8000",
        token="platform-token",
        user_email="ada@example.com",
        organization_url="analytical-engines",
        hostname="edge-node-01",
    )

    assert home.load_config()["platform_url"] == "http://localhost:8000"
    assert home.platform_connection() == {
        "url": "http://localhost:8000",
        "token": "platform-token",
        "user_email": "ada@example.com",
        "organization_url": "analytical-engines",
        "hostname": "edge-node-01",
    }
    assert stat.S_IMODE(home.auth_path.stat().st_mode) == 0o600

    home.clear_platform_connection()

    assert home.load_config()["platform_url"] is None
    assert home.load_config()["platform_last_url"] == "https://anomalies.msktools.desy.de"
    assert home.load_config()["platform_last_email"] == "ada@example.com"
    assert home.platform_connection() is None


def test_normalize_platform_url_defaults_localhost_to_http():
    assert normalize_platform_url("localhost:8000/") == "http://localhost:8000"
    assert normalize_platform_url("platform.anomx.ai/") == "https://platform.anomx.ai"


def test_resolve_platform_api_url_falls_back_to_api_prefix(monkeypatch):
    calls: list[str] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"allow_user_registration": True}).encode()

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if request.full_url == "https://anomalies.msktools.desy.de/auth/registration":
            raise HTTPError(request.full_url, 404, "Not Found", hdrs=None, fp=None)
        return FakeResponse()

    monkeypatch.setattr(platform_client_module, "urlopen", fake_urlopen)

    assert (
        resolve_platform_api_url("https://anomalies.msktools.desy.de/")
        == "https://anomalies.msktools.desy.de/api"
    )
    assert calls == [
        "https://anomalies.msktools.desy.de/auth/registration",
        "https://anomalies.msktools.desy.de/api/auth/registration",
    ]


def test_connect_platform_uses_cli_agent_login_payload(monkeypatch):
    calls: list[dict[str, object]] = []

    class FakeResponse:
        def __init__(self, payload: dict[str, object]):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    def fake_urlopen(request, timeout):
        calls.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "timeout": timeout,
                "payload": json.loads(request.data.decode()) if request.data else None,
            }
        )
        if request.full_url == "https://anomalies.msktools.desy.de/auth/registration":
            raise HTTPError(request.full_url, 404, "Not Found", hdrs=None, fp=None)
        if request.full_url == "https://anomalies.msktools.desy.de/api/auth/registration":
            return FakeResponse({"allow_user_registration": True})
        return FakeResponse(
            {
                "token": "platform-token",
                "user": {
                    "email": "ada@example.com",
                    "organization": {"url": "analytical-engines"},
                },
            }
        )

    monkeypatch.setattr(platform_client_module, "local_hostname", lambda: "edge-node-01")
    monkeypatch.setattr(platform_client_module, "urlopen", fake_urlopen)

    result = connect_platform(
        "https://anomalies.msktools.desy.de/",
        "ada@example.com",
        "correcthorse",
    )

    assert result.url == "https://anomalies.msktools.desy.de/api"
    assert result.token == "platform-token"
    assert result.user_email == "ada@example.com"
    assert result.organization_url == "analytical-engines"
    assert result.hostname == "edge-node-01"
    assert calls[-1]["url"] == "https://anomalies.msktools.desy.de/api/auth/login"
    assert calls[-1]["method"] == "POST"
    assert calls[-1]["timeout"] == platform_client_module.DEFAULT_TIMEOUT_SECONDS
    assert calls[-1]["payload"] == {
        "email": "ada@example.com",
        "password": "correcthorse",
        "client": "cli_agent",
        "client_hostname": "edge-node-01",
        "client_version": platform_client_module.__version__,
    }


def test_heartbeat_platform_connection_uses_saved_bearer_token(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    home.set_platform_connection(
        url="http://localhost:8000",
        token="platform-token",
        user_email="ada@example.com",
        organization_url="analytical-engines",
        hostname="edge-node-01",
    )
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["authorization"] = request.get_header("Authorization")
        captured["payload"] = json.loads(request.data.decode())
        return FakeResponse()

    monkeypatch.setattr(platform_client_module, "local_hostname", lambda: "edge-node-02")
    monkeypatch.setattr(platform_client_module, "urlopen", fake_urlopen)

    assert heartbeat_platform_connection(home) is True
    assert captured["url"] == "http://localhost:8000/auth/me/agent/heartbeat"
    assert captured["authorization"] == "Bearer platform-token"
    assert captured["payload"] == {
        "client_hostname": "edge-node-02",
        "client_version": platform_client_module.__version__,
    }


def test_heartbeat_platform_connection_repairs_root_frontend_url(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    home.set_platform_connection(
        url="https://anomalies.msktools.desy.de",
        token="platform-token",
        user_email="ada@example.com",
        organization_url="analytical-engines",
        hostname="edge-node-01",
    )
    calls: list[str] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if request.full_url == "https://anomalies.msktools.desy.de/auth/me/agent/heartbeat":
            raise HTTPError(request.full_url, 404, "Not Found", hdrs=None, fp=None)
        return FakeResponse()

    monkeypatch.setattr(platform_client_module, "local_hostname", lambda: "edge-node-02")
    monkeypatch.setattr(platform_client_module, "urlopen", fake_urlopen)

    assert heartbeat_platform_connection(home) is True
    assert calls == [
        "https://anomalies.msktools.desy.de/auth/me/agent/heartbeat",
        "https://anomalies.msktools.desy.de/api/auth/me/agent/heartbeat",
    ]
    assert home.platform_connection()["url"] == "https://anomalies.msktools.desy.de/api"


def test_heartbeat_platform_connection_falls_back_to_profile_touch(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    home.set_platform_connection(
        url="https://anomalies.msktools.desy.de",
        token="platform-token",
        user_email="ada@example.com",
        organization_url="analytical-engines",
        hostname="edge-node-01",
    )
    calls: list[tuple[str, str]] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(request, timeout):
        calls.append((request.get_method(), request.full_url))
        if request.full_url in {
            "https://anomalies.msktools.desy.de/auth/me/agent/heartbeat",
            "https://anomalies.msktools.desy.de/api/auth/me/agent/heartbeat",
        }:
            raise HTTPError(request.full_url, 404, "Not Found", hdrs=None, fp=None)
        return FakeResponse()

    monkeypatch.setattr(platform_client_module, "local_hostname", lambda: "edge-node-02")
    monkeypatch.setattr(platform_client_module, "urlopen", fake_urlopen)

    assert heartbeat_platform_connection(home) is True
    assert calls == [
        ("POST", "https://anomalies.msktools.desy.de/auth/me/agent/heartbeat"),
        ("POST", "https://anomalies.msktools.desy.de/api/auth/me/agent/heartbeat"),
        ("GET", "https://anomalies.msktools.desy.de/api/auth/me"),
    ]
    assert home.platform_connection()["url"] == "https://anomalies.msktools.desy.de/api"


def test_user_skill_storage_round_trips_global_home(tmp_path):
    home = AnomxHome(tmp_path / "home")
    skill = Skill(
        command="profile-data",
        title="Profile data",
        description="Inspect dataset shape and quality.",
        body="Profile the current dataset and report useful statistics.",
        source="user",
    )

    path = write_user_skill(home.skills_dir, skill)

    assert path == home.skills_dir / "profile-data.md"
    assert load_user_skills(home.skills_dir) == (
        Skill(
            command="profile-data",
            title="profile-data",
            description="Inspect dataset shape and quality.",
            body="Profile the current dataset and report useful statistics.",
            source="user",
            path=path,
        ),
    )


def test_bundled_starter_skills_are_hidden_and_callable():
    skills = load_builtin_skills()

    assert {skill.command for skill in skills} >= {
        "map-folder",
        "find-issues",
        "make-report",
    }
    assert all(skill.hidden for skill in skills)


def test_startup_ollama_configures_local_backend(tmp_path):
    home = AnomxHome(tmp_path / "home")

    AnomxCliApp(
        home=home,
        startup_provider="ollama",
        startup_model="qwen3.6",
    ).prepare_startup_config()

    config = home.load_config()
    assert config["onboarding_complete"] is True
    assert config["provider"] == "ollama"
    assert config["model"] == "qwen3.6"


def test_ollama_env_implies_local_provider(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL_ID", "qwen3.6")

    assert _startup_provider(None, False) == "ollama"
    assert _startup_model(None) == "qwen3.6"


def test_slash_commands_show_skills_on_empty_slash(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    all_commands = app._filtered_commands("/")
    model_commands = app._filtered_commands("/mo")

    assert [command.command for command in all_commands[:7]] == [
        "/new",
        "/open",
        "/skills",
        "/config",
        "/model",
        "/info",
        "/exit",
    ]
    assert {"/map-folder", "/find-issues", "/make-report"}.issubset(
        {command.command for command in all_commands}
    )
    map_folder = next(command for command in all_commands if command.command == "/map-folder")
    assert map_folder.description.startswith("Map the folder · Understand the files")
    assert [command.command for command in model_commands] == ["/model"]
    assert [command.command for command in app._filtered_commands("/in")] == ["/info"]
    assert [command.command for command in app._filtered_commands("/ex")] == ["/exit"]
    assert [command.command for command in app._filtered_commands("/map")] == ["/map-folder"]


def test_user_skill_command_enters_slash_command_menu(tmp_path):
    home = AnomxHome(tmp_path / "home")
    write_user_skill(
        home.skills_dir,
        Skill(
            command="quality-scan",
            title="Quality scan",
            description="Inspect data quality.",
            body="Look for missing values and bad timestamps.",
            source="user",
        ),
    )
    app = AnomxCliApp(home=home)

    assert [command.command for command in app._filtered_commands("/qua")] == ["/quality-scan"]
    assert "/quality-scan" in [command.command for command in app._filtered_commands("/")]


def test_submitted_slash_command_prefers_exact_command(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    suggestions = app._filtered_commands("/")

    assert app._submitted_command("/config", suggestions, selected=0) == "/config"
    assert app._submitted_command("/open", suggestions, selected=0) == "/open"
    assert app._submitted_command("/map-folder data", suggestions, selected=0) == "/map-folder"


def test_info_command_opens_session_info_panel(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo)
    opened = []

    monkeypatch.setattr(app, "_run_info_panel", lambda _stdscr, session: opened.append(session))

    assert app._handle_command(object(), "/info", session) is None
    assert opened == [session]


def test_skills_command_opens_skills_panel(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo)
    opened = []

    monkeypatch.setattr(app, "_run_skills_panel", lambda _stdscr, session: opened.append(session))

    assert app._handle_command(object(), "/skills", session) is None
    assert opened == [session]


def test_skills_menu_lists_create_then_user_skills_only(tmp_path):
    home = AnomxHome(tmp_path / "home")
    write_user_skill(
        home.skills_dir,
        Skill(
            command="quality-scan",
            title="Quality scan",
            description="Inspect data quality.",
            body="Look for missing values and bad timestamps.",
            source="user",
        ),
    )
    app = AnomxCliApp(home=home)

    choices = app._skills_menu_choices()

    assert choices[0] == MenuChoice(
        "Create new Skill",
        "__create_skill__",
        "Define a global slash-command skill",
    )
    assert [(choice.label, choice.value) for choice in choices[1:]] == [
        ("/quality-scan", "quality-scan")
    ]
    assert all(
        choice.value not in {"map-folder", "find-issues", "make-report"}
        for choice in choices
    )


def test_draw_skill_editor_panel_marks_selected_field_in_accent(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 100

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._colors = {"light": 10, "accent": 20}
    window = Window()
    draft = SkillFormDraft(
        command="test",
        description="Testing skill edits",
        body="Do the test.",
        path=tmp_path / "home" / "skills" / "test.md",
    )

    app._draw_skill_editor_panel(
        window,
        "Edit Skill",
        draft,
        selected=1,
    )

    assert any(text == "Command" and attr == 10 for _, _, text, attr in window.writes)
    assert any(text == "/test" and attr == 10 for _, _, text, attr in window.writes)
    assert any(text == "Description" and attr == 20 for _, _, text, attr in window.writes)
    assert any(text == "Testing skill edits" and attr == 0 for _, _, text, attr in window.writes)
    assert any(text.startswith("Stored at:") and attr == 10 for _, _, text, attr in window.writes)
    assert any(text == "Skill" and attr == 10 for _, _, text, attr in window.writes)


def test_draw_skill_detail_panel_keeps_skill_label_light(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 100

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._colors = {"light": 10, "accent": 20}
    window = Window()
    skill = Skill(
        command="test",
        title="test",
        description="Testing skill detail.",
        body="Do the test.",
        source="user",
        path=tmp_path / "home" / "skills" / "test.md",
    )

    app._draw_skill_detail_panel(window, skill)

    assert any(text == "Skill" and attr == 10 for _, _, text, attr in window.writes)
    assert not any(text == "Skill" and attr == 20 for _, _, text, attr in window.writes)


def test_run_skill_editor_saves_create_form(tmp_path):
    class Window:
        def __init__(self):
            self._keys = iter(
                (
                    *"test",
                    curses.KEY_DOWN,
                    *"Test description",
                    curses.KEY_DOWN,
                    *"Do the test.",
                    "\x13",
                )
            )
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 100

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

        def get_wch(self):
            return next(self._keys)

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    window = Window()

    saved = app._run_skill_editor(window, title="Create Skill")

    assert saved is not None
    assert saved.command == "test"
    assert saved.title == "test"
    assert saved.description == "Test description"
    assert saved.body == "Do the test."
    assert (tmp_path / "home" / "skills" / "test.md").exists()
    assert any(
        text == "Esc Cancel · Ctrl+S Save · ↑↓ Navigate · Enter Next"
        for _, _, text, _ in window.writes
    )


def test_run_skill_editor_saves_existing_skill_and_renames_file(tmp_path):
    class Window:
        def __init__(self):
            self._keys = iter(
                (
                    *"renamed",
                    "\x13",
                )
            )
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 100

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

        def get_wch(self):
            return next(self._keys)

    home = AnomxHome(tmp_path / "home")
    old_path = write_user_skill(
        home.skills_dir,
        Skill(
            command="test",
            title="Test skills",
            description="Testing skill edits",
            body="Do the test.",
            source="user",
        ),
    )
    existing = load_user_skills(home.skills_dir)[0]
    app = AnomxCliApp(home=home, use_color=False)

    saved = app._run_skill_editor(Window(), title="Edit Skill", existing_skill=existing)

    assert saved is not None
    assert saved.command == "testrenamed"
    assert saved.path == home.skills_dir / "testrenamed.md"
    assert not old_path.exists()
    assert saved.path.exists()


def test_skill_detail_panel_edit_shortcut_updates_current_skill(tmp_path, monkeypatch):
    class Window:
        def __init__(self):
            self._keys = iter(("\x05", "\n"))
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 100

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

        def get_wch(self):
            return next(self._keys)

    home = AnomxHome(tmp_path / "home")
    path = write_user_skill(
        home.skills_dir,
        Skill(
            command="test",
            title="Test skills",
            description="Testing skill edits",
            body="Do the test.",
            source="user",
        ),
    )
    skill = Skill(
        command="test",
        title="Test skills",
        description="Testing skill edits",
        body="Do the test.",
        source="user",
        path=path,
    )
    edited = Skill(
        command="test",
        title="test",
        description="Edited skill description",
        body="Do the test.",
        source="user",
        path=path,
    )
    app = AnomxCliApp(home=home, use_color=False)
    monkeypatch.setattr(app, "_edit_user_skill", lambda _stdscr, _skill: edited)
    window = Window()

    app._run_skill_detail_panel(window, skill)

    assert any(text == "Edited skill description" for _, _, text, _ in window.writes)


def test_delete_user_skill_removes_file_when_confirmed(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    path = write_user_skill(
        home.skills_dir,
        Skill(
            command="test",
            title="Test skills",
            description="Testing skill edits",
            body="Do the test.",
            source="user",
        ),
    )
    skill = Skill(
        command="test",
        title="Test skills",
        description="Testing skill edits",
        body="Do the test.",
        source="user",
        path=path,
    )
    app = AnomxCliApp(home=home, use_color=False)
    monkeypatch.setattr(app, "_menu", lambda *_args, **_kwargs: "delete")
    monkeypatch.setattr(app, "_message", lambda *_args, **_kwargs: None)

    assert app._delete_user_skill(object(), skill) is True
    assert not path.exists()


def test_skill_invocation_records_prompt_payload(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo)
    skill = app._skill_for_command("/map-folder")
    assert skill is not None

    monkeypatch.setattr(app, "_maybe_start_session_rename", lambda _session: None)
    monkeypatch.setattr(app, "_latest_user_anchor_line", lambda _stdscr, _session: None)
    monkeypatch.setattr(app, "_animate_message_anchor", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        app,
        "_run_backend_turn",
        lambda *_args, **_kwargs: BackendTurnResult("", 0),
    )

    assert app._invoke_skill(object(), session, skill, "/map-folder data/raw") is None

    payload = _read_jsonl(session.path)[-1]["payload"]
    assert payload["type"] == "skill_invocation"
    assert payload["message"] == "/map-folder data/raw"
    assert "Use the Anomx skill /map-folder" in payload["prompt"]
    assert "User arguments:\n\ndata/raw" in payload["prompt"]
    assert AgentRuntime(home, repo).conversation_messages(session.path)[-1] == {
        "role": "user",
        "content": payload["prompt"],
    }
    assert app._read_message_lines(session.path) == [MessageLine("user", "/map-folder data/raw")]


def test_config_menu_shows_only_requested_entries(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    choices = app._config_menu_choices()

    assert [(choice.label, choice.value, choice.detail) for choice in choices] == [
        ("Choose backend", "backend", "Select provider and enter API key"),
        ("Choose model", "model", "Pick the model for the selected backend"),
        (
            "Connect Platform",
            "platform",
            "Send agent activity, results, and findings to Anomx Platform",
        ),
        ("History persistence", "history_persistence", "Store all sessions or none"),
        ("Clear all sessions", "clear_sessions", "Delete stored sessions except this one"),
        ("Done", "done", "Same as Esc"),
    ]


def test_config_menu_shows_manage_platform_when_connected(tmp_path):
    home = AnomxHome(tmp_path / "home")
    home.set_platform_connection(
        url="https://anomalies.msktools.desy.de/api",
        token="platform-token",
        user_email="ada@example.com",
    )
    app = AnomxCliApp(home=home)

    choices = app._config_menu_choices()

    assert choices[2].label == "Manage Platform"
    assert choices[2].value == "platform"
    assert choices[2].detail == "Connected to anomalies.msktools.desy.de"


def test_untrusted_workspace_requires_access_check_when_config_disables_it(
    tmp_path, monkeypatch
):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    config = home.load_config()
    config["onboarding_complete"] = True
    config["require_trusted_repo"] = False
    home.save_config(config)
    app = AnomxCliApp(home=home, cwd=repo)
    access_checks: list[Path] = []

    monkeypatch.setattr(app, "_configure_terminal", lambda _stdscr: None)

    def fail_access_check(_stdscr):
        access_checks.append(app.workspace_root)
        return False

    monkeypatch.setattr(app, "_run_access_check", fail_access_check)

    assert app._run(object()) == 1
    assert access_checks == [repo.resolve()]


def test_configure_backend_requires_api_key_for_hosted_provider(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    app = AnomxCliApp(home=home)
    prompts: list[tuple[str, str]] = []

    monkeypatch.setattr(app, "_select_provider", lambda _stdscr: provider_by_key("openai"))

    def fake_prompt(_stdscr, title, label, mask=False, optional=True):
        prompts.append((title, label))
        assert mask is True
        assert optional is False
        return "test-api-key"

    monkeypatch.setattr(app, "_prompt_text", fake_prompt)

    assert app._configure_backend(object()) is True

    config = home.load_config()
    assert config["provider"] == "openai"
    assert home.has_api_key("openai") is True
    assert prompts == [("OpenAI", "API key")]


def test_configure_backend_provider_change_requires_model_selection(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    app = AnomxCliApp(home=home)
    prompts: list[tuple[str, str]] = []
    selected_models: list[str] = []

    monkeypatch.setattr(app, "_select_provider", lambda _stdscr: provider_by_key("desy"))
    monkeypatch.setattr(
        app,
        "_select_model",
        lambda _stdscr, provider: selected_models.append(provider.key) or "reasoning",
    )

    def fake_prompt(_stdscr, title, label, mask=False, optional=True):
        prompts.append((title, label))
        assert mask is True
        assert optional is False
        return "desy-api-key"

    monkeypatch.setattr(app, "_prompt_text", fake_prompt)

    assert app._configure_backend(object()) is True

    config = home.load_config()
    assert config["provider"] == "desy"
    assert config["model"] == "reasoning"
    assert home.has_api_key("desy") is True
    assert prompts == [("DESY Assistant", "API key")]
    assert selected_models == ["desy"]


def test_configure_backend_can_keep_existing_api_key(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    home.set_api_key("desy", "saved-desy-key")
    app = AnomxCliApp(home=home)
    prompts: list[tuple[str, str]] = []
    selected_models: list[str] = []

    monkeypatch.setattr(app, "_select_provider", lambda _stdscr: provider_by_key("desy"))
    monkeypatch.setattr(
        app,
        "_menu",
        lambda _stdscr, title, subtitle, choices: (
            prompts.append((title, subtitle)),
            "keep",
        )[1],
    )
    monkeypatch.setattr(
        app,
        "_select_model",
        lambda _stdscr, provider: selected_models.append(provider.key) or "coding",
    )
    monkeypatch.setattr(
        app,
        "_prompt_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("prompt should not run")),
    )

    assert app._configure_backend(object()) is True

    config = home.load_config()
    assert config["provider"] == "desy"
    assert config["model"] == "coding"
    assert prompts == [("DESY Assistant", "API key already configured")]
    assert selected_models == ["desy"]


def test_configure_backend_does_not_save_provider_change_when_model_selection_is_cancelled(
    tmp_path, monkeypatch
):
    home = AnomxHome(tmp_path / "home")
    app = AnomxCliApp(home=home)

    monkeypatch.setattr(app, "_select_provider", lambda _stdscr: provider_by_key("desy"))
    monkeypatch.setattr(app, "_prompt_text", lambda *_args, **_kwargs: "desy-api-key")
    monkeypatch.setattr(app, "_select_model", lambda _stdscr, provider: None)

    assert app._configure_backend(object()) is False

    config = home.load_config()
    assert config["provider"] == "openai"
    assert config["model"] == "gpt-5.5"


def test_run_session_executes_selected_slash_command(tmp_path, monkeypatch):
    class Window:
        def __init__(self):
            self._keys = iter(("/", curses.KEY_DOWN, "\n"))

        def get_wch(self):
            return next(self._keys)

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, use_color=False)
    executed: list[str] = []

    monkeypatch.setattr(app, "_draw_session", lambda *args, **kwargs: None)

    def record_command(_stdscr, command, _current_session, _submitted=""):
        executed.append(command)
        return "exit"

    monkeypatch.setattr(app, "_handle_command", record_command)

    assert app._run_session(Window(), session) == 0
    assert executed == ["/open"]


def test_run_config_panel_closes_after_backend_configuration(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, use_color=False)
    stdscr = object()
    bottom_calls = 0
    configured: list[object] = []

    def fake_bottom_menu(*_args, **_kwargs):
        nonlocal bottom_calls
        bottom_calls += 1
        return "backend"

    monkeypatch.setattr(app, "_bottom_menu", fake_bottom_menu)
    monkeypatch.setattr(
        app,
        "_configure_backend",
        lambda stdscr: configured.append(stdscr) or True,
    )

    app._run_config_panel(stdscr, session)

    assert bottom_calls == 1
    assert configured == [stdscr]
    assert app.state == AgentState.NEW_SESSION


def test_run_config_panel_closes_after_model_selection(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, use_color=False)
    stdscr = object()
    bottom_calls = 0
    selected: list[tuple[object, object]] = []

    def fake_bottom_menu(*_args, **_kwargs):
        nonlocal bottom_calls
        bottom_calls += 1
        return "model"

    monkeypatch.setattr(app, "_bottom_menu", fake_bottom_menu)
    monkeypatch.setattr(
        app,
        "_run_model_panel",
        lambda stdscr, current_session: selected.append((stdscr, current_session)) or True,
    )

    app._run_config_panel(stdscr, session)

    assert bottom_calls == 1
    assert selected == [(stdscr, session)]
    assert app.state == AgentState.NEW_SESSION


def test_open_session_panel_uses_session_copy(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    current_session = home.create_session(repo, provider="openai", model="gpt-5.5")
    for _ in range(9):
        home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, use_color=False)
    captured: dict[str, str | int] = {}

    def fake_bottom_menu(_stdscr, _session, title, subtitle, _choices, restore_nodelay=False):
        captured["title"] = title
        captured["subtitle"] = subtitle
        captured["count"] = len(_choices)
        return None

    monkeypatch.setattr(app, "_bottom_menu", fake_bottom_menu)

    assert app._open_session_panel(object(), current_session) is None
    assert captured == {
        "count": 10,
        "title": "Open Session",
        "subtitle": "Choose a stored session",
    }


def test_prompt_lines_wrap_and_keep_cursor_visible(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    lines = app._prompt_lines("abcdefghijklmnopqrstuvwxyz", width=10)

    assert lines == ["abcdefghij", "klmnopqrst", "uvwxyz"]


def test_prompt_layout_starts_one_line_and_grows_with_input(tmp_path):
    class Window:
        def getmaxyx(self):
            return 40, 28

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    empty = app._prompt_layout(Window(), "")
    short = app._prompt_layout(Window(), "short")
    wrapped = app._prompt_layout(Window(), "x" * 45)
    capped = app._prompt_layout(Window(), "x" * 500)

    assert empty.prompt_height == 1
    assert short.prompt_height == 1
    assert wrapped.prompt_height == 3
    assert capped.prompt_height == 10


def test_prompt_bar_draws_wrapped_input_on_multiple_rows(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 20, 18

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n]))

        def move(self, y, x):
            self.cursor = (y, x)

    window = Window()
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)

    app._draw_prompt_bar(window, "x" * 15, cursor=15)

    assert (16, 4, "x" * 10) in window.writes
    assert (17, 4, "x" * 5) in window.writes


def test_prompt_bar_draws_current_mode_hint(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 20, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def move(self, y, x):
            self.cursor = (y, x)

    window = Window()
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)

    app._draw_prompt_bar(window, "", cursor=0)

    assert (19, 4, "Δ  Confirm Mode (shift+tab to cycle)", 0) in window.writes


def test_prompt_bar_draws_notice_instead_of_mode_hint(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 20, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def move(self, y, x):
            self.cursor = (y, x)

    window = Window()
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._colors = {
        "accent": 1,
        "bold": 2,
        "light": 3,
        "warning": 4,
    }
    notice = "Do you really want to exit anomx? Press Ctrl+C again to confirm."

    app._draw_prompt_bar(window, "", cursor=0, notice=notice, notice_role="light")

    assert (19, 4, notice, 3) in window.writes
    assert not any("Confirm Mode" in text for _, _, text, _ in window.writes)


def test_agent_mode_cycles_and_updates_runtime(tmp_path):
    home = AnomxHome(tmp_path / "home")
    app = AnomxCliApp(home=home)

    app._cycle_agent_mode()
    assert app.agent_mode == AgentMode.AUTONOMOUS
    assert app.runtime.tool_manager.mode == AgentMode.AUTONOMOUS
    assert app._mode_hint_attr_name() == "warning"
    assert home.load_config()["agent_mode"] == AgentMode.AUTONOMOUS.value

    app._cycle_agent_mode()
    assert app.agent_mode == AgentMode.OBSERVER
    assert app.runtime.tool_manager.mode == AgentMode.OBSERVER
    assert app._mode_hint_attr_name() == "light"
    assert home.load_config()["agent_mode"] == AgentMode.OBSERVER.value


def test_app_restores_saved_agent_mode_from_config(tmp_path):
    home = AnomxHome(tmp_path / "home")
    config = home.load_config()
    config["agent_mode"] = AgentMode.AUTONOMOUS.value
    home.save_config(config)

    app = AnomxCliApp(home=home)

    assert app.agent_mode == AgentMode.AUTONOMOUS
    assert app.runtime.tool_manager.mode == AgentMode.AUTONOMOUS


def test_running_ctrl_c_confirmation_requests_interrupt_without_writing(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, use_color=False)
    before = _read_jsonl(session.path)

    first = app._handle_running_key(object(), session, "\x03", "", 0, "", 0.0)

    assert first.notice == (
        "Do you really want to abort the agent? Press Ctrl+C again to confirm."
    )
    assert first.notice_role == "light"
    assert first.abort_key == "Ctrl+C"
    assert not first.exit_requested
    assert _read_jsonl(session.path) == before

    second = app._handle_running_key(
        object(),
        session,
        "\x03",
        first.input_text,
        first.cursor,
        first.abort_key,
        first.abort_deadline,
    )

    assert second.exit_requested
    assert second.notice == "Aborting agent"
    assert second.notice_role == "light"
    assert _read_jsonl(session.path) == before


def test_running_ctrl_x_confirmation_requests_interrupt_without_writing(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, use_color=False)

    first = app._handle_running_key(object(), session, "\x18", "", 0, "", 0.0)

    assert first.notice == (
        "Do you really want to abort the agent? Press Ctrl+X again to confirm."
    )
    assert first.abort_key == "Ctrl+X"
    assert not first.exit_requested

    second = app._handle_running_key(
        object(),
        session,
        "\x18",
        first.input_text,
        first.cursor,
        first.abort_key,
        first.abort_deadline,
    )

    assert second.exit_requested
    assert app._read_message_lines(session.path) == []


def test_running_ctrl_c_confirmation_aborts_turn_without_exiting_cli(tmp_path, monkeypatch):
    class Window:
        def __init__(self):
            self._keys = iter(("\x03", "\x03"))

        def get_wch(self):
            try:
                return next(self._keys)
            except StopIteration as error:
                raise curses.error from error

        def nodelay(self, flag):
            self.nodelay_flag = flag

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, use_color=False)

    def fake_backend_response(_session_path, callbacks=None):
        del callbacks
        while not app.runtime._turn_aborted():
            runtime_module.time.sleep(0.01)
        return ""

    monkeypatch.setattr(app.runtime, "backend_response", fake_backend_response)
    monkeypatch.setattr(app, "_draw_session", lambda *args, **kwargs: None)

    result = app._run_backend_turn(Window(), session)

    assert result.exit_requested is False
    assert app.runtime._turn_aborted() is True
    assert app._read_message_lines(session.path) == [
        MessageLine("agent", MANUAL_INTERRUPT_MESSAGE)
    ]


def test_running_interrupt_collapses_completed_work_before_manual_notice(
    tmp_path, monkeypatch
):
    class Window:
        def __init__(self):
            self._calls = 0

        def get_wch(self):
            self._calls += 1
            if self._calls <= 2:
                runtime_module.time.sleep(0.05)
                raise curses.error
            if self._calls == 3:
                return "\x03"
            if self._calls == 4:
                return "\x03"
            raise curses.error

        def nodelay(self, flag):
            self.nodelay_flag = flag

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, use_color=False)

    def fake_backend_response(_session_path, callbacks=None):
        assert callbacks is not None
        assert callbacks.tool_message is not None
        callbacks.tool_message("Checking repository")
        while not app.runtime._turn_aborted():
            runtime_module.time.sleep(0.01)
        return ""

    monkeypatch.setattr(app.runtime, "backend_response", fake_backend_response)
    monkeypatch.setattr(app, "_draw_session", lambda *args, **kwargs: None)

    app._run_backend_turn(Window(), session)

    messages = app._read_message_lines(session.path)
    assert len(messages) == 2
    assert messages[0].role == "work_summary"
    assert messages[0].text.startswith("Worked for ")
    assert messages[0].text.endswith(" min · expand")
    assert messages[1] == MessageLine("agent", MANUAL_INTERRUPT_MESSAGE)


def test_abort_current_turn_interrupts_working_workers(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)
    worker = runtime_module.WorkerAgentState(
        worker_id="worker-1",
        name="Engineer",
        prompt="Fix this",
        status="working",
        statement="Thinking",
        started_at=runtime_module.utc_now_iso(),
        cancel_event=runtime_module.threading.Event(),
    )
    with runtime._worker_lock:
        runtime._workers[worker.worker_id] = worker

    runtime.abort_current_turn(session.path)

    assert runtime._turn_aborted()
    assert worker.cancel_event.is_set()
    assert worker.status == "interrupted"
    assert worker.statement == ""
    assert worker.finished_at
    events = _read_jsonl(session.path)
    assert any(
        event["payload"].get("type") == "worker_event"
        and event["payload"].get("worker_id") == "worker-1"
        and event["payload"].get("status") == "interrupted"
        for event in events
    )
    assert any(
        event["payload"].get("type") == "system_message"
        and event["payload"].get("role") == "worker"
        and "Worker was interrupted because Anomx was interrupted." in event["payload"].get(
            "message", ""
        )
        for event in events
    )


def test_worker_cancel_event_counts_as_turn_abort(tmp_path):
    cancel_event = runtime_module.threading.Event()
    runtime = AgentRuntime(
        AnomxHome(tmp_path / "home"),
        tmp_path,
        role=AgentRole.WORKER,
        cancel_event=cancel_event,
    )

    assert not runtime._turn_aborted()

    cancel_event.set()

    assert runtime._turn_aborted()


def test_idle_ctrl_c_requires_confirmation_in_prompt_notice(tmp_path, monkeypatch):
    class Window:
        def __init__(self):
            self._keys = iter(("\x03", "\x03"))

        def get_wch(self):
            return next(self._keys)

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, use_color=False)
    notices = []

    def capture_draw(*_args, **kwargs):
        notices.append(kwargs.get("prompt_notice", ""))

    monkeypatch.setattr(app, "_draw_session", capture_draw)

    assert app._run_session(Window(), session) == 0
    assert notices == [
        "",
        "Do you really want to exit anomx? Press Ctrl+C again to confirm.",
    ]


def test_prompt_cursor_stays_on_previous_line_at_wrap_boundary(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    assert app._prompt_cursor_position("abcdefghij", cursor=10, width=10) == (0, 9)


def test_session_scroll_bounds_allow_bottom_slack(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    assert app._session_scroll_bounds(40, 10) == (-9, 30)
    assert app._session_view_start(-9, 40, 10) == 39
    assert app._session_view_start(30, 40, 10) == 0


def test_draw_session_clamps_overscroll_to_real_top(tmp_path):
    class Window:
        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 80

        def addnstr(self, y, x, text, n, attr=0):
            del y, x, text, n, attr

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    messages = [MessageLine("user", f"Line {index}") for index in range(40)]

    viewport = app._draw_session(Window(), session, messages, "", 0, 999)

    assert viewport.start == 0
    assert viewport.scroll == app._session_scroll_bounds(
        viewport.rendered_line_count,
        viewport.body_height,
    )[1]


def test_draw_session_allows_last_line_to_reach_top(tmp_path):
    class Window:
        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 80

        def addnstr(self, y, x, text, n, attr=0):
            del y, x, text, n, attr

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    messages = [MessageLine("user", f"Line {index}") for index in range(40)]

    viewport = app._draw_session(Window(), session, messages, "", 0, -999)

    assert viewport.start == viewport.rendered_line_count - 1
    assert viewport.scroll == app._session_scroll_bounds(
        viewport.rendered_line_count,
        viewport.body_height,
    )[0]


def test_draw_session_maps_anchor_to_equivalent_scroll(tmp_path):
    class Window:
        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 80

        def addnstr(self, y, x, text, n, attr=0):
            del y, x, text, n, attr

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    messages = [MessageLine("user", f"Line {index}") for index in range(40)]

    viewport = app._draw_session(Window(), session, messages, "", 0, 0, anchor_line=21)

    assert viewport.start == 21
    assert viewport.scroll == app._session_scroll_for_start(
        21,
        viewport.rendered_line_count,
        viewport.body_height,
    )


def test_panel_text_lines_sanitize_multiline_commands(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    lines = app._panel_text_lines("sh -lc 'cat <<EOF\nhello\nEOF'", width=80, max_lines=3)

    assert lines == ["sh -lc 'cat <<EOF / hello / EOF'"]


def test_bottom_panel_viewport_shows_hidden_session_counts(tmp_path):
    class Window:
        def getmaxyx(self):
            return 32, 80

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    choices = tuple(
        MenuChoice(f"Session {index}", str(index), f"detail {index}") for index in range(18)
    )
    panel = BottomPanel("Open Session", "Choose a stored session", choices, selected=14)

    viewport = app._bottom_panel_viewport(Window(), panel)

    assert viewport.show_overflow_counts is True
    assert viewport.more_above == 4
    assert viewport.more_below == 3
    assert viewport.visible_indices == tuple(range(4, 15))


def test_openai_backend_requires_configured_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5")
    app = AnomxCliApp(home=home, cwd=repo)

    response = app._openai_response(session.path, "gpt-5")

    assert "OpenAI API key is not configured" in response


def test_anthropic_backend_requires_configured_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    config = home.load_config()
    config["provider"] = "anthropic"
    config["model"] = "claude-opus-4-8"
    home.save_config(config)
    session = home.create_session(repo, provider="anthropic", model="claude-opus-4-8")
    runtime = AgentRuntime(home, repo)

    response = runtime.backend_response(session.path)

    assert "Anthropic API key is not configured" in response


def test_desy_backend_requires_configured_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("DESY_ASSISTANT_API_KEY", raising=False)
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    config = home.load_config()
    config["provider"] = "desy"
    config["model"] = "desy-assistant"
    home.save_config(config)
    session = home.create_session(repo, provider="desy", model="desy-assistant")
    runtime = AgentRuntime(home, repo)

    response = runtime.backend_response(session.path)

    assert "DESY Assistant API key is not configured" in response


def test_openai_invalid_api_key_uses_specific_message(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)

    response = runtime._api_error(
        "openai",
        "OpenAI",
        "OPENAI_API_KEY",
        401,
        json.dumps(
            {
                "error": {
                    "type": "authentication_error",
                    "message": "Incorrect API key provided: sk-invalid",
                }
            }
        ),
    )

    assert "OpenAI credentials were rejected" in response
    assert "OPENAI_API_KEY" in response


def test_anthropic_invalid_api_key_uses_specific_message(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)

    response = runtime._api_error(
        "anthropic",
        "Anthropic",
        "ANTHROPIC_API_KEY",
        401,
        json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "The API key provided is invalid or has been revoked.",
                },
            }
        ),
    )

    assert "Anthropic credentials were rejected" in response
    assert "ANTHROPIC_API_KEY" in response


def test_desy_invalid_api_key_uses_specific_message(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)

    response = runtime._api_error(
        "desy",
        "DESY Assistant",
        "DESY_ASSISTANT_API_KEY",
        401,
        json.dumps({"detail": "Not authenticated"}),
    )

    assert "DESY Assistant credentials were rejected" in response
    assert "DESY_ASSISTANT_API_KEY" in response


def test_runtime_suggests_session_title_without_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(
        session.path,
        "user_message",
        {"message": "Find anomalies in repository data"},
    )
    runtime = AgentRuntime(home, repo)

    assert runtime.suggest_session_title(session.path) == "Find anomalies in repository data"


def test_context_status_is_shown_after_first_message(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(session.path, "user_message", {"message": "Inspect this repo"})
    app = AnomxCliApp(home=home, cwd=repo)

    assert app._context_status(session, "gpt-5.5").endswith("% context left")


def test_session_header_lines_keep_location_as_subtitle(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo)

    assert app._session_header_lines(session, "gpt-5.5") == (
        f"Location: {repo.resolve()}",
    )
    assert app._session_header_meta(session, "openai", "gpt-5.5") == (
        f"{session.session_id[:8]} · openai/gpt-5.5"
    )


def test_openai_text_extraction_prefers_output_text(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)

    assert runtime.extract_openai_text({"output_text": "  ready  "}) == "ready"


def test_openai_text_extraction_reads_output_content(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)

    response = runtime.extract_openai_text(
        {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "first"},
                        {"type": "output_text", "text": "second"},
                    ]
                }
            ]
        }
    )

    assert response == "first\nsecond"


def test_anthropic_text_extraction_reads_text_blocks(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)

    response = runtime.extract_anthropic_text(
        {
            "content": [
                {"type": "text", "text": "first"},
                {"type": "tool_use", "name": "run_command", "id": "toolu_1", "input": {}},
                {"type": "text", "text": "second"},
            ]
        }
    )

    assert response == "first\nsecond"


def test_openai_stream_reports_reasoning_summary_status(tmp_path, monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            yield (
                b'data: {"type":"response.reasoning_summary_text.delta",'
                b'"delta":"Checking the repo"}\n'
            )
            yield b'data: {"type":"response.output_text.delta","delta":"done"}\n'
            yield b'data: {"type":"response.completed","response":{"id":"resp_1"}}\n'
            yield b"data: [DONE]\n"

    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    statuses: list[str] = []
    deltas: list[str] = []

    monkeypatch.setattr(
        runtime_module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(),
    )

    response = runtime._stream_openai_response(
        "sk-test",
        {"model": "gpt-5.5", "input": [], "stream": True},
        deltas.append,
        statuses.append,
    )

    assert response.response_id == "resp_1"
    assert response.text == "done"
    assert statuses == ["Thinking"]
    assert deltas == ["done"]


def test_anthropic_stream_reports_thinking_status(tmp_path, monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            yield (
                b'data: {"type":"content_block_start","index":0,'
                b'"content_block":{"type":"thinking","thinking":"","signature":""}}\n'
            )
            yield (
                b'data: {"type":"content_block_delta","index":0,'
                b'"delta":{"type":"thinking_delta","thinking":"Plan the tool call"}}\n'
            )
            yield (
                b'data: {"type":"content_block_delta","index":0,'
                b'"delta":{"type":"signature_delta","signature":"sig_1"}}\n'
            )
            yield b'data: {"type":"content_block_stop","index":0}\n'
            yield (
                b'data: {"type":"content_block_start","index":1,'
                b'"content_block":{"type":"text","text":""}}\n'
            )
            yield (
                b'data: {"type":"content_block_delta","index":1,'
                b'"delta":{"type":"text_delta","text":"done"}}\n'
            )
            yield b'data: {"type":"content_block_stop","index":1}\n'

    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    statuses: list[str] = []
    deltas: list[str] = []

    monkeypatch.setattr(
        runtime_module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(),
    )

    response = runtime._stream_anthropic_response(
        "sk-ant-test",
        {"model": "claude-opus-4-8", "messages": [], "stream": True},
        deltas.append,
        statuses.append,
    )

    assert response.text == "done"
    assert statuses == ["Thinking"]
    assert deltas == ["done"]
    assert response.content[0]["type"] == "thinking"
    assert response.content[0]["signature"] == "sig_1"


def test_desy_response_omits_thinking_config(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    home.set_api_key("desy", "sk-desy-test")
    session = home.create_session(repo, provider="desy", model="desy-assistant")
    runtime = AgentRuntime(home, repo)
    captured_payloads: list[dict[str, object]] = []

    def fake_stream(api_key, payload, delta_callback, status_callback):
        assert api_key == "sk-desy-test"
        captured_payloads.append(payload)
        return runtime_module.AnthropicStreamResponse(
            text="done",
            tool_calls=(),
            content=({"type": "text", "text": "done"},),
        )

    monkeypatch.setattr(runtime, "_stream_desy_response", fake_stream)

    response = runtime.desy_response(session.path, "desy-assistant")

    assert response == "done"
    assert "thinking" not in captured_payloads[0]


def test_desy_stream_uses_messages_endpoint_and_api_key_header(tmp_path, monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            yield (
                b'data: {"type":"content_block_start","index":0,'
                b'"content_block":{"type":"text","text":""}}\n'
            )
            yield (
                b'data: {"type":"content_block_delta","index":0,'
                b'"delta":{"type":"text_delta","text":"OK"}}\n'
            )
            yield b'data: {"type":"content_block_stop","index":0}\n'

    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    captured_request: dict[str, object] = {}

    def fake_urlopen(request, timeout=120):
        captured_request["url"] = request.full_url
        captured_request["headers"] = dict(request.header_items())
        return FakeResponse()

    monkeypatch.setattr(runtime_module.urllib.request, "urlopen", fake_urlopen)

    response = runtime._stream_desy_response(
        "sk-desy-test",
        {"model": "desy-assistant", "messages": [], "stream": True},
        None,
        None,
    )

    assert response.text == "OK"
    assert captured_request["url"] == "https://assistant.desy.de/api/v1/messages"
    assert captured_request["headers"]["X-api-key"] == "sk-desy-test"


def test_ollama_response_reports_loading_model_status(tmp_path, monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            yield b'{"message":{"content":"hello"}}\n'

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="ollama", model="qwen3.6")
    runtime = AgentRuntime(home, repo)
    statuses: list[str] = []

    monkeypatch.setattr(
        runtime_module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(),
    )

    response = runtime.ollama_response(
        session.path,
        "qwen3.6",
        RuntimeCallbacks(status=statuses.append),
    )

    assert statuses == ["Loading model", "Thinking"]
    assert response == "hello"


def test_ollama_response_executes_streamed_tool_calls(tmp_path, monkeypatch):
    class FakeResponse:
        def __init__(self, lines):
            self._lines = lines

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter(self._lines)

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="ollama", model="qwen3.6")
    runtime = AgentRuntime(home, repo)
    captured_payloads: list[dict[str, object]] = []
    executed: list[tuple[str, dict[str, object]]] = []

    responses = iter(
        [
            FakeResponse(
                [
                    (
                        b'{"message":{"thinking":"Looking for files","tool_calls":['
                        b'{"function":{"name":"run_command","arguments":'
                        b'{"statement":"List files","command":"ls"}}}]}}\n'
                    )
                ]
            ),
            FakeResponse([b'{"message":{"content":"done"}}\n']),
        ]
    )

    def fake_urlopen(request, timeout=120):
        captured_payloads.append(json.loads(request.data.decode("utf-8")))
        return next(responses)

    def fake_execute_tool(name, arguments, callbacks, session_path=None):
        executed.append((name, arguments))
        return json.dumps({"approved": True, "output": "README.md"})

    monkeypatch.setattr(runtime_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(runtime, "_execute_tool", fake_execute_tool)

    response = runtime.ollama_response(session.path, "qwen3.6")

    assert response == "done"
    assert executed == [("run_command", {"statement": "List files", "command": "ls"})]
    assert captured_payloads[0]["tools"]
    assert captured_payloads[0]["think"] is True
    second_messages = captured_payloads[1]["messages"]
    assert second_messages[-2]["role"] == "assistant"
    assert second_messages[-2]["tool_calls"][0]["function"]["name"] == "run_command"
    assert second_messages[-1] == {
        "role": "tool",
        "tool_name": "run_command",
        "content": '{"approved": true, "output": "README.md"}',
    }


def test_runtime_status_events_keep_runtime_text(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="ollama", model="qwen3.6")
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
    events.put(RuntimeUiEvent("status", "Loading model"))

    working_text, working_deadline, final_text, work_count = app._process_runtime_events(
        object(),
        session,
        events,
        "Thinking",
        None,
        "",
        "turn-1",
        0,
        None,
    )

    assert working_text == "Loading model"
    assert working_deadline is None
    assert final_text == ""
    assert work_count == 0


def test_runtime_status_events_keep_thinking_live_only(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="ollama", model="qwen3.6")
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
    events.put(RuntimeUiEvent("status", "Thinking"))

    working_text, working_deadline, final_text, work_count = app._process_runtime_events(
        object(),
        session,
        events,
        None,
        None,
        "",
        "turn-1",
        0,
        None,
    )

    assert working_text == "Thinking"
    assert working_deadline is None
    assert final_text == ""
    assert work_count == 0


def test_wait_returns_immediately_without_running_workers(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    messages: list[str] = []
    statuses: list[str] = []

    output = runtime._wait_tool(
        {},
        RuntimeCallbacks(tool_message=messages.append, status=statuses.append),
    )

    payload = json.loads(output)
    assert payload["waited_seconds"] == 0.0
    assert messages == []
    assert statuses == []


def test_wait_defaults_to_sixty_seconds_and_interrupts_when_workers_finish(
    tmp_path,
    monkeypatch,
):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    statuses: list[str] = []
    calls = 0

    def running_workers():
        nonlocal calls
        calls += 1
        return ("running",) if calls == 1 else ()

    monkeypatch.setattr(runtime, "_running_worker_states", running_workers)

    output = runtime._wait_tool(
        {},
        RuntimeCallbacks(status=statuses.append),
    )

    payload = json.loads(output)
    assert 0.0 <= payload["waited_seconds"] < 1.0
    assert statuses == ["Waiting:60.0"]


def test_waiting_status_renders_countdown(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="ollama", model="qwen3.6")
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
    events.put(RuntimeUiEvent("status", "Waiting:65"))

    working_text, working_deadline, final_text, work_count = app._process_runtime_events(
        object(),
        session,
        events,
        "Thinking",
        None,
        "",
        "turn-1",
        0,
        None,
    )

    assert working_text == "Waiting"
    assert working_deadline is not None
    assert app._working_status_text("Waiting", 165.0, now=100.0) == "Waiting 01:05"
    assert final_text == ""
    assert work_count == 0


def test_waiting_working_line_does_not_animate_dots(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 20, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    window = Window()

    app._draw_working_line(window, 3, 4, "Waiting 00:35", 40, frame=12)
    app._draw_working_line(window, 4, 4, "Thinking", 40, frame=12)
    app._draw_working_line(
        window,
        5,
        4,
        "waiting for long-running command",
        40,
        frame=12,
    )

    assert window.writes[0][2] == "Waiting 00:35"
    assert window.writes[1][2] == "Thinking..."
    assert window.writes[2][2] == "waiting for long-running command..."


def test_platform_connect_loading_renders_connecting_status(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 32, 100

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    window = Window()

    app._draw_platform_connect_loading(window, frame=12)

    assert any(text == "Connect Platform" for _, _, text, _ in window.writes)
    assert any(
        y == 18 and x == 4 and text == "Connecting..." and attr == 0
        for y, x, text, attr in window.writes
    )


def test_platform_connection_form_highlights_selected_row(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []
            self.cursor = None

        def erase(self):
            pass

        def getmaxyx(self):
            return 32, 100

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def move(self, y, x):
            self.cursor = (y, x)

        def refresh(self):
            pass

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._colors = {"accent": 10, "light": 20, "bold": 30, "selected": 40, "ok": 50}
    window = Window()

    app._draw_platform_connection_form(
        window,
        PlatformConnectionDraft(
            url="anomalies.desy.de",
            email="theo.rieken@desy.de",
            password="secret",
        ),
        selected=2,
    )

    assert any(
        x == 4 and text == "Domain:" and attr == 20
        for _, x, text, attr in window.writes
    )
    assert any(
        x == 4 and text == "Email:" and attr == 20
        for _, x, text, attr in window.writes
    )
    assert any(
        x == 4 and text == "Password:" and attr == 10
        for _, x, text, attr in window.writes
    )
    assert any(
        x == 16 and text == "anomalies.desy.de" and attr == 0
        for _, x, text, attr in window.writes
    )
    assert any(
        x == 16 and text == "theo.rieken@desy.de" and attr == 0
        for _, x, text, attr in window.writes
    )
    assert any(
        x == 16 and text == "******" and attr == 0
        for _, x, text, attr in window.writes
    )
    assert any(
        text == "Esc Cancel · ↑↓ Navigate · Enter for Login"
        for _, _, text, _ in window.writes
    )

    window.writes.clear()
    app._draw_platform_connection_form(
        window,
        PlatformConnectionDraft(
            url="anomalies.desy.de",
            email="theo.rieken@desy.de",
            password="secret",
        ),
        selected=2,
        status="Connection alive.",
        status_role="ok",
    )

    assert any(
        x == 4 and text == "Connection alive." and attr == 50
        for _, x, text, attr in window.writes
    )

    window.writes.clear()
    app._draw_platform_connection_form(
        window,
        PlatformConnectionDraft(
            url="anomalies.desy.de",
            email="theo.rieken@desy.de",
            password="*****",
        ),
        selected=2,
        status="Checking connection...",
        editable=False,
    )

    assert any(text == "Password:" and attr == 20 for _, _, text, attr in window.writes)
    assert any(
        x == 4 and text == "Checking connection..." and attr == 0
        for _, x, text, attr in window.writes
    )


def test_platform_connection_form_returns_to_password_after_login_error(
    tmp_path,
    monkeypatch,
):
    class Window:
        def __init__(self):
            self.writes = []
            self._keys = iter((*"wrong-password", "\n", "\x1b"))

        def erase(self):
            pass

        def getmaxyx(self):
            return 32, 100

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def get_wch(self):
            return next(self._keys)

        def move(self, y, x):
            pass

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    home.set_platform_form_defaults(
        url="anomalies.desy.de",
        email="theo.rieken@desy.de",
    )
    app = AnomxCliApp(home=home, use_color=False)
    app._colors = {"accent": 10, "light": 20, "bold": 30, "danger": 40, "selected": 50}
    monkeypatch.setattr(
        app,
        "_connect_platform_with_loading",
        lambda *_args: (_ for _ in ()).throw(
            platform_client_module.PlatformClientError("Invalid credentials.")
        ),
    )
    window = Window()

    assert app._run_platform_connection_form(window) is None
    assert any(
        x == 4 and text == "Invalid credentials." and attr == 40
        for _, x, text, attr in window.writes
    )
    assert any(text == "Password:" and attr == 10 for _, _, text, attr in window.writes)
    config = home.load_config()
    assert config["platform_last_url"] == "anomalies.desy.de"
    assert config["platform_last_email"] == "theo.rieken@desy.de"


def test_manage_platform_logout_requires_confirmation(tmp_path, monkeypatch):
    class Window:
        def __init__(self):
            self.writes = []
            self._keys = iter(("\x04", "\x1b", "\x04", "\n", "\n"))

        def erase(self):
            pass

        def getmaxyx(self):
            return 32, 100

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def get_wch(self):
            return next(self._keys)

        def move(self, y, x):
            pass

        def refresh(self):
            pass

        def nodelay(self, flag):
            pass

    home = AnomxHome(tmp_path / "home")
    home.set_platform_connection(
        url="https://anomalies.msktools.desy.de/api",
        token="platform-token",
        user_email="theo.rieken@desy.de",
    )
    app = AnomxCliApp(home=home, use_color=False)
    monkeypatch.setattr(platform_client_module, "heartbeat_platform_connection", lambda _home: True)
    window = Window()

    assert (
        app._run_platform_management_form(
            window,
            home.platform_connection(),
            initial_status="Connection alive.",
            initial_status_role="ok",
            check_connection=False,
        )
        is True
    )

    assert any(text == "Logout this CLI agent?" for _, _, text, _ in window.writes)
    assert home.platform_connection() is None


def test_approval_events_persist_command_decision(tmp_path, monkeypatch):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="ollama", model="qwen3.6")
    response_queue: queue.SimpleQueue[ApprovalChoice] = queue.SimpleQueue()
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
    events.put(
        RuntimeUiEvent(
            "approval",
            approval_request=CommandApprovalRequest(
                command="cat README.md",
                statement="Read project overview",
                reason="Needs approval",
                canonical_command="cat README.md",
                allowance_subject="cat",
            ),
            approval_response=response_queue,
        )
    )
    monkeypatch.setattr(
        app,
        "_request_command_approval",
        lambda *args: ApprovalChoice.ALLOW,
    )

    working_text, working_deadline, final_text, work_count = app._process_runtime_events(
        object(),
        session,
        events,
        "Thinking",
        None,
        "",
        "turn-1",
        0,
        None,
    )

    assert response_queue.get_nowait() == ApprovalChoice.ALLOW
    assert working_text == "Thinking"
    assert working_deadline is None
    assert final_text == ""
    assert work_count == 1
    assert app._read_message_lines(session.path) == [
        MessageLine("tool", "Approved command: cat README.md", "turn-1")
    ]


def test_question_events_return_answer_and_persist_hidden_context(tmp_path, monkeypatch):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="ollama", model="qwen3.6")
    response_queue: queue.SimpleQueue[QuestionResponse] = queue.SimpleQueue()
    question = QuestionRequest(
        question="Which stack should I use?",
        kind="select",
    )
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
    events.put(
        RuntimeUiEvent(
            "question",
            question_request=question,
            question_response=response_queue,
        )
    )
    captured: dict[str, object] = {}

    def fake_request_question(*_args, **kwargs):
        captured.update(kwargs)
        return QuestionResponse(
            answered=True,
            answer="next",
            selected_label="Next.js",
            kind="select",
        )

    monkeypatch.setattr(app, "_request_question", fake_request_question)

    working_text, working_deadline, final_text, work_count = app._process_runtime_events(
        object(),
        session,
        events,
        "Thinking",
        None,
        "",
        "turn-1",
        0,
        11,
        scroll=4,
    )

    answer = response_queue.get_nowait()
    raw_events = home.read_session_events(session.path)

    assert answer.answer == "next"
    assert working_text == "Thinking"
    assert working_deadline is None
    assert final_text == ""
    assert work_count == 0
    assert captured == {"scroll": 4, "anchor_line": 11}
    assert app._read_message_lines(session.path) == []
    assert any(
        event_payload_type(event) == "system_message"
        and event_payload(event).get("role") == "question"
        and "Which stack should I use?" in str(event_payload(event).get("message"))
        for event in raw_events
    )


def test_output_message_events_persist_as_agent_messages(tmp_path, monkeypatch):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="ollama", model="qwen3.6")
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
    events.put(RuntimeUiEvent("message", "I am checking the repository."))
    monkeypatch.setattr(app, "_fake_type_message", lambda *args, **kwargs: None)

    working_text, working_deadline, final_text, work_count = app._process_runtime_events(
        object(),
        session,
        events,
        "Thinking",
        None,
        "",
        "turn-1",
        0,
        None,
    )

    assert working_text is None
    assert working_deadline is None
    assert final_text == ""
    assert work_count == 1
    assert app._read_message_lines(session.path) == [
        MessageLine("agent", "I am checking the repository.", "turn-1")
    ]


def test_intermediate_message_clears_streamed_final_buffer(tmp_path, monkeypatch):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="ollama", model="qwen3.6")
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
    events.put(RuntimeUiEvent("delta", "This looked final."))
    events.put(RuntimeUiEvent("message", "This is actually a progress update."))
    monkeypatch.setattr(app, "_fake_type_message", lambda *args, **kwargs: None)

    working_text, working_deadline, final_text, work_count = app._process_runtime_events(
        object(),
        session,
        events,
        "Thinking",
        None,
        "",
        "turn-1",
        0,
        None,
    )

    assert working_text is None
    assert working_deadline is None
    assert final_text == ""
    assert work_count == 1
    assert app._read_message_lines(session.path) == [
        MessageLine("agent", "This is actually a progress update.", "turn-1")
    ]


def test_tool_message_events_persist_as_work_messages(tmp_path, monkeypatch):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="ollama", model="qwen3.6")
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
    events.put(RuntimeUiEvent("tool_message", "Starting Engineer Worker"))
    monkeypatch.setattr(app, "_fake_type_message", lambda *args, **kwargs: None)

    working_text, working_deadline, final_text, work_count = app._process_runtime_events(
        object(),
        session,
        events,
        "Thinking",
        None,
        "",
        "turn-1",
        0,
        None,
    )

    assert working_text is None
    assert working_deadline is None
    assert final_text == ""
    assert work_count == 1
    assert app._read_message_lines(session.path) == [
        MessageLine("tool", "Starting Engineer Worker", "turn-1")
    ]


def test_wait_schema_is_openai_strict_compatible(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path, role=AgentRole.OPERATOR)
    runtime._running_worker_states = lambda: ("running",)  # type: ignore[method-assign]
    wait_tool = next(tool for tool in runtime._openai_tools() if tool["name"] == "wait")

    assert wait_tool["parameters"]["properties"] == {}
    assert wait_tool["parameters"]["required"] == []


def test_markdown_is_rendered_as_terminal_lines():
    lines = markdown_to_terminal_lines(
        "## Finding\n\n- **Signal** changed in `sensor_a`\n```text\nraw value\n```",
        width=80,
    )

    assert lines == ["Finding", "", "- Signal changed in sensor_a", "  raw value"]


def test_markdown_pipe_tables_render_as_aligned_terminal_tables():
    rendered = markdown_to_terminal_rendered_lines(
        "\n".join(
            [
                "Data channels",
                "",
                "| Channel | Description |",
                "| --- | --- |",
                "| LOCK_STATUS.VALUE.RD | Lock status |",
                "| RF_FUND_MAG.SPEC | RF magnitude spectrum |",
            ]
        ),
        width=80,
    )
    texts = [line.text for line in rendered]
    table_lines = texts[2:]

    assert texts[:2] == ["Data channels", ""]
    assert table_lines[0].startswith("┌")
    assert table_lines[-1].startswith("└")
    assert "Channel" in table_lines[1]
    assert "Description" in table_lines[1]
    assert rendered[2].style == "table_border"
    assert rendered[3].style == "table_header"
    assert rendered[5].style == "table_row"
    assert len({len(line) for line in table_lines}) == 1
    assert not any("---" in line for line in table_lines)


def test_markdown_pipe_tables_wrap_to_terminal_width():
    lines = markdown_to_terminal_lines(
        "\n".join(
            [
                "| Channel | Description |",
                "| --- | --- |",
                "| VERY_LONG_SIGNAL_IDENTIFIER_WITHOUT_SPACES | "
                "A long description that should wrap cleanly inside the cell |",
            ]
        ),
        width=42,
    )

    assert all(len(line) <= 42 for line in lines)
    assert any("VERY_LONG_SIGNAL" in line for line in lines)
    assert any("cleanly inside" in line for line in lines)


def test_table_row_sides_use_border_attribute(tmp_path):
    class Window:
        def __init__(self):
            self.calls = []

        def getmaxyx(self):
            return 5, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.calls.append((y, x, text, n, attr))

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._colors = {
        "table_border": 10,
        "table_header": 20,
    }

    header_window = Window()
    app._draw_table_line(header_window, 0, 0, "│ Header │", 80, "table_header")

    assert [call[2] for call in header_window.calls] == ["│", " Header ", "│"]
    assert [call[4] for call in header_window.calls] == [10, 20, 10]

    body_window = Window()
    app._draw_table_line(body_window, 0, 0, "│ Value │", 80, "table_row")

    assert [call[2] for call in body_window.calls] == ["│", " Value ", "│"]
    assert [call[4] for call in body_window.calls] == [10, curses.A_NORMAL, 10]


def test_transcript_rendering_does_not_prefix_roles(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    rendered = app._render_messages(
        [
            MessageLine("user", "Hi there"),
            MessageLine("agent", "**Hello** from Anomx"),
        ],
        width=80,
    )

    assert rendered[0] == MessageLine("user", "Hi there")
    assert rendered[2] == MessageLine("agent", "Hello from Anomx")


def test_work_messages_render_without_blank_gaps(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    rendered = app._render_messages(
        [
            MessageLine("tool", "Thinking..."),
            MessageLine("tool", "Checking repo..."),
            MessageLine("tool", "Started Engineer Worker"),
        ],
        width=80,
    )

    assert rendered == [
        MessageLine("tool", "Thinking..."),
        MessageLine("tool", "Checking repo..."),
        MessageLine("tool", "Started Engineer Worker"),
    ]


def test_long_work_messages_render_as_single_click_to_expand_line(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    message = "Approved command: " + "python -c 'print(\"long command\")' " * 4

    rendered = app._render_messages(
        [MessageLine("approved", message, expansion_key="line-1")],
        width=56,
    )

    assert len(rendered) == 1
    assert rendered[0].role == "approved"
    assert rendered[0].expansion_key == "line-1"
    assert rendered[0].text == "Approved command ... click to expand"
    assert len(rendered[0].text) <= 56


def test_approval_work_messages_always_collapse_to_label(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    rendered = app._render_messages(
        [
            MessageLine(
                "tool",
                "Always approved cat: cat /Users/theorieken/Software/test/README.md",
                expansion_key="line-1",
            ),
            MessageLine(
                "tool",
                "Rejected command: rm -rf /Users/theorieken/Software/test",
                expansion_key="line-2",
            ),
        ],
        width=120,
    )

    assert rendered == [
        MessageLine("tool", "Always approved cat ... click to expand", expansion_key="line-1"),
        MessageLine("tool", "Rejected command ... click to expand", expansion_key="line-2"),
    ]


def test_expanded_work_message_renders_inside_box(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    app._toggle_work_line("line-1")

    rendered = app._render_messages(
        [
            MessageLine(
                "tool",
                "Approved command: printf 'hello'\ncp /tmp/page.tsx src/app/page.tsx",
                expansion_key="line-1",
            )
        ],
        width=48,
    )

    assert [line.role for line in rendered] == ["work_box"] * len(rendered)
    assert rendered[0].text.startswith("╭")
    assert rendered[-1].text.startswith("╰")
    assert any("Approved command" in line.text for line in rendered)
    assert all(line.expansion_key == "line-1" for line in rendered)


def test_rendering_inserts_blank_lines_when_message_kind_changes(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    rendered = app._render_messages(
        [
            MessageLine("user", "Build this"),
            MessageLine("tool", "Checking repo"),
            MessageLine("approved", "Approved command: cat README.md"),
            MessageLine("agent", "Done"),
        ],
        width=80,
    )

    assert rendered == [
        MessageLine("user", "Build this"),
        MessageLine("meta", ""),
        MessageLine("tool", "Checking repo"),
        MessageLine("approved", "Approved command ... click to expand"),
        MessageLine("meta", ""),
        MessageLine("agent", "Done"),
    ]


def test_live_working_status_gets_spacing_after_user_message(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    session = AnomxHome(tmp_path / "home").create_session(
        tmp_path,
        provider="ollama",
        model="qwen3.6",
    )

    rendered = app._session_rendered_lines(
        session,
        [MessageLine("user", "Checkout this repo")],
        width=80,
        working_text="Loading model",
    )

    assert rendered == [
        MessageLine("user", "Checkout this repo"),
        MessageLine("meta", ""),
        MessageLine("working", "Loading model"),
    ]


def test_work_summary_collapses_turn_local_agent_and_work_messages(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(
        session.path,
        "agent_message",
        {"message": "I am checking the repository.", "turn_id": "turn-1"},
    )
    home.append_session_event(
        session.path,
        "work_message",
        {"message": "Reading README", "role": "tool", "turn_id": "turn-1"},
    )
    home.append_session_event(
        session.path,
        "work_summary",
        {"message": "Worked for 00:01 min", "turn_id": "turn-1"},
    )
    home.append_session_event(
        session.path,
        "agent_message",
        {"message": "Final response"},
    )
    app = AnomxCliApp(home=home, cwd=repo)

    assert app._read_message_lines(session.path) == [
        MessageLine("work_summary", "Worked for 00:01 min · expand", "turn-1"),
        MessageLine("agent", "Final response"),
    ]


def test_session_rendering_starts_with_messages_not_status_metadata(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    session = AnomxHome(tmp_path / "home").create_session(
        tmp_path,
        provider="openai",
        model="gpt-5.5",
    )

    rendered = app._session_rendered_lines(
        session,
        [MessageLine("user", "Hi")],
        width=80,
    )

    assert rendered[0] == MessageLine("user", "Hi")


def test_system_messages_render_with_persisted_roles(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(
        session.path,
        "system_message",
        {"message": "Blocked command: reboot", "role": "forbidden"},
    )
    app = AnomxCliApp(home=home, cwd=repo)

    assert app._read_message_lines(session.path) == [
        MessageLine("forbidden", "Blocked command: reboot")
    ]


def test_forbidden_messages_use_light_attr(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._colors = {
        "light": curses.A_DIM,
        "danger": curses.A_BOLD,
    }

    assert app._line_attr("forbidden") == curses.A_DIM
    assert app._line_attr("notice") == curses.A_DIM


def test_runtime_rejects_tool_paths_outside_workspace(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), repo)

    output = runtime._execute_tool(
        "run_command",
        {"statement": "I am checking a file", "command": "cat ../secret.txt"},
        RuntimeCallbacks(),
    )

    assert "outside the trusted workspace" in output


def test_runtime_allows_nested_launch_to_edit_discovered_workspace_root(tmp_path):
    repo = tmp_path / "repo"
    nested = repo / "src" / "anomx" / "agent"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()
    (repo / "source.txt").write_text("hello", encoding="utf-8")
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), nested)

    output = runtime._execute_tool(
        "run_command",
        {
            "statement": "Writing file from nested launch directory",
            "command": "cat ../../../source.txt > ../../../target.txt",
        },
        RuntimeCallbacks(approval=lambda _request: ApprovalChoice.ALLOW),
    )

    payload = json.loads(output)
    assert payload["approved"] is True
    assert (repo / "target.txt").read_text(encoding="utf-8") == "hello"


def test_runtime_still_rejects_paths_outside_discovered_workspace_root(tmp_path):
    repo = tmp_path / "repo"
    nested = repo / "src" / "anomx" / "agent"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), nested)

    output = runtime._execute_tool(
        "run_command",
        {"statement": "Reading outside file", "command": "cat ../../../../secret.txt"},
        RuntimeCallbacks(),
    )

    assert "outside the trusted workspace" in output


def test_tool_manager_terminates_running_command_when_cancelled(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    cancel_event = runtime_module.threading.Event()
    manager = CliToolManager(repo, mode=AgentMode.AUTONOMOUS, cancel_event=cancel_event)
    result: dict[str, str] = {}

    def run_command():
        result["output"] = manager._execute_shell_command("sleep 10; echo done")

    thread = runtime_module.threading.Thread(target=run_command)
    thread.start()
    runtime_module.time.sleep(0.1)

    cancel_event.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert result["output"] == "Command stopped because the agent was interrupted."


def test_runtime_records_forbidden_command_callback(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), repo)
    events: list[tuple[str, str]] = []

    runtime._execute_tool(
        "run_command",
        {"statement": "Restarting host", "command": "reboot"},
        RuntimeCallbacks(system_message=lambda role, message: events.append((role, message))),
    )

    assert events == [
        (
            "forbidden",
            "Blocked command: reboot · reboot can modify or control the host system.",
        )
    ]


def test_runtime_includes_current_mode_in_system_prompt(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path, mode=AgentMode.OBSERVER)

    assert "Current mode: Observer Mode." in runtime._instructions()

    runtime.set_mode(AgentMode.AUTONOMOUS)

    assert "Current mode: Autonomous Mode." in runtime._instructions()


def test_runtime_includes_workspace_access_in_system_prompt(tmp_path):
    repo = tmp_path / "repo"
    nested = repo / "src" / "anomx" / "agent"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), nested)

    instructions = runtime._instructions()

    assert f"Trusted workspace root: {repo}" in instructions
    assert f"Shell starts in: {nested}" in instructions


def test_operator_prompt_pushes_execution_after_planning(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path, mode=AgentMode.CONFIRM)
    instructions = runtime._instructions()

    assert "A plan is not a stopping point." in instructions
    assert "start at least one Worker agent" in instructions
    assert "Do not ask for that approval in prose before calling tools." in instructions


def test_runtime_includes_session_command_policy_in_operator_and_worker_prompts(tmp_path):
    allowed = {"cmd:python"}
    rejected = {"cmd:curl"}

    operator = AgentRuntime(
        AnomxHome(tmp_path / "home"),
        tmp_path,
        allowed,
        rejected,
        role=AgentRole.OPERATOR,
    )
    worker = AgentRuntime(
        AnomxHome(tmp_path / "home"),
        tmp_path,
        allowed,
        rejected,
        role=AgentRole.WORKER,
    )

    operator_instructions = operator._instructions()
    worker_instructions = worker._instructions()

    assert "Already approved for this session: python." in operator_instructions
    assert "Never call run_command with these command families in this session: curl." in (
        operator_instructions
    )
    assert "Never call run_command with these command families in this session: curl." in (
        worker_instructions
    )


def test_runtime_tool_schemas_are_role_specific(tmp_path):
    operator = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    worker = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path, role=AgentRole.WORKER)

    operator_tools = operator._openai_tools()
    operator_names = [tool["name"] for tool in operator_tools]
    operator_run_command = next(tool for tool in operator_tools if tool["name"] == "run_command")
    worker_tools = worker._openai_tools()

    assert "run_command" in operator_names
    assert "run_cli_command" not in operator_names
    assert "statement" in operator_run_command["parameters"]["properties"]
    assert operator_run_command["parameters"]["required"] == ["statement", "command"]
    expected_operator_tools = {
        "start_agent",
        "prompt_agent",
        "interrupt_agent",
        "remove_agent",
        "start_process",
        "end_process",
        "ask_question",
        "create_plan",
        "update_plan",
        "remove_plan",
    }
    assert expected_operator_tools.issubset(set(operator_names))
    assert "check_agent" not in operator_names
    assert "stop_agent" not in operator_names
    assert "wait" not in operator_names
    for tool in operator_tools:
        if tool["name"] in {"output_message", "wait"}:
            continue
        assert "statement" in tool["parameters"]["properties"]
        assert "statement" in tool["parameters"]["required"]
    assert [tool["name"] for tool in worker_tools] == ["run_command"]
    assert worker_tools[0]["parameters"]["required"] == ["statement", "command"]


def test_wait_tool_is_only_advertised_with_running_workers(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)

    assert "wait" not in [tool["name"] for tool in runtime._openai_tools()]

    runtime._running_worker_states = lambda: ("running",)  # type: ignore[method-assign]

    wait_tool = next(tool for tool in runtime._openai_tools() if tool["name"] == "wait")
    assert wait_tool["parameters"] == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def test_ask_question_tool_returns_interactive_answer(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    seen_requests: list[QuestionRequest] = []

    def answer_question(request: QuestionRequest) -> QuestionResponse:
        seen_requests.append(request)
        return QuestionResponse(
            answered=True,
            answer="next",
            selected_label="Next.js",
            kind=request.kind,
        )

    output = runtime._execute_tool(
        "ask_question",
        {
            "statement": "Choosing framework",
            "question": "Which framework should I use?",
            "kind": "select",
            "options": [
                {
                    "label": "Next.js",
                    "value": "next",
                    "description": "Use the existing website stack.",
                }
            ],
            "placeholder": None,
            "default": None,
            "allow_custom": False,
        },
        RuntimeCallbacks(question=answer_question),
    )

    payload = json.loads(output)

    assert payload == {
        "answer": "next",
        "answered": True,
        "cancelled": False,
        "kind": "select",
        "selected_label": "Next.js",
    }
    assert seen_requests[0].options[0].label == "Next.js"


def test_ask_question_rejects_select_without_options(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)

    output = runtime._execute_tool(
        "ask_question",
        {
            "statement": "Choosing framework",
            "question": "Which framework should I use?",
            "kind": "select",
            "options": [],
            "placeholder": None,
            "default": None,
            "allow_custom": False,
        },
        RuntimeCallbacks(question=lambda request: QuestionResponse(True, "unused")),
    )

    payload = json.loads(output)

    assert payload["answered"] is False
    assert "select questions require options" in payload["error"]


def test_ollama_final_text_becomes_progress_update_while_worker_runs(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="ollama", model="qwen3.6")
    runtime = AgentRuntime(home, tmp_path)
    progress_messages: list[str] = []
    captured_messages: list[list[dict[str, object]]] = []
    worker_running = True

    responses = iter(
        (
            runtime_module.OllamaStreamResponse(
                "The Engineer is working. I will wait.",
                "",
                (),
                {
                    "role": "assistant",
                    "content": "The Engineer is working. I will wait.",
                },
            ),
            runtime_module.OllamaStreamResponse(
                "The work is complete.",
                "",
                (),
                {"role": "assistant", "content": "The work is complete."},
            ),
        )
    )

    def fake_running_workers():
        return ("running",) if worker_running else ()

    def fake_wait(arguments, callbacks=None):
        nonlocal worker_running
        assert arguments == {}
        worker_running = False
        return '{"waited_seconds": 0.0, "workers": []}'

    def fake_stream(model, messages, callbacks):
        assert model == "qwen3.6"
        captured_messages.append([dict(message) for message in messages])
        return next(responses)

    monkeypatch.setattr(runtime, "_running_worker_states", fake_running_workers)
    monkeypatch.setattr(runtime, "_wait_tool", fake_wait)
    monkeypatch.setattr(runtime, "_stream_ollama_response", fake_stream)

    response = runtime.ollama_response(
        session.path,
        "qwen3.6",
        RuntimeCallbacks(message=progress_messages.append),
    )

    assert response == "The work is complete."
    assert progress_messages == ["The Engineer is working. I will wait."]
    assert len(captured_messages) == 2
    assert captured_messages[1][-1]["role"] == "user"
    assert "not a final answer" in str(captured_messages[1][-1]["content"])


def test_operator_run_command_persists_statement_as_work_message(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("hello", encoding="utf-8")
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), repo)
    messages: list[str] = []

    output = runtime._execute_tool(
        "run_command",
        {"statement": "Reading project overview", "command": "cat README.md"},
        RuntimeCallbacks(tool_message=messages.append),
    )

    payload = json.loads(output)
    assert payload["approved"] is True
    assert payload["output"] == "hello"
    assert messages == ["Reading project overview"]


def test_pending_approval_auto_allows_after_switch_to_autonomous(tmp_path):
    class Window:
        def nodelay(self, flag):
            self.nodelay_flag = flag

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    app.agent_mode = AgentMode.AUTONOMOUS

    decision = app._request_command_approval(
        Window(),
        session,
        CommandApprovalRequest(
            command="cat source.txt > target.txt",
            statement="Writing file",
            reason="Shell operators require explicit approval.",
            canonical_command="cat source.txt > target.txt",
            allowance_label="cat commands",
            allowance_subject="cat",
        ),
    )

    assert decision == ApprovalChoice.ALLOW


def test_approval_menu_describes_command_family_allowance(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    captured: dict[str, object] = {}

    def fake_bottom_menu(
        _stdscr,
        _session,
        title,
        subtitle,
        choices,
        restore_nodelay=False,
        autonomous_value=None,
    ):
        captured["title"] = title
        captured["choices"] = choices
        captured["autonomous_value"] = autonomous_value
        return ApprovalChoice.ALWAYS_ALLOW.value

    monkeypatch.setattr(app, "_bottom_menu", fake_bottom_menu)

    decision = app._request_command_approval(
        object(),
        session,
        CommandApprovalRequest(
            command="cat source.txt > target.txt",
            statement="Writing file",
            reason="Shell operators require explicit approval.",
            canonical_command="cat source.txt > target.txt",
            allowance_label="cat commands",
            allowance_subject="cat",
        ),
    )

    choices = captured["choices"]
    assert decision == ApprovalChoice.ALWAYS_ALLOW
    assert captured["autonomous_value"] == ApprovalChoice.ALLOW.value
    assert choices[0].label == "Approve"
    assert choices[2].label == "Always approve cat"
    assert choices[2].detail == "Trust cat commands for this session"
    assert choices[3].label == "Always reject cat"


def test_approval_menu_can_always_reject_command_family(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)

    monkeypatch.setattr(
        app,
        "_bottom_menu",
        lambda *_args, **_kwargs: ApprovalChoice.ALWAYS_REJECT.value,
    )

    decision = app._request_command_approval(
        object(),
        session,
        CommandApprovalRequest(
            command="curl -s https://example.com",
            statement="Fetching URL",
            reason="curl may read, compute, install, or modify files.",
            canonical_command="curl -s https://example.com",
            allowance_label="curl commands",
            allowance_subject="curl",
        ),
    )

    assert decision == ApprovalChoice.ALWAYS_REJECT


def test_request_question_select_uses_bottom_menu(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    captured: dict[str, object] = {}

    def fake_bottom_menu(_stdscr, _session, title, subtitle, choices, **kwargs):
        captured["title"] = title
        captured["subtitle"] = subtitle
        captured["choices"] = choices
        captured["scroll"] = kwargs.get("scroll")
        captured["anchor_line"] = kwargs.get("anchor_line")
        return "react"

    monkeypatch.setattr(app, "_bottom_menu", fake_bottom_menu)

    answer = app._request_question(
        object(),
        session,
        QuestionRequest(
            question="Which UI stack?",
            kind="select",
            options=(
                runtime_module.QuestionOption("React", "react", "Use React."),
                runtime_module.QuestionOption("Vue", "vue", "Use Vue."),
            ),
        ),
        scroll=6,
        anchor_line=3,
    )

    assert answer == QuestionResponse(
        answered=True,
        answer="react",
        selected_label="React",
        kind="select",
    )
    assert captured["title"] == "Question"
    assert captured["subtitle"] == "Which UI stack?"
    assert captured["choices"][0].label == "React"
    assert captured["scroll"] == 6
    assert captured["anchor_line"] == 3


def test_request_question_confirm_adds_yes_no_options(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    captured: dict[str, object] = {}

    def fake_bottom_menu(_stdscr, _session, _title, _subtitle, choices, **_kwargs):
        captured["choices"] = choices
        return "yes"

    monkeypatch.setattr(app, "_bottom_menu", fake_bottom_menu)

    answer = app._request_question(
        object(),
        session,
        QuestionRequest(question="Continue?", kind="confirm"),
    )

    assert answer.answer == "yes"
    assert answer.selected_label == "Yes"
    assert [choice.value for choice in captured["choices"]] == ["yes", "no"]


def test_request_question_text_preserves_viewport(tmp_path, monkeypatch):
    class Window:
        def get_wch(self):
            return "\n"

        def nodelay(self, _flag):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    captured: list[dict[str, object]] = []

    def fake_draw_session(
        _stdscr,
        _session,
        _messages,
        _input_text,
        _cursor,
        scroll,
        **kwargs,
    ):
        captured.append(
            {
                "scroll": scroll,
                "anchor_line": kwargs.get("anchor_line"),
            }
        )
        return None

    monkeypatch.setattr(app, "_draw_session", fake_draw_session)

    answer = app._request_question(
        Window(),
        session,
        QuestionRequest(question="What is your name?", kind="text", default="Ada"),
        scroll=9,
        anchor_line=2,
    )

    assert answer.answer == "Ada"
    assert captured == [{"scroll": 9, "anchor_line": 2}]


def test_create_and_update_plan_are_persisted(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)

    runtime._execute_tool(
        "create_plan",
        {
            "steps": [
                {
                    "title": "Inspect",
                    "description": "Read the relevant files.",
                    "is_done": False,
                },
                {
                    "title": "Validate",
                    "description": "Run focused checks.",
                    "is_done": False,
                },
            ]
        },
        RuntimeCallbacks(),
        session.path,
    )
    runtime._execute_tool(
        "update_plan",
        {
            "steps": [
                {
                    "position": 1,
                    "title": "Inspect",
                    "description": "Read the relevant files.",
                    "is_done": True,
                },
                {
                    "position": 2,
                    "title": "Validate",
                    "description": "Run focused checks.",
                    "is_done": False,
                },
            ]
        },
        RuntimeCallbacks(),
        session.path,
    )

    steps = latest_plan_steps(home.read_session_events(session.path))
    instructions = runtime._instructions(session.path)

    assert [(step.position, step.title, step.is_done) for step in steps] == [
        (1, "Inspect", True),
        (2, "Validate", False),
    ]
    assert "1. [done] Inspect" in instructions


def test_remove_plan_clears_current_plan(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)
    messages: list[str] = []

    runtime._execute_tool(
        "create_plan",
        {
            "steps": [
                {
                    "title": "Inspect",
                    "description": "Read files.",
                    "is_done": False,
                }
            ]
        },
        RuntimeCallbacks(),
        session.path,
    )
    output = runtime._execute_tool(
        "remove_plan",
        {},
        RuntimeCallbacks(tool_message=messages.append),
        session.path,
    )

    assert json.loads(output)["removed"] is True
    assert latest_plan_steps(home.read_session_events(session.path)) == ()
    assert messages == ["Removed plan"]


def test_worker_lifecycle_events_are_persisted(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)

    def fake_worker_response(self, prompt, callbacks=None):
        assert prompt == "Inspect files"
        if callbacks is not None and callbacks.tool_message is not None:
            callbacks.tool_message("Listing files")
        if callbacks is not None and callbacks.command is not None:
            callbacks.command("Listing files", "ls", "README.md")
        if callbacks is not None and callbacks.status is not None:
            callbacks.status("Thinking")
        return "Worker report"

    monkeypatch.setattr(AgentRuntime, "backend_response_for_prompt", fake_worker_response)

    started = json.loads(
        runtime._execute_tool(
            "start_agent",
            {"name": "Engineer", "prompt": "Inspect files"},
            RuntimeCallbacks(),
            session.path,
        )
    )
    runtime._wait_tool({"seconds": 1})

    events = home.read_session_events(session.path)
    snapshots = worker_snapshots(events)
    messages = AnomxCliApp(home=home, cwd=repo)._read_message_lines(session.path)
    checked = json.loads(
        runtime._execute_tool(
            "check_agent",
            {"agent_id": started["agent_id"]},
            RuntimeCallbacks(),
            session.path,
        )
    )

    assert started["name"] == "Engineer"
    assert snapshots[-1].status == "ready"
    assert snapshots[-1].statement == ""
    assert running_worker_snapshots(events) == ()
    assert not any(line.role == "worker" and "Worker report" in line.text for line in messages)
    assert checked["commands"] == [
        {
            "statement": "Listing files",
            "command": "ls",
            "output": "README.md",
        }
    ]
    running_statements = [
        str(event["payload"].get("statement"))
        for event in events
        if event_payload_type(event) == "worker_event"
        and event_payload(event).get("status") == "working"
    ]
    assert running_statements[-1] == "Listing files"
    assert "Thinking" not in running_statements[1:]


def test_async_process_lifecycle_is_persisted_and_contextualized(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)

    started = json.loads(
        runtime._execute_tool(
            "start_process",
            {"statement": "Starting dev server", "command": "sleep 10"},
            RuntimeCallbacks(approval=lambda _request: ApprovalChoice.ALLOW),
            session.path,
        )
    )

    process_id = started["process_id"]
    events = home.read_session_events(session.path)
    assert started["started"] is True
    assert running_process_snapshots(events)[0].process_id == process_id
    assert process_id in runtime._instructions(session.path)
    assert "sleep 10" in runtime._instructions(session.path)

    ended = json.loads(runtime.end_process(process_id, session.path))

    assert ended["ended"] is True
    assert running_process_snapshots(home.read_session_events(session.path)) == ()


def test_operator_long_running_command_is_temporarily_promoted_to_bottom_panel(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(tool_manager_module, "LONG_RUNNING_COMMAND_SECONDS", 0.01)
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)
    statuses: list[str] = []

    output = runtime._execute_tool(
        "run_command",
        {
            "statement": "Installing dependencies",
            "command": "sleep 0.2; echo done",
        },
        RuntimeCallbacks(
            approval=lambda _request: ApprovalChoice.ALLOW,
            status=statuses.append,
        ),
        session.path,
    )

    payload = json.loads(output)
    events = home.read_session_events(session.path)
    process_events = [
        event_payload(event)
        for event in events
        if event_payload_type(event) == "process_event"
    ]

    assert payload["approved"] is True
    assert payload["output"] == "done"
    assert "waiting for long-running command" in statuses
    assert statuses[-1] == "Thinking"
    assert any(
        event.get("source") == "command" and event.get("status") == "running"
        for event in process_events
    )
    assert any(
        event.get("source") == "command" and event.get("status") == "ended"
        for event in process_events
    )
    assert running_process_snapshots(events) == ()


def test_header_box_draws_plan_steps(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 24, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    window = Window()
    steps = (
        PlanStep(1, "Inspect", "Read files", False),
        PlanStep(2, "Validate", "Run checks", True),
    )

    app._draw_header_box(
        window,
        "Session",
        "Location: /repo",
        steps,
        header_meta="abc123 · openai/gpt-5.5",
    )

    assert any(
        text == f"abc123 · openai/gpt-5.5 · v{__version__}"
        for _, _, text, _ in window.writes
    )
    assert any("☐ Inspect" in text for _, _, text, _ in window.writes)
    assert any("☑" in text and "V\u0336" in text for _, _, text, _ in window.writes)
    assert app._session_body_top(steps) == 10
    assert app._session_body_top(steps, subtitle_line_count=2) == 11


def test_approved_command_rows_show_session_allowances(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home)

    assert app._approved_command_rows(session) == (
        InfoRow("Session", "No commands approved yet"),
    )

    app.session_allowed_commands.update({"cmd:python", "cmd:curl"})
    home.append_session_event(
        session.path,
        "work_message",
        {"message": "Approved command: python -V", "role": "tool"},
    )

    assert app._approved_command_rows(session) == (
        InfoRow("Always approved", "curl"),
        InfoRow("Always approved", "python"),
        InfoRow("Approved", "python -V"),
    )


def test_draw_session_registers_click_target_for_collapsed_work_line(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    window = Window()
    message = "Approved command: " + "python -c 'print(\"long command\")' " * 4

    app._draw_session(
        window,
        session,
        [MessageLine("approved", message, expansion_key="line-1")],
        "",
        0,
        0,
    )

    assert any("... click to expand" in text for _, _, text, _ in window.writes)
    assert any(
        action.kind == "toggle_work_line" and action.text == "line-1"
        for actions in app._click_targets.values()
        for action in actions
    )


def test_draw_session_renders_slash_commands_in_bottom_panel(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    window = Window()
    suggestions = app._filtered_commands("/")

    app._draw_session(
        window,
        session,
        [],
        "/",
        1,
        0,
        command_suggestions=suggestions,
        command_selected=3,
    )

    assert any(text == "Commands" for _, _, text, _ in window.writes)
    assert any(text == "Choose a command to run" for _, _, text, _ in window.writes)
    assert any(text == "› /config" for _, _, text, _ in window.writes)


def test_draw_empty_session_renders_starter_skill_hints(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 36, 120

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    window = Window()

    app._draw_session(window, session, [], "", 0, 0)

    assert any(text == "Map the folder" for _, _, text, _ in window.writes)
    assert any(text == "Find issues" for _, _, text, _ in window.writes)
    assert any(text == "Make a report" for _, _, text, _ in window.writes)


def test_session_mouse_action_maps_starter_hint_click_to_skill(tmp_path, monkeypatch):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 36, 120

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    window = Window()

    app._draw_session(window, session, [], "", 0, 0)
    target_y, target_action = next(
        (y, action)
        for y, actions in app._click_targets.items()
        for action in actions
        if action.kind == "skill" and action.text == "map-folder"
    )
    monkeypatch.setattr(
        curses,
        "getmouse",
        lambda: (0, target_action.x_start, target_y, 0, curses.BUTTON1_CLICKED),
    )

    action = app._session_mouse_action(window, "", [])

    assert action is not None
    assert action.kind == "skill"
    assert action.text == "map-folder"


def test_run_session_invokes_clicked_starter_skill(tmp_path, monkeypatch):
    class Window:
        def __init__(self):
            self._keys = iter((curses.KEY_MOUSE,))

        def get_wch(self):
            return next(self._keys)

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    invoked: list[tuple[str, str]] = []

    monkeypatch.setattr(app, "_draw_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        app,
        "_session_mouse_action",
        lambda *_args, **_kwargs: SessionMouseAction("skill", 0, "map-folder"),
    )

    def record_skill(_stdscr, _session, skill, submitted):
        invoked.append((skill.command, submitted))
        return "exit"

    monkeypatch.setattr(app, "_invoke_skill", record_skill)

    assert app._run_session(Window(), session) == 0
    assert invoked == [("map-folder", "/map-folder")]


def test_session_mouse_action_maps_bottom_panel_command_click(tmp_path, monkeypatch):
    class Window:
        def getmaxyx(self):
            return 28, 80

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    suggestions = app._filtered_commands("/")
    panel = app._command_bottom_panel(suggestions, selected=0)
    assert panel is not None

    viewport = app._bottom_panel_viewport(Window(), panel, input_text="/")
    choice_y = viewport.choice_y + (1 if viewport.show_overflow_counts else 0)
    monkeypatch.setattr(
        curses,
        "getmouse",
        lambda: (0, 4, choice_y + 1, 0, curses.BUTTON1_CLICKED),
    )

    action = app._session_mouse_action(Window(), "/", suggestions)

    assert action is not None
    assert action.kind == "command"
    assert action.value == 1


def test_running_workers_render_at_bottom(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 24, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n]))

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    window = Window()

    app._draw_running_workers(
        window,
        (
            worker_snapshots(
                [
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "worker_event",
                            "worker_id": "abc123",
                            "name": "Engineer",
                            "status": "running",
                            "statement": "Reading files",
                            "started_at": runtime_module.utc_now_iso(),
                        },
                    }
                ]
            )[0],
        ),
        start_y=20,
        frame=12,
    )

    assert any(
        text == "Engineer (abc123) · Reading files..." for _, _, text in window.writes
    )
    assert any(x > 60 and text.count(":") == 1 for _, x, text in window.writes)


def test_running_worker_initial_thinking_renders_with_activity_dots(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 24, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n]))

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    window = Window()

    app._draw_running_workers(
        window,
        (
            worker_snapshots(
                [
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "worker_event",
                            "worker_id": "abc123",
                            "name": "Engineer",
                            "status": "running",
                            "statement": "thinking",
                            "started_at": runtime_module.utc_now_iso(),
                        },
                    }
                ]
            )[0],
        ),
        start_y=20,
        frame=12,
    )

    assert any("Engineer (abc123) · Thinking..." in text for _, _, text in window.writes)


def test_ready_and_interrupted_workers_render_state_without_activity(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 24, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n]))

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    window = Window()

    app._draw_running_workers(
        window,
        worker_snapshots(
            [
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "worker_event",
                        "worker_id": "ready1",
                        "name": "Engineer",
                        "status": "ready",
                        "statement": "Hidden work title",
                    },
                },
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "worker_event",
                        "worker_id": "stop1",
                        "name": "Reviewer",
                        "status": "interrupted",
                        "statement": "Also hidden",
                    },
                },
            ]
        ),
        start_y=20,
        frame=12,
    )

    assert any(text == "Engineer (ready1)" for _, _, text in window.writes)
    assert any(text == "Reviewer (stop1)" for _, _, text in window.writes)
    assert not any(
        "Hidden work title" in text or "Also hidden" in text
        for _, _, text in window.writes
    )
    assert any(x > 60 and text == "Ready" for _, x, text in window.writes)
    assert any(x > 60 and text == "Interrupted" for _, x, text in window.writes)


def test_running_process_renders_click_to_kill_target(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 24, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    window = Window()
    process = running_process_snapshots(
        [
            {
                "type": "event_msg",
                "payload": {
                    "type": "process_event",
                    "process_id": "proc123",
                    "command": "npm run dev",
                    "statement": "Starting dev server",
                    "status": "running",
                    "started_at": runtime_module.utc_now_iso(),
                },
            }
        ]
    )[0]

    app._draw_running_workers(window, (), start_y=20, frame=12, processes=(process,))

    assert any(
        "Process (proc123) · Starting dev server" in text
        for _, _, text, _ in window.writes
    )
    assert any(
        x > 60 and text == "Click to kill"
        for _, x, text, _ in window.writes
    )
    assert any(
        action.kind == "kill_process"
        and action.text == "proc123"
        and action.x_end > action.x_start
        for actions in app._click_targets.values()
        for action in actions
    )


def test_observer_mode_rejection_only_feedbacks_model(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), repo, mode=AgentMode.OBSERVER)
    events: list[tuple[str, str]] = []

    output = runtime._execute_tool(
        "run_command",
        {"statement": "Checking Python", "command": "python3 -V"},
        RuntimeCallbacks(system_message=lambda role, message: events.append((role, message))),
    )

    payload = json.loads(output)
    assert payload["approved"] is False
    assert "view the repo" in payload["output"]
    assert events == []


def test_command_manager_classifies_allow_approve_forbidden(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    manager = CliToolManager(repo)

    assert manager.classify("pwd").safety == CommandSafety.ALLOW
    assert manager.classify("cat pyproject.toml").safety == CommandSafety.ALLOW
    assert manager.classify("grep TODO pyproject.toml").safety == CommandSafety.ALLOW
    assert manager.classify("find . -maxdepth 2").safety == CommandSafety.ALLOW
    assert manager.classify("find . -maxdepth 2 | sort").safety == CommandSafety.ALLOW
    assert manager.classify("git status; git log --oneline -5").safety == CommandSafety.ALLOW
    assert manager.classify("git status 2>/dev/null").safety == CommandSafety.ALLOW
    assert (
        manager.classify('cat README.md 2>/dev/null || echo "No README.md"').safety
        == CommandSafety.ALLOW
    )
    assert manager.classify("find . -delete").safety == CommandSafety.APPROVE
    assert manager.classify("rm README.md").safety == CommandSafety.APPROVE
    assert manager.classify("cat > note.txt").safety == CommandSafety.APPROVE
    assert manager.classify("rg anomaly").safety == CommandSafety.ALLOW
    assert manager.classify("rg --pre sh anomaly").safety == CommandSafety.APPROVE
    assert manager.classify("sed -n 1,5p pyproject.toml").safety == CommandSafety.ALLOW
    assert manager.classify("reboot").safety == CommandSafety.FORBIDDEN


def test_command_manager_always_allow_is_session_scoped(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("hello", encoding="utf-8")
    allowed: set[str] = set()
    manager = CliToolManager(repo, allowed)

    first = manager.run_command(
        "python -V",
        "I am checking Python",
        lambda _request: ApprovalChoice.ALWAYS_ALLOW,
    )
    second = manager.classify("python -V")

    assert first.approved is True
    assert "cmd:python" in allowed
    assert "python -V" not in allowed
    assert second.safety == CommandSafety.ALLOW


def test_command_manager_always_allow_trusts_command_family(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "source.txt").write_text("hello", encoding="utf-8")
    allowed: set[str] = set()
    manager = CliToolManager(repo, allowed)

    first = manager.run_command(
        "cat source.txt > first.txt",
        "Writing file",
        lambda _request: ApprovalChoice.ALWAYS_ALLOW,
    )
    second = manager.classify("cat source.txt > second.txt")
    outside = manager.classify("cat ../secret.txt > second.txt")

    assert first.approved is True
    assert "cmd:cat" in allowed
    assert "cat source.txt > first.txt" not in allowed
    assert second.safety == CommandSafety.ALLOW
    assert outside.safety == CommandSafety.FORBIDDEN


def test_command_manager_always_reject_blocks_command_family_for_session(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    rejected: set[str] = set()
    manager = CliToolManager(repo, session_rejected_commands=rejected)

    first = manager.run_command(
        "curl -s https://example.com",
        "Fetching URL",
        lambda _request: ApprovalChoice.ALWAYS_REJECT,
    )
    second = manager.classify("curl -s https://example.com/api")

    assert first.approved is False
    assert first.safety == CommandSafety.FORBIDDEN
    assert "cmd:curl" in rejected
    assert second.safety == CommandSafety.FORBIDDEN
    assert second.reason == "curl is blocked for this session by user policy."


def test_command_manager_modes_control_approval(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("hello", encoding="utf-8")

    observer = CliToolManager(repo, mode=AgentMode.OBSERVER)
    observer_read = observer.run_command("cat README.md", "Reading README", None)
    observer_write = observer.run_command(
        "python3 -V",
        "Checking Python",
        lambda _request: ApprovalChoice.ALLOW,
    )

    assert observer_read.approved is True
    assert observer_read.output == "hello"
    assert observer_write.approved is False
    assert observer_write.blocked_by_mode is True

    autonomous = CliToolManager(repo, mode=AgentMode.AUTONOMOUS)
    autonomous_python = autonomous.run_command("python3 -V", "Checking Python", None)
    autonomous_dangerous = autonomous.run_command("reboot", "Restarting host", None)

    assert autonomous_python.approved is True
    assert autonomous_python.safety == CommandSafety.ALLOW
    assert autonomous_dangerous.approved is False
    assert autonomous_dangerous.safety == CommandSafety.FORBIDDEN


def test_command_manager_handles_non_utf8_subprocess_output(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    manager = CliToolManager(repo)

    output = manager._execute_subprocess(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.stdout.buffer.write(b'\\x93stdout'); "
                "sys.stderr.buffer.write(b'\\x94stderr')"
            ),
        ]
    )

    assert "\\x93stdout" in output
    assert "\\x94stderr" in output
