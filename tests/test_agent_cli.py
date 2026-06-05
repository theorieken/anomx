import curses
import json
import queue
import stat
import tomllib
from pathlib import Path

import anomx.agent.runtime as runtime_module
from anomx.agent import AnomxHome
from anomx.agent.mode import AgentMode
from anomx.agent.runtime import AgentRole, AgentRuntime, RuntimeCallbacks
from anomx.agent.state import (
    PlanStep,
    latest_plan_steps,
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
from anomx.agent.terminal import markdown_to_terminal_lines
from anomx.agent.tool_manager import (
    ApprovalChoice,
    CliToolManager,
    CommandApprovalRequest,
    CommandSafety,
)
from anomx.agent.ui import (
    AgentState,
    AnomxCliApp,
    BottomPanel,
    MenuChoice,
    MessageLine,
    RuntimeUiEvent,
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


def test_slash_commands_filter_to_best_five(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    all_commands = app._filtered_commands("/")
    model_commands = app._filtered_commands("/mo")

    assert len(all_commands) <= 5
    assert [command.command for command in all_commands] == [
        "/new",
        "/session",
        "/config",
        "/model",
        "/exit",
    ]
    assert [command.command for command in model_commands] == ["/model"]


def test_submitted_slash_command_prefers_exact_command(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    suggestions = app._filtered_commands("/")

    assert app._submitted_command("/config", suggestions, selected=0) == "/config"
    assert app._submitted_command("/session", suggestions, selected=0) == "/session"


def test_config_menu_shows_only_requested_entries(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    choices = app._config_menu_choices()

    assert [(choice.label, choice.value, choice.detail) for choice in choices] == [
        ("Choose backend", "backend", "Select provider and enter API key"),
        ("Choose model", "model", "Pick the model for the selected backend"),
        ("History persistence", "history_persistence", "Store all sessions or none"),
        ("Require trusted repo", "require_trusted_repo", "Ask before using untrusted folders"),
        ("Clear all sessions", "clear_sessions", "Delete stored sessions except this one"),
        ("Done", "done", "Same as Esc"),
    ]


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

    def record_command(_stdscr, command, _current_session):
        executed.append(command)
        return "exit"

    monkeypatch.setattr(app, "_handle_command", record_command)

    assert app._run_session(Window(), session) == 0
    assert executed == ["/session"]


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


def test_agent_mode_cycles_and_updates_runtime(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    app._cycle_agent_mode()
    assert app.agent_mode == AgentMode.AUTONOMOUS
    assert app.runtime.tool_manager.mode == AgentMode.AUTONOMOUS
    assert app._mode_hint_attr_name() == "warning"

    app._cycle_agent_mode()
    assert app.agent_mode == AgentMode.OBSERVER
    assert app.runtime.tool_manager.mode == AgentMode.OBSERVER
    assert app._mode_hint_attr_name() == "light"


def test_prompt_cursor_stays_on_previous_line_at_wrap_boundary(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    assert app._prompt_cursor_position("abcdefghij", cursor=10, width=10) == (0, 9)


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
        MessageLine("approved", "Approved command: cat README.md"),
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
        "check_agent",
        "stop_agent",
        "create_plan",
        "update_plan",
        "remove_plan",
    }
    assert expected_operator_tools.issubset(set(operator_names))
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
    assert snapshots[-1].status == "finished"
    assert snapshots[-1].statement == "finished"
    assert running_worker_snapshots(events) == ()
    assert not any(line.role == "worker" and "Worker report" in line.text for line in messages)
    assert checked["commands"] == [
        {
            "statement": "Listing files",
            "command": "ls",
            "output": "README.md",
        }
    ]


def test_header_box_draws_plan_steps(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 24, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n]))

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    window = Window()
    steps = (
        PlanStep(1, "Inspect", "Read files", False),
        PlanStep(2, "Validate", "Run checks", True),
    )

    app._draw_header_box(window, "Session", "abc123 · openai/gpt-5.5", steps)

    assert any("☐ Inspect" in text for _, _, text in window.writes)
    assert any("☑" in text and "V\u0336" in text for _, _, text in window.writes)
    assert app._session_body_top(steps) == 10


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
        frame=0,
    )

    assert any("Engineer (abc123) · Reading files" in text for _, _, text in window.writes)
    assert any(x > 60 and text.count(":") == 1 for _, x, text in window.writes)


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
