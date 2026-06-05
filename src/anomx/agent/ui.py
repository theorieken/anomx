"""Full-screen terminal UI for the Anomx agent."""

from __future__ import annotations

import curses
import math
import queue
import random
import textwrap
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from anomx import __version__
from anomx.agent.mode import AgentMode
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks, StatusCallback
from anomx.agent.state import (
    PlanStep,
    WorkerAgentSnapshot,
    latest_plan_steps,
    running_worker_snapshots,
)
from anomx.agent.store import (
    AI_PROVIDERS,
    AnomxHome,
    ProviderOption,
    SessionRecord,
    model_context_window,
    model_detail,
    provider_by_key,
)
from anomx.agent.terminal import markdown_to_terminal_lines
from anomx.agent.tool_manager import ApprovalChoice, CommandApprovalRequest


class AgentState(StrEnum):
    """Top-level CLI states."""

    ONBOARDING = "Onboarding"
    ACCESS_CHECK = "Access Check"
    NEW_SESSION = "New Session"
    OPEN_SESSION = "Open Session"
    CONFIG = "Config"
    MODEL = "Model"
    EXIT = "Exit"


class CursesWindow(Protocol):
    """Subset of curses window methods used by the UI."""

    def erase(self) -> None: ...
    def refresh(self) -> None: ...
    def getmaxyx(self) -> tuple[int, int]: ...
    def addnstr(self, y: int, x: int, text: str, n: int, attr: int = 0) -> None: ...
    def get_wch(self) -> str | int: ...
    def keypad(self, flag: bool) -> None: ...
    def move(self, y: int, x: int) -> None: ...
    def clrtoeol(self) -> None: ...
    def bkgd(self, ch: str, attr: int = 0) -> None: ...
    def nodelay(self, flag: bool) -> None: ...


@dataclass(frozen=True)
class MenuChoice:
    """Selectable full-screen menu item."""

    label: str
    value: str
    detail: str = ""


@dataclass(frozen=True)
class MessageLine:
    """Rendered session transcript line."""

    role: str
    text: str
    meta: str = ""


@dataclass(frozen=True)
class CommandSpec:
    """Prompt command entry."""

    command: str
    description: str


@dataclass(frozen=True)
class BottomPanel:
    """Inline command panel rendered above the prompt bar."""

    title: str
    subtitle: str
    choices: tuple[MenuChoice, ...]
    selected: int


@dataclass(frozen=True)
class BottomPanelViewport:
    """Visible bottom-panel rows for the current selection."""

    start_y: int
    subtitle_lines: tuple[str, ...]
    choice_y: int
    visible_indices: tuple[int, ...]
    more_above: int
    more_below: int
    show_overflow_counts: bool


@dataclass(frozen=True)
class RuntimeUiEvent:
    """Event emitted by the runtime worker for the TUI loop."""

    kind: str
    text: str = ""
    role: str = ""
    approval_request: CommandApprovalRequest | None = None
    approval_response: queue.SimpleQueue[ApprovalChoice] | None = None


@dataclass(frozen=True)
class PromptLayout:
    """Prompt bar geometry."""

    top_line: int
    prompt_line: int
    bottom_line: int
    hint_line: int
    input_x: int
    input_width: int
    prompt_height: int


@dataclass(frozen=True)
class SessionMouseAction:
    """Mouse action resolved from a curses mouse event."""

    kind: str
    value: int
    text: str = ""


@dataclass(frozen=True)
class BackendTurnResult:
    """State returned from a non-blocking backend turn."""

    input_text: str
    cursor: int
    exit_requested: bool = False
    anchor_line: int | None = None
    scroll: int = 0


@dataclass(frozen=True)
class RunningKeyResult:
    """Input state update while a backend turn is active."""

    input_text: str
    cursor: int
    notice: str
    notice_role: str
    abort_key: str
    abort_deadline: float
    exit_requested: bool = False
    scroll_delta: int = 0
    clear_anchor: bool = False


PROMPT_PLACEHOLDERS = (
    "Let's find the anomalies in the data of this repo",
    "Inspect this repository and map the data pipeline",
    "Look for time-series signals that need anomaly detection",
    "Summarize the datasets and suggest detection jobs",
)

COMMANDS = (
    CommandSpec("/new", "Start a new session"),
    CommandSpec("/session", "Open a stored session"),
    CommandSpec("/config", "Edit configuration"),
    CommandSpec("/model", "Change model"),
    CommandSpec("/exit", "Exit Anomx"),
)

RUNNING_NOTICE = "Agent is working · Enter disabled · Esc/Ctrl+C interrupt"


class AnomxCliApp:
    """Full-screen state-machine based Anomx agent."""

    def __init__(
        self,
        home: AnomxHome | None = None,
        cwd: Path | None = None,
        startup_provider: str | None = None,
        startup_model: str | None = None,
        use_color: bool = True,
    ) -> None:
        self.home = AnomxHome() if home is None else home
        self.cwd = Path.cwd() if cwd is None else cwd
        self.startup_provider = startup_provider
        self.startup_model = startup_model
        self.use_color = use_color
        self.session_allowed_commands: set[str] = set()
        self.session_rejected_commands: set[str] = set()
        self.agent_mode = AgentMode.CONFIRM
        self.runtime = AgentRuntime(
            self.home,
            self.cwd,
            self.session_allowed_commands,
            self.session_rejected_commands,
            self.agent_mode,
        )
        self.state = AgentState.ONBOARDING
        self._colors: dict[str, int] = {}
        self._prompt_placeholder = random.choice(PROMPT_PLACEHOLDERS)
        self._expanded_work_turns: set[str] = set()
        self._click_targets: dict[int, str] = {}
        self._title_events: queue.SimpleQueue[tuple[str, str]] = queue.SimpleQueue()
        self._title_jobs: set[str] = set()

    def prepare_startup_config(self) -> None:
        """Apply command-line startup provider/model overrides."""

        self.home.ensure()
        provider_key = self.startup_provider
        model = self.startup_model
        if provider_key is None and model is None:
            return

        config = self.home.load_config()
        if provider_key is None:
            provider_key = str(config.get("provider") or "ollama")
        provider = provider_by_key(provider_key)
        if provider is None:
            raise ValueError(f"unknown provider: {provider_key}")

        selected_model = model or str(config.get("model") or provider.models[0])
        if not self._model_allowed(provider, selected_model):
            selected_model = provider.models[0]

        config["onboarding_complete"] = True
        config["provider"] = provider.key
        config["model"] = selected_model
        self.home.save_config(config)

    def run(self) -> int:
        """Run the full-screen terminal UI."""

        self.prepare_startup_config()
        return int(curses.wrapper(self._run))

    def _run(self, stdscr: CursesWindow) -> int:
        self._configure_terminal(stdscr)

        while True:
            config = self.home.load_config()
            if not self._onboarding_complete(config):
                if not self._run_onboarding(stdscr):
                    return 1
                config = self.home.load_config()

            should_check_access = bool(config.get("require_trusted_repo", True))
            should_check_access = should_check_access and not self.home.is_repo_trusted(self.cwd)
            if should_check_access and not self._run_access_check(stdscr):
                return 1

            self.state = AgentState.NEW_SESSION
            session = self._create_session()
            return self._run_session(stdscr, session)

    def _configure_terminal(self, stdscr: CursesWindow) -> None:
        stdscr.keypad(True)
        curses.noecho()
        with suppress(curses.error):
            curses.raw()
        with suppress(curses.error):
            curses.set_escdelay(120)
        with suppress(curses.error):
            curses.curs_set(0)
        with suppress(curses.error):
            curses.mousemask(curses.ALL_MOUSE_EVENTS)
            curses.mouseinterval(0)
        if self.use_color and curses.has_colors():
            curses.start_color()
            curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)
            curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
            curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(5, curses.COLOR_RED, curses.COLOR_BLACK)
            curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLACK)
            curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLACK)
            self._colors = {
                "accent": curses.color_pair(1) | curses.A_BOLD,
                "selected": curses.color_pair(2) | curses.A_BOLD,
                "background": curses.color_pair(7),
                "muted": curses.color_pair(7) | curses.A_DIM,
                "light": curses.color_pair(6) | curses.A_DIM,
                "warning": curses.color_pair(3) | curses.A_BOLD,
                "ok": curses.color_pair(4) | curses.A_BOLD,
                "danger": curses.color_pair(5) | curses.A_BOLD,
                "bold": curses.color_pair(6) | curses.A_BOLD,
            }
        else:
            self._colors = {
                "accent": curses.A_BOLD,
                "selected": curses.A_REVERSE,
                "background": curses.A_NORMAL,
                "light": curses.A_DIM,
                "muted": curses.A_DIM,
                "warning": curses.A_BOLD,
                "ok": curses.A_BOLD,
                "danger": curses.A_BOLD,
                "bold": curses.A_BOLD,
            }
        with suppress(curses.error):
            stdscr.bkgd(" ", self._attr("background"))

    def _run_onboarding(self, stdscr: CursesWindow) -> bool:
        self.state = AgentState.ONBOARDING
        provider = self._select_provider(stdscr)
        if provider is None:
            return False

        model = self._select_model(stdscr, provider)
        if model is None:
            return False

        config = self.home.load_config()
        config["onboarding_complete"] = True
        config["provider"] = provider.key
        config["model"] = model
        self.home.save_config(config)

        if provider.key in {"openai", "anthropic", "desy"}:
            api_key = self._prompt_text(
                stdscr,
                title=provider.label,
                label="API key",
                mask=True,
                optional=True,
            )
            if api_key:
                self.home.set_api_key(provider.key, api_key)

        return True

    def _run_access_check(self, stdscr: CursesWindow) -> bool:
        self.state = AgentState.ACCESS_CHECK
        selected = 0
        while True:
            self._draw_access_check(stdscr, selected)
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key):
                return False
            if key == curses.KEY_UP:
                selected = max(0, selected - 1)
            elif key == curses.KEY_DOWN:
                selected = min(1, selected + 1)
            elif self._is_enter(key):
                if selected == 0:
                    self.home.trust_repo(self.cwd)
                    return True
                return False

    def _create_session(self) -> SessionRecord:
        config = self.home.load_config()
        return self.home.create_session(
            self.cwd,
            provider=str(config.get("provider", "openai")),
            model=str(config.get("model", "gpt-5.5")),
        )

    def _run_session(self, stdscr: CursesWindow, session: SessionRecord) -> int:
        current_session = session
        input_text = ""
        cursor = 0
        scroll = 0
        command_selected = 0
        pinned_anchor: int | None = None

        while True:
            current_session = self._process_title_events(stdscr, current_session)
            messages = self._read_message_lines(current_session.path)
            command_suggestions = (
                self._filtered_commands(input_text) if input_text.startswith("/") else []
            )
            if command_suggestions:
                command_selected = min(command_selected, len(command_suggestions) - 1)
            else:
                command_selected = 0
            self._draw_session(
                stdscr,
                current_session,
                messages,
                input_text,
                cursor,
                scroll,
                command_suggestions,
                command_selected,
                anchor_line=pinned_anchor,
            )
            key = stdscr.get_wch()

            if self._is_shift_tab(key):
                self._cycle_agent_mode()
                continue
            if self._is_ctrl_c(key):
                return 0
            if self._is_escape(key):
                if input_text:
                    input_text = ""
                    cursor = 0
                    command_selected = 0
                elif pinned_anchor is not None:
                    pinned_anchor = None
                continue

            if key == curses.KEY_UP:
                if command_suggestions:
                    command_selected = max(0, command_selected - 1)
                    continue
                pinned_anchor = None
                scroll += 1
                continue
            if key == curses.KEY_DOWN:
                if command_suggestions:
                    command_selected = min(len(command_suggestions) - 1, command_selected + 1)
                    continue
                pinned_anchor = None
                scroll = max(0, scroll - 1)
                continue
            if key == curses.KEY_PPAGE:
                pinned_anchor = None
                scroll += 5
                continue
            if key == curses.KEY_NPAGE:
                pinned_anchor = None
                scroll = max(0, scroll - 5)
                continue
            if key == curses.KEY_LEFT:
                cursor = max(0, cursor - 1)
                continue
            if key == curses.KEY_RIGHT:
                cursor = min(len(input_text), cursor + 1)
                continue
            if key == curses.KEY_HOME:
                cursor = 0
                continue
            if key == curses.KEY_END:
                cursor = len(input_text)
                continue
            if key == curses.KEY_MOUSE:
                mouse_action = self._session_mouse_action(
                    stdscr,
                    input_text,
                    command_suggestions,
                )
                if mouse_action is None:
                    continue
                if mouse_action.kind == "cursor":
                    cursor = mouse_action.value
                elif mouse_action.kind == "scroll":
                    pinned_anchor = None
                    scroll = max(0, scroll + mouse_action.value)
                elif mouse_action.kind == "toggle_work":
                    self._toggle_work_turn(mouse_action.text)
                elif mouse_action.kind == "command":
                    command = command_suggestions[mouse_action.value].command
                    command_result = self._handle_command(stdscr, command, current_session)
                    if command_result == "exit":
                        return 0
                    if isinstance(command_result, SessionRecord):
                        current_session = command_result
                        self._prompt_placeholder = random.choice(PROMPT_PLACEHOLDERS)
                        scroll = 0
                        pinned_anchor = None
                    input_text = ""
                    cursor = 0
                    command_selected = 0
                continue
            if self._is_enter(key):
                submitted = input_text.strip()
                selected_command = command_selected
                input_text = ""
                cursor = 0
                command_selected = 0
                if not submitted:
                    continue
                if submitted.startswith("/"):
                    command = self._submitted_command(
                        submitted,
                        command_suggestions,
                        selected_command,
                    )
                    command_result = self._handle_command(stdscr, command, current_session)
                    if command_result == "exit":
                        return 0
                    if isinstance(command_result, SessionRecord):
                        current_session = command_result
                        self._prompt_placeholder = random.choice(PROMPT_PLACEHOLDERS)
                        scroll = 0
                        pinned_anchor = None
                    continue
                self.home.append_session_event(
                    current_session.path,
                    "user_message",
                    {"message": submitted},
                )
                self._maybe_start_session_rename(current_session)
                anchor_line = self._latest_user_anchor_line(stdscr, current_session)
                self._animate_message_anchor(stdscr, current_session, anchor_line)
                turn_result = self._run_backend_turn(
                    stdscr,
                    current_session,
                    anchor_line=anchor_line,
                )
                if turn_result.exit_requested:
                    return 0
                input_text = turn_result.input_text
                cursor = turn_result.cursor
                pinned_anchor = turn_result.anchor_line
                scroll = turn_result.scroll
                continue

            if self._is_backspace(key):
                if cursor > 0:
                    input_text = input_text[: cursor - 1] + input_text[cursor:]
                    cursor -= 1
                    command_selected = 0
                continue
            if isinstance(key, str) and key.isprintable():
                input_text = input_text[:cursor] + key + input_text[cursor:]
                cursor += len(key)
                command_selected = 0

    def _maybe_start_session_rename(self, session: SessionRecord) -> None:
        if session.title != "New session" or not self._has_user_messages(session.path):
            return
        if session.session_id in self._title_jobs:
            return

        self._title_jobs.add(session.session_id)

        def suggest_title() -> None:
            title = self.runtime.suggest_session_title(session.path)
            if title is None or title == session.title:
                return
            self.home.update_session_title(session.path, title)
            self._title_events.put((session.session_id, title))

        threading.Thread(target=suggest_title, daemon=True).start()

    def _process_title_events(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
    ) -> SessionRecord:
        current_session = session
        while True:
            try:
                session_id, title = self._title_events.get_nowait()
            except queue.Empty:
                return current_session
            if session_id != current_session.session_id:
                continue
            self._animate_session_title(stdscr, current_session, title)
            current_session = replace(current_session, title=title)

    def _animate_session_title(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        new_title: str,
    ) -> None:
        messages = self._read_message_lines(session.path)
        old_title = session.title
        for index in range(len(old_title), -1, -1):
            self._draw_session(
                stdscr,
                replace(session, title=old_title[:index]),
                messages,
                "",
                0,
                0,
            )
            time.sleep(0.012)
        for index in range(1, len(new_title) + 1):
            self._draw_session(
                stdscr,
                replace(session, title=new_title[:index]),
                messages,
                "",
                0,
                0,
            )
            time.sleep(0.012)

    def _has_user_messages(self, session_path: Path) -> bool:
        for event in self.home.read_session_events(session_path):
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            event_type = (
                payload.get("type") if event.get("type") == "event_msg" else event.get("type")
            )
            if event_type == "user_message" and str(payload.get("message", "")).strip():
                return True
        return False

    def _handle_command(
        self,
        stdscr: CursesWindow,
        command: str,
        current_session: SessionRecord,
    ) -> str | SessionRecord | None:
        if command == "/exit":
            return "exit"
        if command == "/session":
            return self._open_session_panel(stdscr, current_session)
        if command == "/config":
            self._run_config_panel(stdscr, current_session)
            return None
        if command == "/model":
            self._run_model_panel(stdscr, current_session)
            return None
        if command == "/new":
            return self._create_session()

        self._message(stdscr, "Unknown Command", command)
        return current_session

    def _open_session_panel(
        self,
        stdscr: CursesWindow,
        current_session: SessionRecord,
    ) -> SessionRecord | None:
        self.state = AgentState.OPEN_SESSION
        sessions = self.home.list_sessions(limit=None)
        if not sessions:
            self._message(stdscr, "Open Session", "No stored sessions")
            self.state = AgentState.NEW_SESSION
            return None

        choices = tuple(
            MenuChoice(
                label=session.title,
                detail=f"{session.created_at} · {session.provider}/{session.model}",
                value=str(index),
            )
            for index, session in enumerate(sessions)
        )
        selected = self._bottom_menu(
            stdscr,
            current_session,
            "Open Session",
            "Choose a stored session",
            choices,
        )
        self.state = AgentState.NEW_SESSION
        return sessions[int(selected)] if selected is not None else None

    def _run_model_panel(self, stdscr: CursesWindow, current_session: SessionRecord) -> bool:
        self.state = AgentState.MODEL
        config = self.home.load_config()
        provider = provider_by_key(str(config.get("provider", "openai"))) or AI_PROVIDERS[0]
        choices = [MenuChoice(model, model, model_detail(model)) for model in provider.models]
        if provider.allow_custom_model:
            choices.append(
                MenuChoice(
                    "Custom model",
                    "__custom__",
                    f"Use a custom {provider.label} model name",
                )
            )
        selected = self._bottom_menu(
            stdscr,
            current_session,
            "Model",
            f"Provider: {provider.label}",
            tuple(choices),
        )
        if selected is None:
            self.state = AgentState.NEW_SESSION
            return False
        model = (
            self._prompt_text(stdscr, "Model", "Model name", optional=False)
            if selected == "__custom__"
            else selected
        )
        if model:
            config["provider"] = provider.key
            config["model"] = model
            config["onboarding_complete"] = True
            self.home.save_config(config)
        self.state = AgentState.NEW_SESSION
        return bool(model)

    def _run_config_panel(self, stdscr: CursesWindow, current_session: SessionRecord) -> None:
        self.state = AgentState.CONFIG

        while True:
            config = self.home.load_config()
            choices = self._config_menu_choices()
            selected = self._bottom_menu(
                stdscr,
                current_session,
                "Config",
                "Choose a setting to change",
                choices,
            )
            if selected is None:
                self.state = AgentState.NEW_SESSION
                return
            if selected == "done":
                self.state = AgentState.NEW_SESSION
                return
            if selected == "backend":
                if self._configure_backend(stdscr):
                    self.state = AgentState.NEW_SESSION
                    return
                continue
            if selected == "model":
                if self._run_model_panel(stdscr, current_session):
                    self.state = AgentState.NEW_SESSION
                    return
                self.state = AgentState.CONFIG
                continue
            if selected == "require_trusted_repo":
                config["require_trusted_repo"] = not bool(config.get("require_trusted_repo"))
                self.home.save_config(config)
                continue
            if selected == "history_persistence":
                value = self._select_history_persistence(stdscr, current_session, config)
                if value is not None:
                    config["history_persistence"] = value
                    self.home.save_config(config)
                continue
            if selected == "clear_sessions":
                if self._confirm_clear_sessions(stdscr, current_session):
                    self.home.clear_sessions(keep_session_path=current_session.path)
                continue

    def _config_menu_choices(self) -> tuple[MenuChoice, ...]:
        return (
            MenuChoice("Choose backend", "backend", "Select provider and enter API key"),
            MenuChoice("Choose model", "model", "Pick the model for the selected backend"),
            MenuChoice("History persistence", "history_persistence", "Store all sessions or none"),
            MenuChoice(
                "Require trusted repo",
                "require_trusted_repo",
                "Ask before using untrusted folders",
            ),
            MenuChoice(
                "Clear all sessions",
                "clear_sessions",
                "Delete stored sessions except this one",
            ),
            MenuChoice("Done", "done", "Same as Esc"),
        )

    def _configure_backend(self, stdscr: CursesWindow) -> bool:
        config = self.home.load_config()
        previous_provider = str(config.get("provider", ""))
        provider = self._select_provider(stdscr)
        if provider is None:
            return False
        if provider.key in {"openai", "anthropic", "desy"}:
            should_prompt_api_key = True
            if self.home.has_api_key(provider.key):
                selected = self._menu(
                    stdscr,
                    provider.label,
                    "API key already configured",
                    (
                        MenuChoice("Keep API Key", "keep", "Use the saved API key"),
                        MenuChoice("New API Key", "new", "Replace the saved API key"),
                    ),
                )
                if selected is None:
                    return False
                should_prompt_api_key = selected == "new"
            if should_prompt_api_key:
                api_key = self._prompt_text(
                    stdscr,
                    title=provider.label,
                    label="API key",
                    mask=True,
                    optional=False,
                )
                if not api_key:
                    return False
                self.home.set_api_key(provider.key, api_key)
        selected_model = str(config.get("model", ""))
        if provider.key != previous_provider:
            model = self._select_model(stdscr, provider)
            if model is None:
                return False
            selected_model = model
        elif not self._model_allowed(provider, selected_model):
            selected_model = provider.models[0]
        config["provider"] = provider.key
        config["model"] = selected_model
        self.home.save_config(config)
        return True

    def _select_history_persistence(
        self,
        stdscr: CursesWindow,
        current_session: SessionRecord,
        config: dict[str, object],
    ) -> str | None:
        selected = str(config.get("history_persistence", "save_all"))
        choices = (
            MenuChoice("Save all sessions", "save_all"),
            MenuChoice("Do not save sessions", "none"),
        )
        return self._bottom_menu(
            stdscr,
            current_session,
            "History Persistence",
            "Choose how session history should be stored",
            choices,
        ) or selected

    def _confirm_clear_sessions(
        self,
        stdscr: CursesWindow,
        current_session: SessionRecord,
    ) -> bool:
        selected = self._bottom_menu(
            stdscr,
            current_session,
            "Clear All Sessions",
            "Delete stored sessions and keep only the current open session",
            (
                MenuChoice("Cancel", "cancel"),
                MenuChoice("Clear all sessions", "confirm"),
            ),
        )
        return selected == "confirm"

    def _select_provider(self, stdscr: CursesWindow) -> ProviderOption | None:
        choices = tuple(
            MenuChoice(provider.label, provider.key, ", ".join(provider.models))
            for provider in AI_PROVIDERS
        )
        selected = self._menu(stdscr, "AI Backend", "Select provider", choices)
        return provider_by_key(selected) if selected is not None else None

    def _select_model(self, stdscr: CursesWindow, provider: ProviderOption) -> str | None:
        choices = [MenuChoice(model, model, model_detail(model)) for model in provider.models]
        if provider.allow_custom_model:
            choices.append(
                MenuChoice(
                    "Custom model",
                    "__custom__",
                    f"Use a custom {provider.label} model name",
                )
            )
        selected = self._menu(stdscr, "Model", provider.label, tuple(choices))
        if selected is None:
            return None
        if selected == "__custom__":
            return self._prompt_text(stdscr, "Model", "Model name", optional=False)
        return selected

    def _menu(
        self,
        stdscr: CursesWindow,
        title: str,
        subtitle: str,
        choices: tuple[MenuChoice, ...],
    ) -> str | None:
        selected = 0
        while True:
            height, width = self._draw_shell(stdscr, title, subtitle)
            start_y = max(7, (height - len(choices)) // 2)
            for index, choice in enumerate(choices):
                attr = self._attr("selected") if index == selected else curses.A_NORMAL
                label = f" {choice.label}"
                self._add(stdscr, start_y + index, 4, label, width - 8, attr)
                if choice.detail:
                    detail_x = min(width - 4, 32)
                    self._add(
                        stdscr,
                        start_y + index,
                        detail_x,
                        choice.detail,
                        width - detail_x - 4,
                    )
            self._footer(stdscr, "Esc Back · ↑↓ Navigate · Enter Select")
            stdscr.refresh()
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key):
                return None
            if key == curses.KEY_UP:
                selected = max(0, selected - 1)
            elif key == curses.KEY_DOWN:
                selected = min(len(choices) - 1, selected + 1)
            elif self._is_enter(key):
                return choices[selected].value

    def _bottom_menu(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        title: str,
        subtitle: str,
        choices: tuple[MenuChoice, ...],
        restore_nodelay: bool = False,
        autonomous_value: str | None = None,
    ) -> str | None:
        selected = 0
        with suppress(curses.error):
            stdscr.nodelay(False)
        try:
            while True:
                if autonomous_value is not None and self.agent_mode == AgentMode.AUTONOMOUS:
                    return autonomous_value
                messages = self._read_message_lines(session.path)
                panel = BottomPanel(title, subtitle, choices, selected)
                self._draw_session(
                    stdscr,
                    session,
                    messages,
                    "",
                    0,
                    0,
                    bottom_panel=panel,
                )
                key = stdscr.get_wch()
                if self._is_escape(key) or self._is_ctrl_c(key):
                    return None
                if self._is_shift_tab(key):
                    self._cycle_agent_mode()
                    continue
                if key == curses.KEY_UP:
                    selected = max(0, selected - 1)
                elif key == curses.KEY_DOWN:
                    selected = min(len(choices) - 1, selected + 1)
                elif key == curses.KEY_PPAGE:
                    viewport = self._bottom_panel_viewport(stdscr, panel)
                    page_size = max(1, len(viewport.visible_indices))
                    selected = max(0, selected - page_size)
                elif key == curses.KEY_NPAGE:
                    viewport = self._bottom_panel_viewport(stdscr, panel)
                    page_size = max(1, len(viewport.visible_indices))
                    selected = min(len(choices) - 1, selected + page_size)
                elif key == curses.KEY_MOUSE:
                    choice = self._bottom_panel_mouse_choice(stdscr, panel)
                    if choice is not None:
                        return choices[choice].value
                elif self._is_enter(key):
                    return choices[selected].value
        finally:
            if restore_nodelay:
                with suppress(curses.error):
                    stdscr.nodelay(True)

    def _draw_shell(
        self,
        stdscr: CursesWindow,
        title: str,
        subtitle: str = "",
        plan_steps: tuple[PlanStep, ...] = (),
    ) -> tuple[int, int]:
        with suppress(curses.error):
            curses.curs_set(0)
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        self._paint_background(stdscr)
        self._draw_header_box(stdscr, title, subtitle, plan_steps)
        return height, width

    def _paint_background(self, stdscr: CursesWindow) -> None:
        height, width = stdscr.getmaxyx()
        attr = self._attr("background")
        for y in range(height):
            self._add(stdscr, y, 0, " " * max(1, width), width, attr)

    def _clear_row(self, stdscr: CursesWindow, y: int) -> None:
        _, width = stdscr.getmaxyx()
        self._add(stdscr, y, 0, " " * max(1, width), width, self._attr("background"))

    def _draw_header_box(
        self,
        stdscr: CursesWindow,
        title: str,
        subtitle: str = "",
        plan_steps: tuple[PlanStep, ...] = (),
    ) -> None:
        _, width = stdscr.getmaxyx()
        version = f"v{__version__}"
        top = 1
        bottom = self._header_bottom(plan_steps)
        horizontal = "─" * max(1, width - 6)
        self._add(stdscr, top, 2, f"╭{horizontal}╮", width - 4, self._attr("accent"))
        for y in range(top + 1, bottom):
            self._add(stdscr, y, 2, "│", 1, self._attr("accent"))
            self._add(stdscr, y, max(2, width - 3), "│", 1, self._attr("accent"))
        self._add(stdscr, bottom, 2, f"╰{horizontal}╯", width - 4, self._attr("accent"))

        brand = "Anomx"
        descriptor = "Anomaly Detection and Data Analysis Agent"
        self._add(stdscr, top + 1, 4, brand, width - 8, self._attr("accent"))
        self._add(
            stdscr,
            top + 1,
            4 + len(brand) + 2,
            descriptor,
            max(1, width - len(brand) - len(version) - 14),
            self._attr("light"),
        )
        self._add(
            stdscr,
            top + 1,
            max(4, width - len(version) - 5),
            version,
            len(version),
            self._attr("light"),
        )
        self._add(
            stdscr,
            top + 2,
            4,
            title,
            width - 8,
            self._attr("bold"),
        )
        if subtitle:
            self._add(
                stdscr,
                top + 3,
                4,
                subtitle,
                width - 8,
                self._attr("light"),
            )
        if plan_steps:
            for index, step in enumerate(plan_steps):
                y = top + 5 + index
                checkbox = "☑" if step.is_done else "☐"
                title_text = self._strike_text(step.title) if step.is_done else step.title
                attr = self._attr("light") if step.is_done else self._attr("bold")
                self._add(
                    stdscr,
                    y,
                    4,
                    f"{checkbox} {title_text}",
                    width - 8,
                    attr,
                )

    def _header_bottom(self, plan_steps: tuple[PlanStep, ...] = ()) -> int:
        if not plan_steps:
            return 5
        return 6 + len(plan_steps)

    def _session_body_top(self, plan_steps: tuple[PlanStep, ...] = ()) -> int:
        return self._header_bottom(plan_steps) + 2

    def _strike_text(self, text: str) -> str:
        return "".join(
            f"{character}\u0336" if character != " " else character for character in text
        )

    def _draw_access_check(self, stdscr: CursesWindow, selected: int) -> None:
        height, width = self._draw_shell(stdscr, "Access Check", "Accessing workspace")
        repo = str(self.cwd.expanduser().resolve())
        self._add(stdscr, 8, 4, repo, width - 8, self._attr("bold"))

        copy = (
            "Quick safety check: Is this a project you created or one you trust? "
            "If not, take a moment to review what's in this folder first."
        )
        y = 11
        for line in textwrap.wrap(copy, width=max(24, width - 8)):
            self._add(stdscr, y, 4, line, width - 8)
            y += 1
        y += 1
        self._add(
            stdscr,
            y,
            4,
            "Anomx will be able to read, edit, and execute files here.",
            width - 8,
        )
        y += 2
        self._add(stdscr, y, 4, "Security guide", width - 8, self._attr("light"))
        y += 2

        choices = ("Yes, I trust this folder", "No, exit")
        for index, choice in enumerate(choices):
            marker = "›" if index == selected else " "
            attr = self._attr("accent") if index == selected else curses.A_NORMAL
            self._add(stdscr, y + index, 4, f"{marker} {index + 1}. {choice}", width - 8, attr)

        self._add(
            stdscr,
            min(height - 2, y + len(choices) + 2),
            4,
            "Enter to confirm · Esc to cancel",
            width - 8,
            self._attr("light"),
        )
        stdscr.refresh()

    def _draw_table(
        self,
        stdscr: CursesWindow,
        title: str,
        rows: list[tuple[str, str]],
        selected: int,
    ) -> None:
        height, width = self._draw_shell(stdscr, title)
        left_width = max(24, min(40, width // 3))
        y = 7
        self._add(stdscr, y, 4, "Config", left_width - 4, self._attr("bold"))
        self._add(stdscr, y, left_width, "Value", width - left_width - 4, self._attr("bold"))
        y += 2
        for index, (field, value) in enumerate(rows):
            if y >= height - 2:
                break
            attr = self._attr("selected") if index == selected else curses.A_NORMAL
            self._add(stdscr, y, 4, field, left_width - 6, attr)
            self._add(stdscr, y, left_width, value, width - left_width - 4, attr)
            y += 1
        self._footer(stdscr, "Esc Back · ↑↓ Navigate · Enter Change")
        stdscr.refresh()

    def _draw_session(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        messages: list[MessageLine],
        input_text: str,
        cursor: int,
        scroll: int,
        command_suggestions: list[CommandSpec] | None = None,
        command_selected: int = 0,
        bottom_panel: BottomPanel | None = None,
        working_text: str | None = None,
        working_deadline: float | None = None,
        working_frame: int = 0,
        anchor_line: int | None = None,
        prompt_notice: str = "",
        prompt_notice_role: str = "light",
    ) -> None:
        config = self.home.load_config()
        provider = str(config.get("provider", session.provider))
        model = str(config.get("model", session.model))
        session_events = self.home.read_session_events(session.path)
        plan_steps = latest_plan_steps(session_events)
        running_workers = (
            running_worker_snapshots(session_events) if bottom_panel is None else ()
        )
        height, width = self._draw_shell(
            stdscr,
            session.title,
            self._session_status_line(session, provider, model),
            plan_steps,
        )
        layout = self._prompt_layout(stdscr, input_text)
        suggestions = command_suggestions or []
        worker_panel_height = len(running_workers) + (1 if running_workers else 0)
        body_top = self._session_body_top(plan_steps)
        body_bottom = max(body_top + 1, layout.top_line - worker_panel_height)
        body_height = max(1, body_bottom - body_top)
        rendered = self._session_rendered_lines(
            session,
            messages,
            max(20, width - 8),
            self._working_status_text(working_text, working_deadline),
        )
        max_scroll = max(0, len(rendered) - body_height)
        if anchor_line is None:
            scroll = min(scroll, max_scroll)
            start = max(0, len(rendered) - body_height - scroll)
        else:
            start = max(0, min(anchor_line, max(0, len(rendered) - 1)))
        visible = rendered[start : start + body_height]
        self._click_targets = {}
        for offset, line in enumerate(visible):
            y = body_top + offset
            if line.role == "work_summary":
                self._click_targets[y] = line.meta
            if line.role == "working":
                self._draw_working_line(
                    stdscr,
                    y,
                    4,
                    line.text,
                    width - 8,
                    working_frame,
                )
                continue
            attr = self._line_attr(line.role)
            self._add(stdscr, y, 4, line.text, width - 8, attr)

        if running_workers:
            self._draw_running_workers(stdscr, running_workers, body_bottom, working_frame)
        if bottom_panel is not None:
            self._draw_bottom_panel(stdscr, bottom_panel)
        else:
            self._draw_command_suggestions(
                stdscr,
                suggestions,
                command_selected,
                input_text,
                reserved_bottom_rows=worker_panel_height,
            )
        self._draw_prompt_bar(stdscr, input_text, cursor, prompt_notice, prompt_notice_role)
        stdscr.refresh()

    def _line_attr(self, role: str) -> int:
        if role == "user":
            return self._attr("accent")
        if role == "meta_accent":
            return self._attr("accent")
        if role in {"meta", "tool", "work_summary", "worker", "approved"}:
            return self._attr("light")
        if role == "system":
            return self._attr("danger")
        if role == "forbidden":
            return self._attr("light")
        return curses.A_NORMAL

    def _session_rendered_lines(
        self,
        session: SessionRecord,
        messages: list[MessageLine],
        width: int,
        working_text: str | None = None,
    ) -> list[MessageLine]:
        if working_text is not None:
            messages = [*messages, MessageLine("working", working_text)]
        return self._render_messages(messages, width)

    def _working_status_text(
        self,
        working_text: str | None,
        working_deadline: float | None = None,
        now: float | None = None,
    ) -> str | None:
        if working_text is None:
            return None
        if working_deadline is None:
            return working_text
        current_time = time.monotonic() if now is None else now
        remaining = max(0, math.ceil(working_deadline - current_time))
        return f"{working_text} {remaining // 60:02d}:{remaining % 60:02d}"

    def _session_status_line(self, session: SessionRecord, provider: str, model: str) -> str:
        parts = [f"{session.session_id[:8]} · {provider}/{model}"]
        context_status = self._context_status(session, model)
        if context_status:
            parts.append(context_status)
        return " · ".join(parts)

    def _context_status(self, session: SessionRecord, model: str) -> str:
        context_window = model_context_window(model)
        if context_window is None or not self._has_user_messages(session.path):
            return ""

        used_tokens = self._estimate_context_tokens(session.path)
        left_tokens = max(0, context_window - used_tokens)
        percent_left = round((left_tokens / context_window) * 100)
        return f"{percent_left}% context left"

    def _estimate_context_tokens(self, session_path: Path) -> int:
        characters = 0
        for event in self.home.read_session_events(session_path):
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            event_type = (
                payload.get("type") if event.get("type") == "event_msg" else event.get("type")
            )
            if event_type in {"user_message", "agent_message", "work_message"}:
                characters += len(str(payload.get("message", "")))
        prompt_overhead = 1_500
        return max(1, prompt_overhead + ((characters + 3) // 4))

    def _draw_working_line(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        width: int,
        frame: int,
    ) -> None:
        dots = "." * ((frame // 4) % 4)
        self._add(stdscr, y, x, f"{text}{dots}", width, self._attr("light"))

    def _draw_running_workers(
        self,
        stdscr: CursesWindow,
        workers: tuple[WorkerAgentSnapshot, ...],
        start_y: int,
        frame: int,
    ) -> None:
        _, width = stdscr.getmaxyx()
        panel_width = max(1, width - 4)
        self._add(stdscr, start_y, 2, "─" * panel_width, panel_width, self._attr("light"))
        dots = "." * ((frame // 4) % 4)
        for offset, worker in enumerate(workers, start=1):
            statement = worker.statement or "thinking"
            duration = self._worker_runtime_duration(worker)
            right_x = max(4, width - len(duration) - 4) if duration else width
            left_width = max(1, right_x - 6)
            text = f"{worker.name} ({worker.worker_id}) · {statement}{dots}"
            self._add(stdscr, start_y + offset, 4, text, left_width, self._attr("light"))
            if duration:
                self._add(
                    stdscr,
                    start_y + offset,
                    right_x,
                    duration,
                    len(duration),
                    self._attr("light"),
                )

    def _worker_runtime_duration(self, worker: WorkerAgentSnapshot) -> str:
        if not worker.started_at:
            return ""
        with suppress(ValueError):
            started = datetime.fromisoformat(worker.started_at.replace("Z", "+00:00"))
            seconds = max(0, int((datetime.now(tz=UTC) - started).total_seconds()))
            return self._format_duration(seconds)
        return ""

    def _draw_command_suggestions(
        self,
        stdscr: CursesWindow,
        suggestions: list[CommandSpec],
        selected: int,
        input_text: str = "",
        reserved_bottom_rows: int = 0,
    ) -> None:
        if not suggestions:
            return
        _, width = stdscr.getmaxyx()
        start_y = max(
            0,
            self._command_suggestions_start(stdscr, suggestions, input_text)
            - reserved_bottom_rows,
        )
        command_width = min(18, max(10, width // 5))
        for y in range(start_y, start_y + len(suggestions)):
            self._clear_row(stdscr, y)
        for index, command in enumerate(suggestions):
            marker = "›" if index == selected else " "
            attr = self._attr("accent") if index == selected else curses.A_NORMAL
            self._add(
                stdscr,
                start_y + index,
                2,
                f"{marker} {command.command}",
                command_width,
                attr,
            )
            self._add(
                stdscr,
                start_y + index,
                2 + command_width,
                command.description,
                width - command_width - 4,
                self._attr("light"),
            )

    def _draw_bottom_panel(self, stdscr: CursesWindow, panel: BottomPanel) -> None:
        layout = self._prompt_layout(stdscr)
        _, width = stdscr.getmaxyx()
        viewport = self._bottom_panel_viewport(stdscr, panel)
        start_y = viewport.start_y
        panel_width = max(1, width - 4)
        for y in range(start_y, layout.top_line + 1):
            self._clear_row(stdscr, y)
        self._add(stdscr, start_y, 2, "─" * panel_width, panel_width, self._attr("accent"))
        self._add(stdscr, start_y + 1, 4, panel.title, panel_width - 4, self._attr("accent"))
        for offset, line in enumerate(viewport.subtitle_lines):
            self._add(stdscr, start_y + 2 + offset, 4, line, panel_width - 4, self._attr("light"))
        choice_y = viewport.choice_y
        longest_label = max((len(choice.label) for choice in panel.choices), default=0)
        detail_x = min(width - 4, max(44, longest_label + 10))
        if viewport.show_overflow_counts:
            self._add(
                stdscr,
                choice_y,
                4,
                f"↑ {viewport.more_above} more above",
                width - 8,
                self._attr("light"),
            )
            choice_y += 1
        for row_offset, choice_index in enumerate(viewport.visible_indices):
            choice = panel.choices[choice_index]
            marker = "›" if choice_index == panel.selected else " "
            attr = self._attr("accent") if choice_index == panel.selected else curses.A_NORMAL
            self._add(
                stdscr,
                choice_y + row_offset,
                4,
                f"{marker} {choice.label}",
                max(1, detail_x - 6),
                attr,
            )
            if choice.detail:
                self._add(
                    stdscr,
                    choice_y + row_offset,
                    detail_x,
                    choice.detail,
                    width - detail_x - 4,
                    self._attr("light"),
                )
        if viewport.show_overflow_counts:
            self._add(
                stdscr,
                choice_y + len(viewport.visible_indices),
                4,
                f"↓ {viewport.more_below} more below",
                width - 8,
                self._attr("light"),
            )

    def _bottom_panel_height(self, panel: BottomPanel, subtitle_line_count: int) -> int:
        return min(18, len(panel.choices) + subtitle_line_count + 5)

    def _bottom_panel_start(
        self,
        stdscr: CursesWindow,
        panel: BottomPanel,
        subtitle_line_count: int,
    ) -> int:
        layout = self._prompt_layout(stdscr)
        return max(6, layout.top_line - self._bottom_panel_height(panel, subtitle_line_count))

    def _bottom_panel_viewport(
        self,
        stdscr: CursesWindow,
        panel: BottomPanel,
    ) -> BottomPanelViewport:
        layout = self._prompt_layout(stdscr)
        _, width = stdscr.getmaxyx()
        subtitle_lines = tuple(
            self._panel_text_lines(panel.subtitle, max(1, width - 8), max_lines=4)
        )
        start_y = self._bottom_panel_start(stdscr, panel, len(subtitle_lines))
        first_choice_y = start_y + 4 + len(subtitle_lines)
        raw_visible_rows = max(1, layout.top_line - first_choice_y)
        show_overflow_counts = len(panel.choices) > raw_visible_rows and raw_visible_rows >= 3
        visible_rows = raw_visible_rows - 2 if show_overflow_counts else raw_visible_rows
        visible_rows = max(1, min(len(panel.choices), visible_rows))
        max_offset = max(0, len(panel.choices) - visible_rows)
        offset = min(max(panel.selected - visible_rows + 1, 0), max_offset)
        visible_indices = tuple(range(offset, min(len(panel.choices), offset + visible_rows)))
        return BottomPanelViewport(
            start_y=start_y,
            subtitle_lines=subtitle_lines,
            choice_y=first_choice_y,
            visible_indices=visible_indices,
            more_above=offset,
            more_below=max(0, len(panel.choices) - (offset + len(visible_indices))),
            show_overflow_counts=show_overflow_counts,
        )

    def _panel_text_lines(self, text: str, width: int, max_lines: int) -> list[str]:
        if not text:
            return []
        sanitized = " ".join(text.replace("\r", " ").replace("\n", " / ").split())
        return textwrap.wrap(
            sanitized,
            width=max(10, width),
            max_lines=max_lines,
            placeholder="...",
        )

    def _bottom_panel_mouse_choice(
        self,
        stdscr: CursesWindow,
        panel: BottomPanel,
    ) -> int | None:
        with suppress(curses.error):
            _, _x, y, _, button_state = curses.getmouse()
            if not self._is_left_click(button_state):
                return None
            viewport = self._bottom_panel_viewport(stdscr, panel)
            choice_y = viewport.choice_y + (1 if viewport.show_overflow_counts else 0)
            index = y - choice_y
            if 0 <= index < len(viewport.visible_indices):
                return viewport.visible_indices[index]
        return None

    def _draw_prompt_bar(
        self,
        stdscr: CursesWindow,
        input_text: str,
        cursor: int,
        notice: str = "",
        notice_role: str = "light",
    ) -> None:
        layout = self._prompt_layout(stdscr, input_text)
        for y in range(layout.top_line, layout.hint_line + 1):
            self._clear_row(stdscr, y)
        self._add(
            stdscr,
            layout.top_line,
            2,
            "─" * max(1, layout.input_width + 2),
            layout.input_width + 2,
            self._attr("light"),
        )
        display_text = input_text or self._prompt_placeholder
        attr = self._attr("bold") if input_text else self._attr("light")
        lines = self._prompt_lines(display_text, layout.input_width)
        view_start = self._prompt_view_start(input_text, cursor, layout)
        visible_lines = lines[view_start : view_start + layout.prompt_height]
        for offset in range(layout.prompt_height):
            y = layout.prompt_line + offset
            marker = "›" if offset == 0 else " "
            self._add(stdscr, y, 2, marker, 1, self._attr("accent"))
            line = visible_lines[offset] if offset < len(visible_lines) else ""
            self._add(stdscr, y, layout.input_x, line, layout.input_width, attr)
        self._add(
            stdscr,
            layout.bottom_line,
            2,
            "─" * max(1, layout.input_width + 2),
            layout.input_width + 2,
            self._attr("light"),
        )
        self._add(
            stdscr,
            layout.hint_line,
            4,
            self.agent_mode.prompt_hint,
            layout.input_width,
            self._attr(self._mode_hint_attr_name()),
        )
        if input_text:
            cursor_line, cursor_column = self._prompt_cursor_position(
                input_text,
                cursor,
                layout.input_width,
            )
            cursor_x = layout.input_x + cursor_column
            visible_cursor_line = max(
                0,
                min(layout.prompt_height - 1, cursor_line - view_start),
            )
            cursor_y = layout.prompt_line + visible_cursor_line
            with suppress(curses.error):
                curses.curs_set(1)
                stdscr.move(cursor_y, cursor_x)
        else:
            with suppress(curses.error):
                curses.curs_set(0)

    def _prompt_layout(self, stdscr: CursesWindow, input_text: str = "") -> PromptLayout:
        height, width = stdscr.getmaxyx()
        input_width = max(1, width - 8)
        max_prompt_height = max(1, height // 4)
        prompt_line_count = len(self._prompt_lines(input_text, input_width)) if input_text else 1
        prompt_height = max(1, min(max_prompt_height, prompt_line_count))
        bottom_line = max(0, height - 2)
        prompt_line = max(0, bottom_line - prompt_height)
        top_line = max(0, prompt_line - 1)
        return PromptLayout(
            top_line=top_line,
            prompt_line=prompt_line,
            bottom_line=bottom_line,
            hint_line=max(0, height - 1),
            input_x=4,
            input_width=input_width,
            prompt_height=prompt_height,
        )

    def _prompt_lines(self, text: str, width: int) -> list[str]:
        if not text:
            return [""]
        safe_width = max(1, width)
        return [text[index : index + safe_width] for index in range(0, len(text), safe_width)]

    def _prompt_view_start(
        self,
        input_text: str,
        cursor: int,
        layout: PromptLayout,
    ) -> int:
        if not input_text:
            return 0
        cursor_line, _ = self._prompt_cursor_position(input_text, cursor, layout.input_width)
        line_count = len(self._prompt_lines(input_text, layout.input_width))
        max_start = max(0, line_count - layout.prompt_height)
        return max(0, min(max_start, cursor_line - layout.prompt_height + 1))

    def _prompt_cursor_position(self, input_text: str, cursor: int, width: int) -> tuple[int, int]:
        safe_width = max(1, width)
        bounded_cursor = max(0, min(cursor, len(input_text)))
        if bounded_cursor > 0 and bounded_cursor % safe_width == 0:
            return (bounded_cursor - 1) // safe_width, safe_width - 1
        return bounded_cursor // safe_width, bounded_cursor % safe_width

    def _session_mouse_action(
        self,
        stdscr: CursesWindow,
        input_text: str,
        command_suggestions: list[CommandSpec],
    ) -> SessionMouseAction | None:
        with suppress(curses.error):
            _, x, y, _, button_state = curses.getmouse()
            wheel_up = getattr(curses, "BUTTON4_PRESSED", 0)
            wheel_down = getattr(curses, "BUTTON5_PRESSED", 0)
            if wheel_up and button_state & wheel_up:
                return SessionMouseAction("scroll", 1)
            if wheel_down and button_state & wheel_down:
                return SessionMouseAction("scroll", -1)

            if self._is_left_click(button_state) and y in self._click_targets:
                return SessionMouseAction("toggle_work", 0, self._click_targets[y])

            if command_suggestions and self._is_left_click(button_state):
                start_y = self._command_suggestions_start(
                    stdscr,
                    command_suggestions,
                    input_text,
                )
                index = y - start_y
                if 0 <= index < len(command_suggestions):
                    return SessionMouseAction("command", index)

            layout = self._prompt_layout(stdscr, input_text)
            clicked_prompt = layout.prompt_line <= y < layout.prompt_line + layout.prompt_height
            if clicked_prompt and self._is_left_click(button_state):
                view_start = self._prompt_view_start(input_text, len(input_text), layout)
                clicked_line = view_start + (y - layout.prompt_line)
                cursor = (clicked_line * layout.input_width) + (x - layout.input_x)
                cursor = max(0, min(len(input_text), cursor))
                return SessionMouseAction("cursor", cursor)
        return None

    def _command_suggestions_start(
        self,
        stdscr: CursesWindow,
        suggestions: list[CommandSpec],
        input_text: str = "",
    ) -> int:
        layout = self._prompt_layout(stdscr, input_text)
        return max(0, layout.top_line - len(suggestions) - 1)

    def _is_left_click(self, button_state: int) -> bool:
        return bool(
            button_state
            & (
                curses.BUTTON1_CLICKED
                | curses.BUTTON1_PRESSED
                | curses.BUTTON1_RELEASED
            )
        )

    def _prompt_text(
        self,
        stdscr: CursesWindow,
        title: str,
        label: str,
        mask: bool = False,
        optional: bool = True,
    ) -> str | None:
        value = ""
        while True:
            height, width = self._draw_shell(stdscr, title, label)
            display_value = "*" * len(value) if mask else value
            prompt = f"{label}: {display_value}"
            self._add(stdscr, height // 2, 4, prompt, width - 8, self._attr("bold"))
            self._footer(stdscr, "Esc Cancel · Enter Save")
            stdscr.refresh()
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key):
                return None if optional else ""
            if self._is_enter(key):
                if value or optional:
                    return value
            elif self._is_backspace(key):
                value = value[:-1]
            elif isinstance(key, str) and key.isprintable():
                value += key

    def _message(self, stdscr: CursesWindow, title: str, message: str) -> None:
        while True:
            height, width = self._draw_shell(stdscr, title)
            lines = textwrap.wrap(message, width=max(20, width - 8)) or [message]
            start_y = max(4, (height - len(lines)) // 2)
            for offset, line in enumerate(lines):
                self._add(stdscr, start_y + offset, 4, line, width - 8)
            self._footer(stdscr, "Esc Back · Enter Continue")
            stdscr.refresh()
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key) or self._is_enter(key):
                return

    def _run_backend_turn(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        anchor_line: int | None = None,
    ) -> BackendTurnResult:
        events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
        result: dict[str, str] = {}
        turn_id = uuid4().hex
        started_at = time.monotonic()
        work_count = 0
        input_text = ""
        cursor = 0
        prompt_notice = RUNNING_NOTICE
        prompt_notice_role = "light"
        abort_key = ""
        abort_deadline = 0.0

        def status_callback(message: str) -> None:
            events.put(RuntimeUiEvent("status", message))

        def message_callback(message: str) -> None:
            events.put(RuntimeUiEvent("message", message))

        def tool_message_callback(message: str) -> None:
            events.put(RuntimeUiEvent("tool_message", message))

        def delta_callback(delta: str) -> None:
            events.put(RuntimeUiEvent("delta", delta))

        def system_message_callback(role: str, message: str) -> None:
            events.put(RuntimeUiEvent("system_message", message, role))

        def approval_callback(request: CommandApprovalRequest) -> ApprovalChoice:
            response: queue.SimpleQueue[ApprovalChoice] = queue.SimpleQueue()
            events.put(
                RuntimeUiEvent(
                    "approval",
                    approval_request=request,
                    approval_response=response,
                )
            )
            return response.get()

        def run_backend() -> None:
            result["response"] = self.runtime.backend_response(
                session.path,
                callbacks=RuntimeCallbacks(
                    status=status_callback,
                    message=message_callback,
                    tool_message=tool_message_callback,
                    delta=delta_callback,
                    approval=approval_callback,
                    system_message=system_message_callback,
                ),
            )

        worker = threading.Thread(target=run_backend, daemon=True)
        worker.start()
        frame = 0
        working_text: str | None = "Thinking"
        working_deadline: float | None = None
        final_text = ""
        running_anchor = anchor_line
        running_scroll = 0
        with suppress(curses.error):
            stdscr.nodelay(True)
        try:
            while worker.is_alive():
                key = self._read_nonblocking_key(stdscr)
                if key is not None:
                    key_result = self._handle_running_key(
                        stdscr,
                        session,
                        key,
                        input_text,
                        cursor,
                        abort_key,
                        abort_deadline,
                    )
                    input_text = key_result.input_text
                    cursor = key_result.cursor
                    prompt_notice = key_result.notice
                    prompt_notice_role = key_result.notice_role
                    abort_key = key_result.abort_key
                    abort_deadline = key_result.abort_deadline
                    if key_result.clear_anchor:
                        running_anchor = None
                    if key_result.scroll_delta:
                        running_anchor = None
                        running_scroll = max(0, running_scroll + key_result.scroll_delta)
                    if key_result.exit_requested:
                        return BackendTurnResult(
                            input_text,
                            cursor,
                            exit_requested=True,
                            anchor_line=running_anchor,
                            scroll=running_scroll,
                        )
                elif abort_key and time.monotonic() > abort_deadline:
                    abort_key = ""
                    prompt_notice = RUNNING_NOTICE
                    prompt_notice_role = "light"

                (
                    working_text,
                    working_deadline,
                    final_text,
                    work_count,
                ) = self._process_runtime_events(
                    stdscr,
                    session,
                    events,
                    working_text,
                    working_deadline,
                    final_text,
                    turn_id,
                    work_count,
                    anchor_line,
                    input_text,
                    cursor,
                    prompt_notice,
                    prompt_notice_role,
                )
                self._draw_session(
                    stdscr,
                    session,
                    self._read_message_lines(session.path),
                    input_text,
                    cursor,
                    running_scroll,
                    working_text=working_text,
                    working_deadline=working_deadline,
                    working_frame=frame,
                    anchor_line=running_anchor,
                    prompt_notice=prompt_notice,
                    prompt_notice_role=prompt_notice_role,
                )
                time.sleep(0.08)
                frame += 1
        finally:
            with suppress(curses.error):
                stdscr.nodelay(False)
        worker.join()
        working_text, working_deadline, final_text, work_count = self._process_runtime_events(
            stdscr,
            session,
            events,
            working_text,
            working_deadline,
            final_text,
            turn_id,
            work_count,
            anchor_line,
            input_text,
            cursor,
            prompt_notice,
            prompt_notice_role,
        )
        response = final_text or result.get("response", "")
        if work_count:
            duration = self._format_duration(time.monotonic() - started_at)
            self.home.append_session_event(
                session.path,
                "work_summary",
                {
                    "turn_id": turn_id,
                    "message": f"Worked for {duration} min",
                },
            )
        if response:
            if not final_text:
                self._fake_type_message(stdscr, session, response, anchor_line=anchor_line)
            self.home.append_session_event(session.path, "agent_message", {"message": response})
        self._draw_session(
            stdscr,
            session,
            self._read_message_lines(session.path),
            input_text,
            cursor,
            0,
            anchor_line=anchor_line,
        )
        return BackendTurnResult(
            input_text,
            cursor,
            anchor_line=running_anchor,
            scroll=running_scroll,
        )

    def _read_nonblocking_key(self, stdscr: CursesWindow) -> str | int | None:
        try:
            return stdscr.get_wch()
        except curses.error:
            return None

    def _handle_running_key(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        key: str | int,
        input_text: str,
        cursor: int,
        abort_key: str,
        abort_deadline: float,
    ) -> RunningKeyResult:
        now = time.monotonic()
        if self._is_shift_tab(key):
            self._cycle_agent_mode()
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
            )

        if self._is_escape(key) or self._is_ctrl_c(key):
            key_label = "Ctrl+C" if self._is_ctrl_c(key) else "Esc"
            if abort_key == key_label and now <= abort_deadline:
                self._append_system_message(session, "system", "Anomx aborted by user.")
                return RunningKeyResult(
                    input_text,
                    cursor,
                    "Aborting Anomx",
                    "danger",
                    key_label,
                    now,
                    exit_requested=True,
                )

            notice = f"Press {key_label} again to abort Anomx"
            self._append_system_message(session, "system", notice)
            return RunningKeyResult(input_text, cursor, notice, "danger", key_label, now + 3.0)

        if self._is_enter(key):
            return RunningKeyResult(
                input_text,
                cursor,
                "Agent is working · Enter disabled until the response finishes",
                "light",
                abort_key,
                abort_deadline,
            )

        if key == curses.KEY_UP:
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                scroll_delta=1,
                clear_anchor=True,
            )
        if key == curses.KEY_DOWN:
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                scroll_delta=-1,
                clear_anchor=True,
            )
        if key == curses.KEY_PPAGE:
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                scroll_delta=5,
                clear_anchor=True,
            )
        if key == curses.KEY_NPAGE:
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                scroll_delta=-5,
                clear_anchor=True,
            )

        if key == curses.KEY_LEFT:
            return RunningKeyResult(
                input_text,
                max(0, cursor - 1),
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
            )
        if key == curses.KEY_RIGHT:
            return RunningKeyResult(
                input_text,
                min(len(input_text), cursor + 1),
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
            )
        if key == curses.KEY_HOME:
            return RunningKeyResult(
                input_text,
                0,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
            )
        if key == curses.KEY_END:
            return RunningKeyResult(
                input_text,
                len(input_text),
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
            )
        if key == curses.KEY_MOUSE:
            action = self._session_mouse_action(stdscr, input_text, [])
            if action is not None and action.kind == "cursor":
                return RunningKeyResult(
                    input_text,
                    action.value,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                )
            if action is not None and action.kind == "scroll":
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    scroll_delta=action.value,
                    clear_anchor=True,
                )
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
            )
        if self._is_backspace(key):
            if cursor <= 0:
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                )
            updated = input_text[: cursor - 1] + input_text[cursor:]
            return RunningKeyResult(
                updated,
                cursor - 1,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
            )
        if isinstance(key, str) and key.isprintable():
            updated = input_text[:cursor] + key + input_text[cursor:]
            return RunningKeyResult(
                updated,
                cursor + len(key),
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
            )
        return RunningKeyResult(
            input_text,
            cursor,
            RUNNING_NOTICE,
            "light",
            abort_key,
            abort_deadline,
        )

    def _append_system_message(
        self,
        session: SessionRecord,
        role: str,
        message: str,
    ) -> None:
        self.home.append_session_event(
            session.path,
            "system_message",
            {
                "message": message,
                "role": role,
            },
        )

    def _backend_response(
        self,
        session_path: Path,
        status_callback: StatusCallback | None = None,
    ) -> str:
        callbacks = RuntimeCallbacks(status=status_callback)
        return self.runtime.backend_response(session_path, callbacks=callbacks)

    def _openai_response(self, session_path: Path, model: str) -> str:
        return self.runtime.openai_response(session_path, model)

    def _conversation_messages(self, session_path: Path) -> list[dict[str, str]]:
        return self.runtime.conversation_messages(session_path)

    def _extract_openai_text(self, data: dict[str, object]) -> str:
        return self.runtime.extract_openai_text(data)

    def _process_runtime_events(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        events: queue.SimpleQueue[RuntimeUiEvent],
        working_text: str | None,
        working_deadline: float | None,
        final_text: str,
        turn_id: str,
        work_count: int,
        anchor_line: int | None,
        input_text: str = "",
        cursor: int = 0,
        prompt_notice: str = "",
        prompt_notice_role: str = "light",
    ) -> tuple[str | None, float | None, str, int]:
        current_working = working_text
        current_deadline = working_deadline
        current_final = final_text
        current_work_count = work_count
        while True:
            try:
                event = events.get_nowait()
            except queue.Empty:
                return current_working, current_deadline, current_final, current_work_count

            if event.kind == "status":
                status_text, status_seconds = self._parse_runtime_status(event.text)
                current_working = status_text
                current_deadline = (
                    time.monotonic() + status_seconds if status_seconds is not None else None
                )
            elif event.kind == "delta":
                current_final += event.text
            elif event.kind in {"message", "tool_message"} and event.text:
                role = "tool" if event.kind == "tool_message" else "agent"
                current_working = None
                current_deadline = None
                self._fake_type_message(
                    stdscr,
                    session,
                    event.text,
                    role=role,
                    anchor_line=anchor_line,
                    input_text=input_text,
                    cursor=cursor,
                    prompt_notice=prompt_notice,
                    prompt_notice_role=prompt_notice_role,
                )
                if event.kind == "message":
                    self.home.append_session_event(
                        session.path,
                        "agent_message",
                        {"message": event.text, "turn_id": turn_id},
                    )
                    current_work_count += 1
                else:
                    self.home.append_session_event(
                        session.path,
                        "work_message",
                        {
                            "message": event.text,
                            "role": role,
                            "turn_id": turn_id,
                        },
                    )
                    current_work_count += 1
            elif event.kind == "approval":
                request = event.approval_request
                response = event.approval_response
                if request is not None and response is not None:
                    choice = self._request_command_approval(stdscr, session, request)
                    response.put(choice)
                    approval_message = self._approval_work_message(request, choice)
                    if approval_message:
                        self.home.append_session_event(
                            session.path,
                            "work_message",
                            {
                                "message": approval_message,
                                "role": "tool",
                                "turn_id": turn_id,
                            },
                        )
                        current_work_count += 1
            elif event.kind == "system_message" and event.text:
                self.home.append_session_event(
                    session.path,
                    "system_message",
                    {
                        "message": event.text,
                        "role": event.role or "system",
                        "turn_id": turn_id,
                    },
                )
                current_work_count += 1

    def _parse_runtime_status(self, message: str) -> tuple[str, float | None]:
        text = message or "Thinking"
        if text.startswith("Waiting:"):
            raw_seconds = text.partition(":")[2]
            with suppress(ValueError):
                return "Waiting", max(0.0, float(raw_seconds))
            return "Waiting", None
        return text, None

    def _approval_work_message(
        self,
        request: CommandApprovalRequest,
        choice: ApprovalChoice,
    ) -> str:
        if choice == ApprovalChoice.ALLOW:
            return f"Approved command: {request.command}"
        if choice == ApprovalChoice.ALWAYS_ALLOW:
            subject = request.allowance_subject or "command"
            return f"Always approved {subject}: {request.command}"
        if choice == ApprovalChoice.ALWAYS_REJECT:
            subject = request.allowance_subject or "command"
            return f"Always rejected {subject}: {request.command}"
        return f"Rejected command: {request.command}"

    def _fake_type_message(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        message: str,
        role: str = "agent",
        anchor_line: int | None = None,
        input_text: str = "",
        cursor: int = 0,
        prompt_notice: str = "",
        prompt_notice_role: str = "light",
    ) -> None:
        rendered = ""
        for character in message:
            rendered += character
            self._draw_session(
                stdscr,
                session,
                [
                    *self._read_message_lines(session.path),
                    MessageLine(role, rendered),
                ],
                input_text,
                cursor,
                0,
                anchor_line=anchor_line,
                prompt_notice=prompt_notice,
                prompt_notice_role=prompt_notice_role,
            )
            time.sleep(0.006)

    def _request_command_approval(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        request: CommandApprovalRequest,
    ) -> ApprovalChoice:
        allowance_label = request.allowance_label or "matching commands"
        allowance_subject = request.allowance_subject or "this command"
        selected = self._bottom_menu(
            stdscr,
            session,
            "Approve command",
            f"{request.command} · {request.reason}",
            (
                MenuChoice("Approve", ApprovalChoice.ALLOW.value, "Run this command once"),
                MenuChoice("Reject", ApprovalChoice.REJECT.value, "Do not run this command"),
                MenuChoice(
                    f"Always approve {allowance_subject}",
                    ApprovalChoice.ALWAYS_ALLOW.value,
                    f"Trust {allowance_label} for this session",
                ),
                MenuChoice(
                    f"Always reject {allowance_subject}",
                    ApprovalChoice.ALWAYS_REJECT.value,
                    f"Block {allowance_label} for this session and tell all agents not to use it",
                ),
            ),
            restore_nodelay=True,
            autonomous_value=ApprovalChoice.ALLOW.value,
        )
        if selected == ApprovalChoice.ALLOW.value:
            return ApprovalChoice.ALLOW
        if selected == ApprovalChoice.ALWAYS_ALLOW.value:
            return ApprovalChoice.ALWAYS_ALLOW
        if selected == ApprovalChoice.ALWAYS_REJECT.value:
            return ApprovalChoice.ALWAYS_REJECT
        return ApprovalChoice.REJECT

    def _read_message_lines(self, session_path: Path) -> list[MessageLine]:
        lines: list[MessageLine] = []
        pending_turn: dict[str, list[MessageLine]] = {}
        for event in self.home.read_session_events(session_path):
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            event_type = str(
                payload.get("type") if event.get("type") == "event_msg" else event.get("type")
            )
            message = str(payload.get("message", "")).strip()
            if event_type == "user_message" and message:
                lines.append(MessageLine("user", message))
            elif event_type == "agent_message" and message:
                turn_id = str(payload.get("turn_id", ""))
                if turn_id:
                    pending_turn.setdefault(turn_id, []).append(
                        MessageLine("agent", message, turn_id)
                    )
                else:
                    lines.append(MessageLine("agent", message))
            elif event_type == "system_message" and message:
                role = str(payload.get("role", "system"))
                if role == "worker":
                    continue
                turn_id = str(payload.get("turn_id", ""))
                if turn_id:
                    pending_turn.setdefault(turn_id, []).append(MessageLine(role, message, turn_id))
                else:
                    lines.append(MessageLine(role, message))
            elif event_type == "work_message" and message:
                turn_id = str(payload.get("turn_id", ""))
                role = str(payload.get("role", "tool"))
                if turn_id:
                    pending_turn.setdefault(turn_id, []).append(MessageLine(role, message, turn_id))
                else:
                    lines.append(MessageLine(role, message))
            elif event_type == "work_summary" and message:
                turn_id = str(payload.get("turn_id", ""))
                turn_lines = pending_turn.pop(turn_id, [])
                if turn_id in self._expanded_work_turns:
                    lines.extend(turn_lines)
                    lines.append(MessageLine("work_summary", f"{message} · collapse", turn_id))
                else:
                    lines.append(MessageLine("work_summary", f"{message} · expand", turn_id))
        for turn_lines in pending_turn.values():
            lines.extend(turn_lines)
        return lines

    def _render_messages(self, messages: list[MessageLine], width: int) -> list[MessageLine]:
        rendered: list[MessageLine] = []
        previous_kind: str | None = None
        for message in messages:
            kind = self._message_kind(message.role)
            if rendered and previous_kind is not None and kind != previous_kind:
                rendered.append(MessageLine("meta", ""))
            for line in markdown_to_terminal_lines(message.text, width=max(20, width)):
                rendered.append(MessageLine(message.role, line, message.meta))
            previous_kind = kind
        return rendered

    def _message_kind(self, role: str) -> str:
        if role == "user":
            return "user"
        if role == "agent":
            return "agent"
        return "working"

    def _toggle_work_turn(self, turn_id: str) -> None:
        if turn_id in self._expanded_work_turns:
            self._expanded_work_turns.remove(turn_id)
        elif turn_id:
            self._expanded_work_turns.add(turn_id)

    def _cycle_agent_mode(self) -> None:
        self.agent_mode = self.agent_mode.next()
        self.runtime.set_mode(self.agent_mode)

    def _mode_hint_attr_name(self) -> str:
        if self.agent_mode == AgentMode.AUTONOMOUS:
            return "warning"
        return "light"

    def _latest_user_anchor_line(self, stdscr: CursesWindow, session: SessionRecord) -> int | None:
        _, width = stdscr.getmaxyx()
        rendered = self._session_rendered_lines(
            session,
            self._read_message_lines(session.path),
            max(20, width - 8),
        )
        user_lines = [
            index for index, line in enumerate(rendered) if line.role == "user" and line.text
        ]
        return user_lines[-1] if user_lines else None

    def _animate_message_anchor(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        anchor_line: int | None,
    ) -> None:
        if anchor_line is None:
            return
        steps = min(10, max(1, anchor_line))
        for step in range(1, steps + 1):
            interpolated = round(anchor_line * (step / steps))
            self._draw_session(
                stdscr,
                session,
                self._read_message_lines(session.path),
                "",
                0,
                0,
                anchor_line=interpolated,
            )
            time.sleep(0.015)

    def _format_duration(self, seconds: float) -> str:
        total_seconds = max(0, int(round(seconds)))
        minutes, remaining_seconds = divmod(total_seconds, 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"

    def _filtered_commands(self, input_text: str) -> list[CommandSpec]:
        query = input_text.removeprefix("/").strip().lower()
        if not query:
            return list(COMMANDS[:5])

        prefix_matches: list[CommandSpec] = []
        contains_matches: list[CommandSpec] = []
        for command in COMMANDS:
            command_name = command.command.removeprefix("/").lower()
            searchable = f"{command_name} {command.description}".lower()
            if command_name.startswith(query):
                prefix_matches.append(command)
            elif query in searchable:
                contains_matches.append(command)
        return [*prefix_matches, *contains_matches][:5]

    def _submitted_command(
        self,
        submitted: str,
        suggestions: list[CommandSpec],
        selected: int,
    ) -> str:
        exact_command = submitted.split(maxsplit=1)[0]
        if any(command.command == exact_command for command in COMMANDS):
            return exact_command
        if suggestions:
            return suggestions[min(selected, len(suggestions) - 1)].command
        return submitted

    def _config_value(self, field: str, config: dict[str, object]) -> str:
        if field == "home":
            return str(self.home.root)
        if field.endswith("_api_key"):
            provider_key = field.removesuffix("_api_key")
            return "configured" if self.home.has_api_key(provider_key) else "not configured"
        return str(config.get(field, ""))

    def _onboarding_complete(self, config: dict[str, object]) -> bool:
        provider = provider_by_key(str(config.get("provider", "")))
        if provider is None:
            return False
        model = str(config.get("model", ""))
        return bool(config.get("onboarding_complete")) and self._model_allowed(provider, model)

    def _model_allowed(self, provider: ProviderOption, model: str) -> bool:
        return model in provider.models or (provider.allow_custom_model and bool(model))

    def _footer(self, stdscr: CursesWindow, text: str) -> None:
        height, width = stdscr.getmaxyx()
        self._add(stdscr, height - 1, 0, " " * max(1, width), width, self._attr("selected"))
        self._add(stdscr, height - 1, 2, text, width - 4, self._attr("selected"))

    def _add(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        width: int,
        attr: int = 0,
    ) -> None:
        height, max_width = stdscr.getmaxyx()
        if y < 0 or y >= height or x >= max_width:
            return
        safe_width = max(0, min(width, max_width - x))
        if safe_width <= 0:
            return
        with suppress(curses.error):
            stdscr.addnstr(y, x, text, safe_width, attr)

    def _attr(self, name: str) -> int:
        return self._colors.get(name, curses.A_NORMAL)

    def _is_escape(self, key: str | int) -> bool:
        return key == "\x1b" or key == 27

    def _is_ctrl_c(self, key: str | int) -> bool:
        return key == "\x03" or key == 3

    def _is_shift_tab(self, key: str | int) -> bool:
        return key == "\x1b[Z" or key == getattr(curses, "KEY_BTAB", 353)

    def _is_enter(self, key: str | int) -> bool:
        return key in {"\n", "\r", curses.KEY_ENTER}

    def _is_backspace(self, key: str | int) -> bool:
        return key in {"\b", "\x7f", curses.KEY_BACKSPACE, 127, 8}
