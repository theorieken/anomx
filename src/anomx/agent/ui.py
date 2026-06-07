"""Full-screen terminal UI for the Anomx agent."""

from __future__ import annotations

import curses
import hashlib
import math
import os
import queue
import random
import re
import shutil
import subprocess
import textwrap
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from anomx import __version__
from anomx.agent.mode import AgentMode
from anomx.agent.platform_client import (
    PlatformClientError,
    PlatformLoginResult,
    connect_platform,
    heartbeat_platform_connection,
    platform_domain,
)
from anomx.agent.runtime import (
    AgentRuntime,
    QuestionOption,
    QuestionRequest,
    QuestionResponse,
    RuntimeCallbacks,
    StatusCallback,
    context_usage_percent,
)
from anomx.agent.skills import (
    STARTER_SKILL_COMMANDS,
    Skill,
    is_valid_skill_command,
    load_builtin_skills,
    load_user_skills,
    normalize_skill_command,
    skill_invocation_prompt,
    write_user_skill,
)
from anomx.agent.state import (
    WORKER_STATE_INTERRUPTED,
    WORKER_STATE_READY,
    WORKER_STATE_WORKING,
    AsyncProcessSnapshot,
    PlanStep,
    WorkerAgentSnapshot,
    latest_plan_steps,
    running_process_snapshots,
    worker_snapshots,
)
from anomx.agent.store import (
    AI_PROVIDERS,
    AnomxHome,
    ProviderOption,
    SessionRecord,
    model_context_window,
    model_detail,
    normalize_thinking_intensity,
    provider_by_key,
    thinking_intensity_options,
)
from anomx.agent.terminal import markdown_to_terminal_rendered_lines
from anomx.agent.tool_manager import (
    ApprovalChoice,
    CommandApprovalRequest,
    discover_workspace_root,
)


class AgentState(StrEnum):
    """Top-level CLI states."""

    ONBOARDING = "Onboarding"
    ACCESS_CHECK = "Access Check"
    NEW_SESSION = "New Session"
    OPEN_SESSION = "Open Session"
    CONFIG = "Config"
    MODEL = "Model"
    INFO = "Info"
    SKILLS = "Skills"
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
    expansion_key: str = dataclass_field(default="", compare=False)


@dataclass(frozen=True)
class CommandSpec:
    """Prompt command entry."""

    command: str
    description: str


@dataclass(frozen=True)
class InfoRow:
    """A label/value row shown on the session info screen."""

    label: str
    value: str


@dataclass(frozen=True)
class SkillFormDraft:
    """Editable skill form state."""

    command: str = ""
    description: str = ""
    body: str = ""
    path: Path | None = None


@dataclass(frozen=True)
class PlatformConnectionDraft:
    """Editable platform connection form state."""

    url: str = ""
    email: str = ""
    password: str = ""


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
    question_request: QuestionRequest | None = None
    question_response: queue.SimpleQueue[QuestionResponse] | None = None


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
class SessionViewportState:
    """Resolved transcript viewport state for the current frame."""

    start: int
    scroll: int
    body_height: int
    rendered_line_count: int


@dataclass(frozen=True)
class SessionMouseAction:
    """Mouse action resolved from a curses mouse event."""

    kind: str
    value: int
    text: str = ""
    x_start: int = 0
    x_end: int = 0


@dataclass(frozen=True)
class SessionTextRow:
    """A rendered transcript row that can be selected with the mouse."""

    line_index: int
    y: int
    x: int
    width: int
    text: str


@dataclass(frozen=True)
class SessionSelectionPoint:
    """A transcript text coordinate."""

    line_index: int
    column: int


@dataclass(frozen=True)
class SessionTextSelection:
    """Active or retained transcript text selection."""

    anchor: SessionSelectionPoint
    focus: SessionSelectionPoint


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
    command_selected: int = 0
    command: str = ""
    submitted: str = ""
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
    CommandSpec("/open", "Open a stored session"),
    CommandSpec("/rename", "Rename the current session"),
    CommandSpec("/skills", "Create and open skills"),
    CommandSpec("/config", "Edit configuration"),
    CommandSpec("/model", "Change model"),
    CommandSpec("/info", "Show session information"),
    CommandSpec("/exit", "Exit Anomx"),
)
RUNNING_SESSION_COMMANDS = frozenset({"/skills", "/config", "/model", "/info"})

RUNNING_NOTICE = (
    "Agent is working · /commands available · messages blocked · Esc/Ctrl+C/Ctrl+X interrupt"
)
RUNNING_MESSAGE_BLOCKED_NOTICE = (
    "Agent is working · sending messages is blocked until the response finishes"
)
RUNNING_COMMAND_BLOCKED_NOTICE = (
    "Agent is working · this slash command is unavailable until the response finishes"
)
ABORT_AGENT_CONFIRM_NOTICE = (
    "Do you really want to abort the agent? Press {key_label} again to confirm."
)
INTERRUPTED_AGENT_NOTICE = "You have interrupted anomx."
MANUAL_INTERRUPT_MESSAGE = "You interrupted anomx manually"
EXIT_ANOMX_CONFIRM_NOTICE = "Do you really want to exit anomx? Press Ctrl+C again to confirm."
TABLE_BORDER_CHARS = frozenset("│┌┬┐├┼┤└┴┘─")
FILE_REFERENCE_LIMIT = 8
FILE_REFERENCE_SCAN_LIMIT = 5_000
FILE_REFERENCE_CACHE_SECONDS = 2.0
IGNORED_FILE_REFERENCE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".next",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "out",
        "venv",
    }
)


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
        self.cwd = (Path.cwd() if cwd is None else cwd).expanduser().resolve()
        self.workspace_root = discover_workspace_root(self.cwd)
        self.startup_provider = startup_provider
        self.startup_model = startup_model
        self.use_color = use_color
        self.session_allowed_commands: set[str] = set()
        self.session_rejected_commands: set[str] = set()
        self.agent_mode = AgentMode.parse(self.home.load_config().get("agent_mode"))
        self.runtime = AgentRuntime(
            self.home,
            self.cwd,
            self.session_allowed_commands,
            self.session_rejected_commands,
            self.agent_mode,
            workspace_root=self.workspace_root,
        )
        self.state = AgentState.ONBOARDING
        self._colors: dict[str, int] = {}
        self._prompt_placeholder = random.choice(PROMPT_PLACEHOLDERS)
        self._expanded_work_turns: set[str] = set()
        self._expanded_work_lines: set[str] = set()
        self._click_targets: dict[int, list[SessionMouseAction]] = {}
        self._session_text_rows: dict[int, SessionTextRow] = {}
        self._session_selection: SessionTextSelection | None = None
        self._session_selecting = False
        self._title_events: queue.SimpleQueue[tuple[str, str]] = queue.SimpleQueue()
        self._title_jobs: set[str] = set()
        self._file_reference_cache_at = 0.0
        self._file_reference_cache: tuple[Path, ...] = ()

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
        heartbeat_platform_connection(self.home)
        return int(curses.wrapper(self._run))

    def _run(self, stdscr: CursesWindow) -> int:
        self._configure_terminal(stdscr)

        while True:
            config = self.home.load_config()
            if not self._onboarding_complete(config):
                if not self._run_onboarding(stdscr):
                    return 1
                config = self.home.load_config()

            should_check_access = not self.home.is_repo_trusted(self.workspace_root)
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
            curses.mousemask(
                curses.ALL_MOUSE_EVENTS | getattr(curses, "REPORT_MOUSE_POSITION", 0)
            )
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
                "work_box": curses.color_pair(6) | curses.A_REVERSE,
                "table_header": curses.color_pair(6) | curses.A_REVERSE | curses.A_BOLD,
                "table_border": curses.color_pair(6) | curses.A_DIM,
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
                "work_box": curses.A_REVERSE,
                "table_header": curses.A_REVERSE | curses.A_BOLD,
                "table_border": curses.A_DIM,
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

        thinking_intensity = self._select_thinking_intensity(stdscr, provider, model)
        if thinking_intensity is None:
            return False

        config = self.home.load_config()
        config["onboarding_complete"] = True
        config["provider"] = provider.key
        config["model"] = model
        config["thinking_intensity"] = thinking_intensity
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
                    self.home.trust_repo(self.workspace_root)
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
        file_references: dict[str, str] = {}
        scroll = 0
        command_selected = 0
        file_selected = 0
        pinned_anchor: int | None = None
        exit_confirm_deadline = 0.0
        exit_notice = ""
        selection_notice_deadline = 0.0
        selection_notice = ""
        selection_notice_role = "ok"

        while True:
            if exit_confirm_deadline and time.monotonic() > exit_confirm_deadline:
                exit_confirm_deadline = 0.0
                exit_notice = ""
            if (
                selection_notice_deadline
                and time.monotonic() > selection_notice_deadline
            ):
                selection_notice_deadline = 0.0
                selection_notice = ""
            current_session = self._process_title_events(stdscr, current_session)
            messages = self._read_message_lines(current_session.path)
            command_suggestions = (
                self._filtered_commands(input_text) if input_text.startswith("/") else []
            )
            file_reference_token = self._active_file_reference_token(input_text, cursor)
            file_suggestions = (
                self._filtered_file_references(file_reference_token[2])
                if file_reference_token is not None
                else []
            )
            if command_suggestions:
                command_selected = min(command_selected, len(command_suggestions) - 1)
            else:
                command_selected = 0
            file_selected = (
                min(file_selected, len(file_suggestions) - 1) if file_suggestions else 0
            )
            viewport = self._draw_session(
                stdscr,
                current_session,
                messages,
                input_text,
                cursor,
                scroll,
                command_suggestions,
                command_selected,
                file_suggestions=file_suggestions,
                file_selected=file_selected,
                file_references=file_references,
                anchor_line=pinned_anchor,
                prompt_notice=exit_notice or selection_notice,
                prompt_notice_role="light" if exit_notice else selection_notice_role,
            )
            if viewport is not None:
                scroll = viewport.scroll
            key = self._read_prompt_key(stdscr)

            if self._is_shift_tab(key):
                self._cycle_agent_mode()
                continue
            if self._is_ctrl_c(key):
                if input_text:
                    input_text = ""
                    cursor = 0
                    command_selected = 0
                    file_selected = 0
                    file_references = {}
                    exit_confirm_deadline = 0.0
                    exit_notice = ""
                    continue
                now = time.monotonic()
                if exit_confirm_deadline and now <= exit_confirm_deadline:
                    return 0
                exit_confirm_deadline = now + 3.0
                exit_notice = EXIT_ANOMX_CONFIRM_NOTICE
                continue
            if self._is_option_left(key):
                cursor = self._previous_prompt_word(input_text, cursor)
                continue
            if self._is_option_right(key):
                cursor = self._next_prompt_word(input_text, cursor)
                continue

            if self._is_escape(key):
                exit_confirm_deadline = 0.0
                exit_notice = ""
                if input_text:
                    input_text = ""
                    cursor = 0
                    command_selected = 0
                    file_selected = 0
                    file_references = {}
                elif pinned_anchor is not None:
                    pinned_anchor = None
                elif self._session_selection is not None:
                    self._clear_session_selection()
                continue

            if key == curses.KEY_UP:
                if file_suggestions:
                    file_selected = max(0, file_selected - 1)
                    continue
                if command_suggestions:
                    command_selected = max(0, command_selected - 1)
                    continue
                moved_cursor = self._move_prompt_cursor_row(
                    stdscr,
                    input_text,
                    cursor,
                    direction=-1,
                )
                if moved_cursor != cursor:
                    cursor = moved_cursor
                    continue
                pinned_anchor = None
                scroll += 1
                continue
            if key == curses.KEY_DOWN:
                if file_suggestions:
                    file_selected = min(len(file_suggestions) - 1, file_selected + 1)
                    continue
                if command_suggestions:
                    command_selected = min(len(command_suggestions) - 1, command_selected + 1)
                    continue
                moved_cursor = self._move_prompt_cursor_row(
                    stdscr,
                    input_text,
                    cursor,
                    direction=1,
                )
                if moved_cursor != cursor:
                    cursor = moved_cursor
                    continue
                pinned_anchor = None
                scroll -= 1
                continue
            if key == curses.KEY_PPAGE:
                pinned_anchor = None
                scroll += 5
                continue
            if key == curses.KEY_NPAGE:
                pinned_anchor = None
                scroll -= 5
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
                    command_selected,
                    file_suggestions,
                    file_selected,
                )
                if mouse_action is None:
                    continue
                if mouse_action.kind == "cursor":
                    cursor = mouse_action.value
                elif mouse_action.kind == "scroll":
                    pinned_anchor = None
                    scroll += mouse_action.value
                elif mouse_action.kind == "toggle_work":
                    self._toggle_work_turn(mouse_action.text)
                elif mouse_action.kind == "toggle_work_line":
                    self._toggle_work_line(mouse_action.text)
                elif mouse_action.kind == "kill_process":
                    self.runtime.end_process(mouse_action.text, current_session.path)
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
                    file_selected = 0
                    file_references = {}
                elif mouse_action.kind == "file_reference":
                    if file_reference_token is None:
                        continue
                    input_text, cursor = self._insert_file_reference(
                        input_text,
                        cursor,
                        file_reference_token,
                        file_suggestions[mouse_action.value],
                        file_references,
                    )
                    file_selected = 0
                elif mouse_action.kind == "skill":
                    skill = self._skill_for_command(f"/{mouse_action.text}")
                    if skill is None:
                        continue
                    command_result = self._invoke_skill(
                        stdscr,
                        current_session,
                        skill,
                        skill.slash_command,
                    )
                    if command_result == "exit":
                        return 0
                    input_text = ""
                    cursor = 0
                    command_selected = 0
                    file_selected = 0
                    file_references = {}
                elif mouse_action.kind == "copy_selection":
                    selection_notice = mouse_action.text
                    selection_notice_role = "ok" if mouse_action.value else "danger"
                    selection_notice_deadline = time.monotonic() + 3.0
                continue
            if self._is_enter(key):
                if file_suggestions and file_reference_token is not None:
                    input_text, cursor = self._insert_file_reference(
                        input_text,
                        cursor,
                        file_reference_token,
                        file_suggestions[file_selected],
                        file_references,
                    )
                    file_selected = 0
                    continue
                submitted = input_text.strip()
                backend_message = self._backend_message_for_prompt(
                    submitted,
                    file_references,
                )
                submitted_file_references = dict(file_references)
                selected_command = command_selected
                input_text = ""
                cursor = 0
                file_references = {}
                command_selected = 0
                file_selected = 0
                if not submitted:
                    continue
                if submitted.startswith("/"):
                    command = self._submitted_command(
                        submitted,
                        command_suggestions,
                        selected_command,
                    )
                    command_result = self._handle_command(
                        stdscr,
                        command,
                        current_session,
                        submitted,
                    )
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
                    {
                        "message": submitted,
                        "backend_message": backend_message,
                        "file_references": submitted_file_references,
                    },
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
                    file_selected = 0
                continue
            if isinstance(key, str) and key.isprintable():
                input_text = input_text[:cursor] + key + input_text[cursor:]
                cursor += len(key)
                command_selected = 0
                file_selected = 0

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
            if event_type in {"user_message", "skill_invocation"} and str(
                payload.get("message", ""),
            ).strip():
                return True
        return False

    def _handle_command(
        self,
        stdscr: CursesWindow,
        command: str,
        current_session: SessionRecord,
        submitted: str = "",
    ) -> str | SessionRecord | None:
        if command == "/exit":
            return "exit"
        if command == "/open":
            return self._open_session_panel(stdscr, current_session)
        if command == "/rename":
            return self._rename_session(stdscr, current_session, submitted)
        if command == "/skills":
            self._run_skills_panel(stdscr, current_session)
            return None
        if command == "/config":
            self._run_config_panel(stdscr, current_session)
            return None
        if command == "/model":
            self._run_model_panel(stdscr, current_session)
            return None
        if command == "/info":
            self._run_info_panel(stdscr, current_session)
            return None
        if command == "/new":
            return self._create_session()
        skill = self._skill_for_command(command)
        if skill is not None:
            return self._invoke_skill(stdscr, current_session, skill, submitted or command)

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
                detail=(
                    f"{self._message_count_label(session.message_count)} · "
                    f"{session.created_at} · {session.provider}/{session.model}"
                ),
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

    def _rename_session(
        self,
        stdscr: CursesWindow,
        current_session: SessionRecord,
        submitted: str = "",
    ) -> SessionRecord | None:
        parts = submitted.strip().split(maxsplit=1)
        title = parts[1].strip() if len(parts) > 1 else ""
        if not title:
            prompted_title = self._prompt_text(
                stdscr,
                title="Rename Session",
                label="Title",
                default=current_session.title,
            )
            if prompted_title is None:
                return None
            title = prompted_title.strip()
        if not title or title == current_session.title:
            return None
        self.home.update_session_title(current_session.path, title)
        return replace(current_session, title=title)

    def _message_count_label(self, count: int) -> str:
        noun = "message" if count == 1 else "messages"
        return f"{count} {noun}"

    def _run_skills_panel(self, stdscr: CursesWindow, current_session: SessionRecord) -> None:
        self.state = AgentState.SKILLS
        while True:
            selected = self._menu(
                stdscr,
                "Skills",
                "Create or open a user skill",
                self._skills_menu_choices(),
            )
            if selected is None:
                self.state = AgentState.NEW_SESSION
                return
            if selected == "__create_skill__":
                self._create_user_skill(stdscr)
                continue
            skill = self._user_skill_by_command(selected)
            if skill is not None:
                self._run_skill_detail_panel(stdscr, skill)

    def _skills_menu_choices(self) -> tuple[MenuChoice, ...]:
        choices = [
            MenuChoice(
                "Create new Skill",
                "__create_skill__",
                "Define a global slash-command skill",
            )
        ]
        choices.extend(
            MenuChoice(f"/{skill.command}", skill.command, skill.description)
            for skill in self._user_skills()
            if not skill.hidden
        )
        return tuple(choices)

    def _create_user_skill(self, stdscr: CursesWindow) -> Skill | None:
        saved = self._run_skill_editor(stdscr, title="Create Skill")
        if saved is not None:
            self._message(stdscr, "Create Skill", f"Saved /{saved.command}.")
        return saved

    def _edit_user_skill(self, stdscr: CursesWindow, skill: Skill) -> Skill | None:
        return self._run_skill_editor(stdscr, title="Edit Skill", existing_skill=skill)

    def _run_skill_editor(
        self,
        stdscr: CursesWindow,
        *,
        title: str,
        existing_skill: Skill | None = None,
    ) -> Skill | None:
        draft = self._skill_form_draft(existing_skill)
        selected = 0
        while True:
            self._draw_skill_editor_panel(stdscr, title, draft, selected)
            self._footer(
                stdscr,
                "Esc Cancel · Ctrl+S Save · ↑↓ Navigate · Enter Next",
            )
            stdscr.refresh()
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key):
                return None
            if self._is_ctrl_s(key):
                saved = self._save_skill_draft(stdscr, draft, existing_skill)
                if saved is not None:
                    return saved
                continue
            if key == curses.KEY_UP:
                selected = max(0, selected - 1)
                continue
            if key == curses.KEY_DOWN:
                selected = min(2, selected + 1)
                continue
            if self._is_enter(key):
                selected = min(2, selected + 1)
            elif self._is_shift_enter(key):
                selected = max(0, selected - 1)
            elif self._is_backspace(key):
                draft = self._update_skill_draft(
                    draft,
                    self._skill_editor_field(selected),
                    self._skill_editor_value(draft, selected)[:-1],
                )
            elif isinstance(key, str) and key.isprintable():
                draft = self._update_skill_draft(
                    draft,
                    self._skill_editor_field(selected),
                    self._skill_editor_value(draft, selected) + key,
                )

    def _skill_form_draft(self, skill: Skill | None = None) -> SkillFormDraft:
        if skill is None:
            return SkillFormDraft()
        return SkillFormDraft(
            command=skill.command,
            description=skill.description,
            body=skill.body,
            path=skill.path,
        )

    def _save_skill_draft(
        self,
        stdscr: CursesWindow,
        draft: SkillFormDraft,
        existing_skill: Skill | None,
    ) -> Skill | None:
        command = normalize_skill_command(draft.command)
        if not command or not is_valid_skill_command(command):
            self._message(
                stdscr,
                "Skill",
                (
                    "Use letters, numbers, dashes, or underscores. "
                    "Commands must start with a letter or number."
                ),
            )
            return None
        original_command = existing_skill.command if existing_skill is not None else None
        if self._command_exists(command, exclude_command=original_command):
            self._message(stdscr, "Skill", f"/{command} already exists.")
            return None
        if not draft.description.strip():
            self._message(stdscr, "Skill", "Description is required.")
            return None
        if not draft.body.strip():
            self._message(stdscr, "Skill", "Skill instructions are required.")
            return None

        skill = Skill(
            command=command,
            title=command,
            description=draft.description.strip(),
            body=draft.body.strip(),
            source="user",
        )
        path = write_user_skill(self.home.skills_dir, skill)
        old_path = existing_skill.path if existing_skill is not None else draft.path
        if old_path is not None and old_path != path:
            with suppress(FileNotFoundError):
                old_path.unlink()
        return Skill(
            command=skill.command,
            title=skill.title,
            description=skill.description,
            body=skill.body,
            source=skill.source,
            hidden=skill.hidden,
            path=path,
        )

    def _draw_skill_editor_panel(
        self,
        stdscr: CursesWindow,
        title: str,
        draft: SkillFormDraft,
        selected: int,
    ) -> None:
        height, width = self._draw_shell(stdscr, title, self._skill_editor_path_line(draft))
        y = self._session_body_top()
        for index, row in enumerate(self._skill_editor_scalar_rows(draft)):
            label_attr = self._attr("accent") if index == selected else self._attr("light")
            value_attr = curses.A_NORMAL if index == selected else self._attr("light")
            self._draw_skill_form_row(stdscr, y, row, width, label_attr, value_attr)
            y += 1

        y += 2

        body_label_attr = self._attr("accent") if selected == 2 else self._attr("light")
        self._add(stdscr, y, 4, "Skill", width - 8, body_label_attr)
        y += 2
        body_attr = curses.A_NORMAL if selected == 2 else self._attr("light")
        visible_height = max(1, height - y - 2)
        display_lines = self._work_box_content_lines(draft.body, max(20, width - 8))
        start = max(0, len(display_lines) - visible_height)
        for offset, line in enumerate(display_lines[start : start + visible_height]):
            self._add(stdscr, y + offset, 4, line, width - 8, body_attr)

    def _skill_editor_scalar_rows(self, draft: SkillFormDraft) -> tuple[InfoRow, ...]:
        return (
            InfoRow("Command", self._skill_form_display_value("Command", draft.command)),
            InfoRow("Description", draft.description),
        )

    def _skill_editor_label(self, selected: int) -> str:
        return ("Command", "Description", "Skill")[selected]

    def _skill_editor_field(self, selected: int) -> str:
        return ("command", "description", "body")[selected]

    def _skill_editor_value(self, draft: SkillFormDraft, selected: int) -> str:
        return str(getattr(draft, self._skill_editor_field(selected)))

    def _update_skill_draft(
        self,
        draft: SkillFormDraft,
        field_name: str,
        value: str,
    ) -> SkillFormDraft:
        if field_name == "command":
            return replace(draft, command=value)
        if field_name == "description":
            return replace(draft, description=value)
        return replace(draft, body=value)

    def _skill_editor_path_line(self, draft: SkillFormDraft) -> str:
        return f"Stored at: {self._skill_editor_path(draft)}"

    def _skill_editor_path(self, draft: SkillFormDraft) -> str:
        command = normalize_skill_command(draft.command)
        if command:
            return str(self.home.skills_dir / f"{command}.md")
        if draft.path is not None:
            return str(draft.path)
        return str(self.home.skills_dir / "<command>.md")

    def _skill_form_display_value(self, active_label: str, active_value: str) -> str:
        if active_label == "Command":
            return f"/{active_value.removeprefix('/')}"
        return active_value

    def _draw_skill_form_row(
        self,
        stdscr: CursesWindow,
        y: int,
        row: InfoRow,
        width: int,
        label_attr: int,
        value_attr: int,
    ) -> None:
        label_width = min(24, max(12, width // 4))
        self._add(stdscr, y, 4, row.label, label_width - 4, label_attr)
        self._add(stdscr, y, label_width, row.value, width - label_width - 4, value_attr)

    def _run_skill_detail_panel(self, stdscr: CursesWindow, skill: Skill) -> None:
        current_skill = skill
        while True:
            self._draw_skill_detail_panel(stdscr, current_skill)
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key) or self._is_enter(key):
                return
            if self._is_ctrl_e(key) and self._skill_manageable(current_skill):
                edited = self._edit_user_skill(stdscr, current_skill)
                if edited is not None:
                    current_skill = edited
                continue
            should_delete = self._is_ctrl_d(key) and self._skill_manageable(current_skill)
            if should_delete and self._delete_user_skill(stdscr, current_skill):
                return

    def _draw_skill_detail_panel(self, stdscr: CursesWindow, skill: Skill) -> None:
        height, width = self._draw_shell(
            stdscr,
            f"Skill /{skill.command}",
            self._skill_detail_path_line(skill),
        )
        y = self._session_body_top()
        rows = (
            InfoRow("Command", f"/{skill.command}"),
            InfoRow("Description", skill.description),
        )
        for row in rows:
            self._draw_info_row(stdscr, y, row, width)
            y += 1

        y += 2
        self._add(stdscr, y, 4, "Skill", width - 8, self._attr("light"))
        y += 2
        for raw_line in skill.body.splitlines() or [""]:
            wrapped = textwrap.wrap(raw_line, width=max(20, width - 8)) or [""]
            for line in wrapped:
                if y >= height - 2:
                    self._add(stdscr, y, 4, "...", width - 8, self._attr("light"))
                    self._footer(stdscr, self._skill_detail_footer(skill))
                    stdscr.refresh()
                    return
                self._add(stdscr, y, 4, line, width - 8)
                y += 1
        self._footer(stdscr, self._skill_detail_footer(skill))
        stdscr.refresh()

    def _skill_manageable(self, skill: Skill) -> bool:
        return skill.source == "user" and skill.path is not None

    def _skill_detail_footer(self, skill: Skill) -> str:
        if self._skill_manageable(skill):
            return "Esc Back · Enter Back · Ctrl+E Edit · Ctrl+D Delete"
        return "Esc Back · Enter Back"

    def _skill_detail_path_line(self, skill: Skill) -> str:
        if skill.path is not None:
            return f"Stored at: {skill.path}"
        return "Stored at: bundled skill"

    def _delete_user_skill(self, stdscr: CursesWindow, skill: Skill) -> bool:
        selected = self._menu(
            stdscr,
            "Delete Skill",
            f"Delete /{skill.command}?",
            (
                MenuChoice("Cancel", "cancel", "Keep this skill"),
                MenuChoice("Delete Skill", "delete", "Remove this global skill"),
            ),
        )
        if selected != "delete":
            return False
        if skill.path is not None:
            with suppress(FileNotFoundError):
                skill.path.unlink()
        self._message(stdscr, "Delete Skill", f"Deleted /{skill.command}.")
        return True

    def _prompt_multiline_text(
        self,
        stdscr: CursesWindow,
        title: str,
        label: str,
        optional: bool = True,
    ) -> str | None:
        value = ""
        while True:
            height, width = self._draw_shell(stdscr, title, label)
            y = max(6, self._session_body_top())
            self._add(stdscr, y, 4, f"{label}:", width - 8, self._attr("bold"))
            y += 2
            visible_height = max(1, height - y - 3)
            display_lines = self._work_box_content_lines(value or "", max(20, width - 8))
            start = max(0, len(display_lines) - visible_height)
            for offset, line in enumerate(display_lines[start : start + visible_height]):
                self._add(stdscr, y + offset, 4, line, width - 8)
            self._footer(stdscr, "Esc Cancel · Enter New line · Ctrl+D Save")
            stdscr.refresh()
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key):
                return None if optional else ""
            if self._is_ctrl_d(key):
                if value.strip() or optional:
                    return value
                continue
            if self._is_enter(key):
                value += "\n"
            elif self._is_backspace(key):
                value = value[:-1]
            elif isinstance(key, str) and key.isprintable():
                value += key

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
            thinking_intensity = self._select_thinking_intensity(stdscr, provider, model)
            if thinking_intensity is None:
                self.state = AgentState.NEW_SESSION
                return False
            config["provider"] = provider.key
            config["model"] = model
            config["thinking_intensity"] = thinking_intensity
            config["onboarding_complete"] = True
            self.home.save_config(config)
        self.state = AgentState.NEW_SESSION
        return bool(model)

    def _run_info_panel(self, stdscr: CursesWindow, current_session: SessionRecord) -> None:
        self.state = AgentState.INFO
        while True:
            self._draw_info_panel(stdscr, current_session)
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key) or self._is_enter(key):
                self.state = AgentState.NEW_SESSION
                return

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
            if selected == "platform":
                self._configure_platform(stdscr, current_session)
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
        platform_connection = self.home.platform_connection()
        platform_choice = (
            MenuChoice(
                "Manage Platform",
                "platform",
                f"Connected to {platform_domain(platform_connection['url'])}",
            )
            if platform_connection is not None
            else MenuChoice(
                "Connect Platform",
                "platform",
                "Send agent activity, results, and findings to Anomx Platform",
            )
        )
        return (
            MenuChoice("Choose backend", "backend", "Select provider and enter API key"),
            MenuChoice("Choose model", "model", "Pick the model for the selected backend"),
            platform_choice,
            MenuChoice("History persistence", "history_persistence", "Store all sessions or none"),
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
        model_was_selected = False
        if provider.key != previous_provider:
            model = self._select_model(stdscr, provider)
            if model is None:
                return False
            selected_model = model
            model_was_selected = True
        elif not self._model_allowed(provider, selected_model):
            selected_model = provider.models[0]
            model_was_selected = True
        if model_was_selected:
            thinking_intensity = self._select_thinking_intensity(stdscr, provider, selected_model)
            if thinking_intensity is None:
                return False
            config["thinking_intensity"] = thinking_intensity
        config["provider"] = provider.key
        config["model"] = selected_model
        self.home.save_config(config)
        return True

    def _configure_platform(
        self,
        stdscr: CursesWindow,
        current_session: SessionRecord,
    ) -> bool:
        platform_connection = self.home.platform_connection()
        if platform_connection is not None:
            return self._run_platform_management_form(stdscr, platform_connection)

        result = self._run_platform_connection_form(stdscr)
        return bool(result)

    def _run_platform_connection_form(
        self,
        stdscr: CursesWindow,
    ) -> PlatformLoginResult | None:
        config = self.home.load_config()
        draft = PlatformConnectionDraft(
            url=str(config.get("platform_last_url") or ""),
            email=str(config.get("platform_last_email") or ""),
        )
        selected = 0 if not draft.url else 1 if not draft.email else 2
        error = ""
        while True:
            self._draw_platform_connection_form(stdscr, draft, selected, error=error)
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key):
                self._save_platform_form_defaults(draft)
                return None
            if key == curses.KEY_UP or self._is_shift_enter(key):
                selected = max(0, selected - 1)
                continue
            if key == curses.KEY_DOWN or key == "\t":
                selected = min(2, selected + 1)
                continue
            if self._is_enter(key):
                if selected < 2:
                    selected += 1
                    continue
                missing_row = self._missing_platform_form_row(draft)
                if missing_row is not None:
                    selected = missing_row
                    error = "Domain, email, and password are required."
                    continue
                self._save_platform_form_defaults(draft)
                try:
                    result = self._connect_platform_with_loading(stdscr, draft)
                except PlatformClientError as exc:
                    selected = 2
                    error = str(exc)
                    continue
                connection = {
                    "url": result.url,
                    "token": result.token,
                    "user_email": result.user_email,
                    "organization_url": result.organization_url,
                    "hostname": result.hostname,
                }
                self.home.set_platform_connection(
                    url=connection["url"],
                    token=connection["token"],
                    user_email=connection["user_email"],
                    organization_url=connection["organization_url"],
                    hostname=connection["hostname"],
                )
                self._run_platform_management_form(
                    stdscr,
                    connection,
                    initial_status="Connection alive.",
                    initial_status_role="ok",
                    check_connection=False,
                )
                return result
            if self._is_backspace(key):
                draft = self._update_platform_form_draft(
                    draft,
                    selected,
                    self._platform_form_value(draft, selected)[:-1],
                )
                error = ""
                self._save_platform_form_defaults(draft)
            elif isinstance(key, str) and key.isprintable():
                draft = self._update_platform_form_draft(
                    draft,
                    selected,
                    self._platform_form_value(draft, selected) + key,
                )
                error = ""
                self._save_platform_form_defaults(draft)

    def _run_platform_management_form(
        self,
        stdscr: CursesWindow,
        connection: dict[str, str],
        *,
        initial_status: str = "Checking connection...",
        initial_status_role: str = "normal",
        check_connection: bool = True,
    ) -> bool:
        draft = PlatformConnectionDraft(
            url=connection["url"],
            email=str(
                connection.get("user_email")
                or self.home.load_config().get("platform_last_email", "")
            ),
            password="*****",
        )
        check_result: queue.SimpleQueue[bool] = queue.SimpleQueue()
        worker: threading.Thread | None = None

        def run_check() -> None:
            check_result.put(heartbeat_platform_connection(self.home))

        if check_connection:
            worker = threading.Thread(target=run_check, daemon=True)
            worker.start()
        frame = 0
        status = initial_status
        status_role = initial_status_role
        with suppress(curses.error):
            stdscr.nodelay(True)
        try:
            while True:
                if worker is not None and not worker.is_alive():
                    with suppress(queue.Empty):
                        ok = check_result.get_nowait()
                        if ok:
                            status = "Connection alive."
                            status_role = "ok"
                        else:
                            status = "Connection check failed."
                            status_role = "danger"
                    worker.join(timeout=0)
                    worker = None
                self._draw_platform_connection_form(
                    stdscr,
                    draft,
                    2,
                    title="Manage Platform",
                    status=status,
                    status_role=status_role,
                    frame=frame,
                    footer="Esc Back · Enter Back · Ctrl+D Logout",
                    editable=False,
                )
                frame += 1
                try:
                    key = stdscr.get_wch()
                except curses.error:
                    key = None
                if key is None:
                    time.sleep(0.08)
                    continue
                if self._is_ctrl_d(key):
                    with suppress(curses.error):
                        stdscr.nodelay(False)
                    if self._confirm_platform_logout(stdscr, draft):
                        self.home.clear_platform_connection()
                        self._wait_platform_form_status(
                            stdscr,
                            draft,
                            "Logged out.",
                            "ok",
                            title="Manage Platform",
                            footer="Esc Back · Enter Continue",
                            editable=False,
                        )
                        return True
                    with suppress(curses.error):
                        stdscr.nodelay(True)
                    continue
                if self._is_escape(key) or self._is_ctrl_c(key) or self._is_enter(key):
                    return False
                time.sleep(0.08)
        finally:
            with suppress(curses.error):
                stdscr.nodelay(False)

    def _confirm_platform_logout(
        self,
        stdscr: CursesWindow,
        draft: PlatformConnectionDraft,
    ) -> bool:
        while True:
            self._draw_platform_connection_form(
                stdscr,
                draft,
                2,
                title="Manage Platform",
                status="Logout this CLI agent?",
                status_role="normal",
                footer="Esc Cancel · Enter Logout",
                editable=False,
            )
            key = stdscr.get_wch()
            if self._is_enter(key):
                return True
            if self._is_escape(key) or self._is_ctrl_c(key):
                return False

    def _wait_platform_form_status(
        self,
        stdscr: CursesWindow,
        draft: PlatformConnectionDraft,
        status: str,
        status_role: str,
        *,
        title: str,
        footer: str = "Esc Back · Enter Continue",
        editable: bool = False,
    ) -> None:
        while True:
            self._draw_platform_connection_form(
                stdscr,
                draft,
                2,
                title=title,
                status=status,
                status_role=status_role,
                footer=footer,
                editable=editable,
            )
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key) or self._is_enter(key):
                return

    def _missing_platform_form_row(self, draft: PlatformConnectionDraft) -> int | None:
        for index, value in enumerate((draft.url, draft.email, draft.password)):
            if not value.strip():
                return index
        return None

    def _platform_form_field(self, selected: int) -> str:
        return ("url", "email", "password")[selected]

    def _platform_form_value(self, draft: PlatformConnectionDraft, selected: int) -> str:
        return str(getattr(draft, self._platform_form_field(selected)))

    def _update_platform_form_draft(
        self,
        draft: PlatformConnectionDraft,
        selected: int,
        value: str,
    ) -> PlatformConnectionDraft:
        field_name = self._platform_form_field(selected)
        if field_name == "url":
            return replace(draft, url=value)
        if field_name == "email":
            return replace(draft, email=value)
        return replace(draft, password=value)

    def _save_platform_form_defaults(self, draft: PlatformConnectionDraft) -> None:
        self.home.set_platform_form_defaults(url=draft.url, email=draft.email)

    def _draw_platform_connection_form(
        self,
        stdscr: CursesWindow,
        draft: PlatformConnectionDraft,
        selected: int,
        *,
        title: str = "Connect Platform",
        error: str = "",
        status: str = "",
        status_role: str = "normal",
        connecting: bool = False,
        frame: int = 0,
        footer: str | None = None,
        editable: bool = True,
    ) -> None:
        height, width = self._draw_shell(
            stdscr,
            title,
            (
                "Set the URL of an Anomx Platform. The CLI can then write future "
                "results and findings there."
            ),
        )
        rows = (
            ("Domain", draft.url),
            ("Email", draft.email),
            ("Password", "*" * len(draft.password)),
        )
        label_width = 12
        start_x = 4
        value_x = start_x + label_width
        start_y = max(self._session_body_top(), (height // 2) - 2)
        for index, (label, value) in enumerate(rows):
            label_attr = (
                self._attr("accent")
                if editable and index == selected
                else self._attr("light")
            )
            y = start_y + index
            self._add(stdscr, y, start_x, f"{label}:", label_width, label_attr)
            self._add(stdscr, y, value_x, value, width - value_x - 4)

        status_y = start_y + len(rows) + 1
        if connecting:
            status = f"Connecting{'.' * ((frame // 4) % 4)}"
            self._add(
                stdscr,
                status_y,
                start_x,
                status,
                width - start_x - 4,
                curses.A_NORMAL,
            )
        elif status:
            display_status = textwrap.shorten(
                status,
                width=max(20, width - 8),
                placeholder="...",
            )
            self._add(
                stdscr,
                status_y,
                start_x,
                display_status,
                width - start_x - 4,
                self._attr(status_role),
            )
        elif error:
            display_error = textwrap.shorten(
                error,
                width=max(20, width - 8),
                placeholder="...",
            )
            self._add(
                stdscr,
                status_y,
                start_x,
                display_error,
                width - start_x - 4,
                self._attr("danger"),
            )

        footer_text = footer or (
            "Esc Cancel · ↑↓ Navigate · Enter for Login"
            if selected == 2
            else "Esc Cancel · ↑↓ Navigate · Enter Next"
        )
        self._footer(stdscr, "Please wait" if connecting else footer_text)
        if editable and not connecting:
            selected_value = rows[selected][1]
            cursor_x = min(width - 5, value_x + len(selected_value))
            with suppress(curses.error, AttributeError):
                curses.curs_set(1)
                stdscr.move(start_y + selected, cursor_x)
        else:
            with suppress(curses.error):
                curses.curs_set(0)
        stdscr.refresh()

    def _connect_platform_with_loading(
        self,
        stdscr: CursesWindow,
        draft: PlatformConnectionDraft,
    ) -> PlatformLoginResult:
        results: queue.SimpleQueue[
            tuple[str, PlatformLoginResult | PlatformClientError]
        ] = queue.SimpleQueue()

        def run_connect() -> None:
            try:
                results.put(
                    ("ok", connect_platform(draft.url, draft.email, draft.password))
                )
            except PlatformClientError as exc:
                results.put(("error", exc))
            except Exception as exc:
                error = PlatformClientError(f"Platform connection failed: {exc}")
                results.put(("error", error))

        worker = threading.Thread(target=run_connect, daemon=True)
        worker.start()
        frame = 0
        with suppress(curses.error):
            stdscr.nodelay(True)
        try:
            while worker.is_alive():
                self._draw_platform_connection_form(
                    stdscr,
                    draft,
                    2,
                    connecting=True,
                    frame=frame,
                )
                frame += 1
                with suppress(curses.error):
                    stdscr.get_wch()
                time.sleep(0.08)
        finally:
            with suppress(curses.error):
                stdscr.nodelay(False)

        worker.join(timeout=0)
        try:
            kind, payload = results.get_nowait()
        except queue.Empty as exc:
            raise PlatformClientError("Platform connection failed.") from exc
        if isinstance(payload, PlatformClientError):
            raise payload
        if kind == "error":
            raise PlatformClientError("Platform connection failed.")
        return payload

    def _draw_platform_connect_loading(
        self,
        stdscr: CursesWindow,
        frame: int,
    ) -> None:
        self._draw_platform_connection_form(
            stdscr,
            PlatformConnectionDraft(password=" "),
            2,
            connecting=True,
            frame=frame,
        )

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

    def _select_thinking_intensity(
        self,
        stdscr: CursesWindow,
        provider: ProviderOption,
        model: str,
    ) -> str | None:
        options = thinking_intensity_options(provider.key, model)
        if not options:
            return "auto"
        choices = tuple(MenuChoice(option.label, option.value, option.detail) for option in options)
        selected = self._menu(stdscr, "Thinking Intensity", model, choices)
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
        scroll: int = 0,
        anchor_line: int | None = None,
        prompt_notice: str = "",
        prompt_notice_role: str = "light",
    ) -> str | None:
        selected = 0
        current_scroll = scroll
        with suppress(curses.error):
            stdscr.nodelay(False)
        try:
            while True:
                if autonomous_value is not None and self.agent_mode in {
                    AgentMode.AUTONOMOUS,
                    AgentMode.FULL_CONTROL,
                }:
                    return autonomous_value
                messages = self._read_message_lines(session.path)
                panel = BottomPanel(title, subtitle, choices, selected)
                viewport = self._draw_session(
                    stdscr,
                    session,
                    messages,
                    "",
                    0,
                    current_scroll,
                    bottom_panel=panel,
                    anchor_line=anchor_line,
                    prompt_notice=prompt_notice,
                    prompt_notice_role=prompt_notice_role,
                )
                if viewport is not None:
                    current_scroll = viewport.scroll
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
                    panel_viewport = self._bottom_panel_viewport(stdscr, panel)
                    page_size = max(1, len(panel_viewport.visible_indices))
                    selected = max(0, selected - page_size)
                elif key == curses.KEY_NPAGE:
                    panel_viewport = self._bottom_panel_viewport(stdscr, panel)
                    page_size = max(1, len(panel_viewport.visible_indices))
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
        subtitle: str | tuple[str, ...] = "",
        plan_steps: tuple[PlanStep, ...] = (),
        header_meta: str = "",
    ) -> tuple[int, int]:
        with suppress(curses.error):
            curses.curs_set(0)
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        self._paint_background(stdscr)
        self._draw_header_box(stdscr, title, subtitle, plan_steps, header_meta)
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
        subtitle: str | tuple[str, ...] = "",
        plan_steps: tuple[PlanStep, ...] = (),
        header_meta: str = "",
    ) -> None:
        _, width = stdscr.getmaxyx()
        right_text = self._header_right_text(header_meta)
        top = 1
        subtitle_lines = self._header_subtitle_lines(subtitle)
        bottom = self._header_bottom(plan_steps, len(subtitle_lines))
        horizontal = "─" * max(1, width - 6)
        self._add(stdscr, top, 2, f"╭{horizontal}╮", width - 4, self._attr("accent"))
        for y in range(top + 1, bottom):
            self._add(stdscr, y, 2, "│", 1, self._attr("accent"))
            self._add(stdscr, y, max(2, width - 3), "│", 1, self._attr("accent"))
        self._add(stdscr, bottom, 2, f"╰{horizontal}╯", width - 4, self._attr("accent"))

        brand = "Anomx"
        descriptor = "Anomaly Detection and Data Analysis Agent"
        right_text = self._fit_header_right_text(right_text, max(1, width - 8))
        right_x = max(4, width - len(right_text) - 5)
        descriptor_x = 4 + len(brand) + 2
        self._add(stdscr, top + 1, 4, brand, width - 8, self._attr("accent"))
        self._add(
            stdscr,
            top + 1,
            descriptor_x,
            descriptor,
            max(1, right_x - descriptor_x - 2),
            self._attr("light"),
        )
        self._add(
            stdscr,
            top + 1,
            right_x,
            right_text,
            len(right_text),
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
        for index, line in enumerate(subtitle_lines):
            self._add(
                stdscr,
                top + 3 + index,
                4,
                line,
                width - 8,
                self._attr("light"),
            )
        if plan_steps:
            plan_start_y = top + 4 + len(subtitle_lines)
            for index, step in enumerate(plan_steps):
                y = plan_start_y + index
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

    def _header_right_text(self, header_meta: str = "") -> str:
        version = f"v{__version__}"
        meta = header_meta.strip()
        return f"{meta} · {version}" if meta else version

    def _fit_header_right_text(self, text: str, width: int) -> str:
        safe_width = max(1, width)
        if len(text) <= safe_width:
            return text
        if safe_width <= 1:
            return text[:safe_width]
        return f"…{text[-(safe_width - 1):]}"

    def _header_subtitle_lines(self, subtitle: str | tuple[str, ...]) -> tuple[str, ...]:
        if isinstance(subtitle, str):
            return (subtitle,) if subtitle else ()
        return tuple(line for line in subtitle if line)

    def _header_bottom(
        self,
        plan_steps: tuple[PlanStep, ...] = (),
        subtitle_line_count: int = 0,
    ) -> int:
        base_bottom = 5 + max(0, subtitle_line_count - 1)
        if not plan_steps:
            return base_bottom
        return base_bottom + 1 + len(plan_steps)

    def _session_body_top(
        self,
        plan_steps: tuple[PlanStep, ...] = (),
        subtitle_line_count: int = 1,
    ) -> int:
        return self._header_bottom(plan_steps, subtitle_line_count) + 2

    def _strike_text(self, text: str) -> str:
        return "".join(
            f"{character}\u0336" if character != " " else character for character in text
        )

    def _draw_access_check(self, stdscr: CursesWindow, selected: int) -> None:
        height, width = self._draw_shell(stdscr, "Access Check", "Accessing workspace")
        self._add(stdscr, 8, 4, str(self.workspace_root), width - 8, self._attr("bold"))
        y = 9
        if self.cwd != self.workspace_root:
            self._add(
                stdscr,
                y,
                4,
                f"Started in: {self.cwd}",
                width - 8,
                self._attr("light"),
            )
            y += 1

        copy = (
            "Quick safety check: Is this a project you created or one you trust? "
            "If not, take a moment to review what's in this folder first."
        )
        y += 2
        for line in textwrap.wrap(copy, width=max(24, width - 8)):
            self._add(stdscr, y, 4, line, width - 8)
            y += 1
        y += 1
        self._add(
            stdscr,
            y,
            4,
            "Anomx will be able to read, edit, and execute files in this workspace.",
            width - 8,
        )
        y += 2
        self._add(stdscr, y, 4, "Security guide", width - 8, self._attr("light"))
        y += 2

        choices = ("Yes, I trust this workspace", "No, exit")
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

    def _draw_info_panel(self, stdscr: CursesWindow, session: SessionRecord) -> None:
        config = self.home.load_config()
        provider = str(config.get("provider", session.provider))
        model = str(config.get("model", session.model))
        header_lines = self._session_header_lines(session, model)
        height, width = self._draw_shell(
            stdscr,
            "Info",
            header_lines,
            header_meta=self._session_header_meta(session, provider, model),
        )
        y = self._session_body_top(subtitle_line_count=len(header_lines))
        self._add(stdscr, y, 4, "Current location", width - 8, self._attr("accent"))
        y += 2
        for row in self._session_location_rows(session):
            self._draw_info_row(stdscr, y, row, width)
            y += 1

        y += 2
        self._add(stdscr, y, 4, "Approved commands", width - 8, self._attr("accent"))
        y += 2
        for row in self._approved_command_rows(session):
            if y >= height - 2:
                break
            self._draw_info_row(stdscr, y, row, width)
            y += 1

        self._footer(stdscr, "Esc Back · Enter Back")
        stdscr.refresh()

    def _draw_info_row(
        self,
        stdscr: CursesWindow,
        y: int,
        row: InfoRow,
        width: int,
    ) -> None:
        label_width = min(24, max(12, width // 4))
        self._add(stdscr, y, 4, row.label, label_width - 4, self._attr("light"))
        self._add(stdscr, y, label_width, row.value, width - label_width - 4)

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
        file_suggestions: list[MenuChoice] | None = None,
        file_selected: int = 0,
        file_references: Mapping[str, str] | None = None,
        bottom_panel: BottomPanel | None = None,
        working_text: str | None = None,
        working_deadline: float | None = None,
        working_frame: int = 0,
        anchor_line: int | None = None,
        prompt_notice: str = "",
        prompt_notice_role: str = "light",
    ) -> SessionViewportState:
        config = self.home.load_config()
        provider = str(config.get("provider", session.provider))
        model = str(config.get("model", session.model))
        session_events = self.home.read_session_events(session.path)
        plan_steps = latest_plan_steps(session_events)
        workers = worker_snapshots(session_events) if bottom_panel is None else ()
        processes = running_process_snapshots(session_events) if bottom_panel is None else ()
        header_lines = self._session_header_lines(session, model)
        height, width = self._draw_shell(
            stdscr,
            session.title,
            header_lines,
            plan_steps,
            header_meta=self._session_header_meta(session, provider, model),
        )
        layout = self._prompt_layout(stdscr, input_text)
        suggestions = command_suggestions or []
        activity_row_count = len(workers) + len(processes)
        activity_panel_height = activity_row_count + (1 if activity_row_count else 0)
        body_top = self._session_body_top(
            plan_steps,
            subtitle_line_count=len(header_lines),
        )
        body_bottom = max(body_top + 1, layout.top_line - activity_panel_height)
        body_height = max(1, body_bottom - body_top)
        command_panel = (
            self._command_bottom_panel(suggestions, command_selected)
            if bottom_panel is None
            else None
        )
        file_panel = (
            self._file_reference_bottom_panel(file_suggestions or [], file_selected)
            if bottom_panel is None
            else None
        )
        active_bottom_panel = bottom_panel or file_panel or command_panel
        rendered = self._session_rendered_lines(
            session,
            messages,
            max(20, width - 8),
            self._working_status_text(working_text, working_deadline),
        )
        rendered_line_count = len(rendered)
        if anchor_line is None:
            scroll = self._clamp_session_scroll(scroll, rendered_line_count, body_height)
            start = self._session_view_start(scroll, rendered_line_count, body_height)
        else:
            start = max(0, min(anchor_line, self._session_max_start(rendered_line_count)))
            scroll = self._session_scroll_for_start(start, rendered_line_count, body_height)
        visible = rendered[start : start + body_height]
        self._click_targets = {}
        self._session_text_rows = {}
        for offset, line in enumerate(visible):
            y = body_top + offset
            line_index = start + offset
            self._session_text_rows[y] = SessionTextRow(
                line_index=line_index,
                y=y,
                x=4,
                width=width - 8,
                text=line.text,
            )
            if line.role == "work_summary":
                self._add_click_target(y, SessionMouseAction("toggle_work", 0, line.meta))
            elif line.expansion_key:
                self._add_click_target(
                    y,
                    SessionMouseAction("toggle_work_line", 0, line.expansion_key),
                )
            if line.role == "working":
                self._draw_working_line(
                    stdscr,
                    y,
                    4,
                    line.text,
                    width - 8,
                    working_frame,
                )
                self._draw_session_selection(stdscr, y, 4, line_index, line.text, width - 8)
                continue
            if line.role == "work_box":
                self._draw_work_box_line(stdscr, y, 4, line.text, width - 8)
                self._draw_session_selection(stdscr, y, 4, line_index, line.text, width - 8)
                continue
            if line.role in {"table_header", "table_row"}:
                self._draw_table_line(stdscr, y, 4, line.text, width - 8, line.role)
                self._draw_session_selection(stdscr, y, 4, line_index, line.text, width - 8)
                continue
            attr = self._line_attr(line.role)
            self._add(stdscr, y, 4, line.text, width - 8, attr)
            self._draw_session_selection(stdscr, y, 4, line_index, line.text, width - 8)

        if self._should_draw_start_hints(
            messages,
            input_text,
            active_bottom_panel,
            working_text,
            plan_steps,
        ):
            self._draw_start_hints(stdscr, body_top, body_bottom, width)

        if activity_row_count:
            self._draw_running_workers(
                stdscr,
                workers,
                body_bottom,
                working_frame,
                processes,
            )
        if active_bottom_panel is not None:
            self._draw_bottom_panel(stdscr, active_bottom_panel, input_text)
        self._draw_prompt_bar(
            stdscr,
            input_text,
            cursor,
            prompt_notice,
            prompt_notice_role,
            file_references,
        )
        stdscr.refresh()
        return SessionViewportState(start, scroll, body_height, rendered_line_count)

    def _should_draw_start_hints(
        self,
        messages: list[MessageLine],
        input_text: str,
        active_bottom_panel: BottomPanel | None,
        working_text: str | None,
        plan_steps: tuple[PlanStep, ...],
    ) -> bool:
        return (
            not messages
            and not input_text
            and active_bottom_panel is None
            and working_text is None
            and not plan_steps
        )

    def _draw_start_hints(
        self,
        stdscr: CursesWindow,
        body_top: int,
        body_bottom: int,
        width: int,
    ) -> None:
        skills = self._starter_skills()
        if not skills:
            return

        available_height = max(1, body_bottom - body_top)
        card_height = 7
        gap = 3
        if width >= 96:
            card_width = min(34, max(24, (width - 8 - (gap * 2)) // 3))
            total_width = (card_width * len(skills)) + (gap * (len(skills) - 1))
            start_x = max(4, (width - total_width) // 2)
            start_y = body_top + max(0, (available_height - card_height) // 2)
            for index, skill in enumerate(skills):
                self._draw_start_hint_card(
                    stdscr,
                    start_y,
                    start_x + (index * (card_width + gap)),
                    card_width,
                    card_height,
                    skill,
                )
            return

        card_width = min(max(24, width - 8), 42)
        total_height = (card_height * len(skills)) + (len(skills) - 1)
        start_x = max(4, (width - card_width) // 2)
        start_y = body_top + max(0, (available_height - total_height) // 2)
        for index, skill in enumerate(skills):
            self._draw_start_hint_card(
                stdscr,
                start_y + (index * (card_height + 1)),
                start_x,
                card_width,
                card_height,
                skill,
            )

    def _draw_start_hint_card(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        width: int,
        height: int,
        skill: Skill,
    ) -> None:
        inner_width = max(1, width - 4)
        horizontal = "─" * max(1, width - 2)
        self._add(stdscr, y, x, f"╭{horizontal}╮", width, self._attr("accent"))
        for offset in range(1, height - 1):
            self._add(stdscr, y + offset, x, "│", 1, self._attr("accent"))
            self._add(stdscr, y + offset, x + width - 1, "│", 1, self._attr("accent"))
        self._add(stdscr, y + height - 1, x, f"╰{horizontal}╯", width, self._attr("accent"))
        for offset in range(height):
            self._add_click_target(
                y + offset,
                SessionMouseAction("skill", 0, skill.command, x, x + width),
            )

        self._add(stdscr, y + 1, x + 2, skill.title, inner_width, self._attr("bold"))
        description_lines = textwrap.wrap(
            skill.description,
            width=inner_width,
            break_long_words=True,
            break_on_hyphens=False,
        )[:3]
        for index, line in enumerate(description_lines):
            self._add(stdscr, y + 3 + index, x + 2, line, inner_width, self._attr("light"))

    def _line_attr(self, role: str) -> int:
        if role == "user":
            return self._attr("accent")
        if role == "meta_accent":
            return self._attr("accent")
        if role in {"meta", "tool", "work_summary", "worker", "approved", "notice"}:
            return self._attr("light")
        if role == "work_box":
            return self._attr("work_box")
        if role == "table_header":
            return self._attr("table_header")
        if role == "table_border":
            return self._attr("table_border")
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

    def _session_max_start(self, rendered_line_count: int) -> int:
        return max(0, rendered_line_count - 1)

    def _session_bottom_start(self, rendered_line_count: int, body_height: int) -> int:
        return max(0, rendered_line_count - max(1, body_height))

    def _session_scroll_bounds(
        self,
        rendered_line_count: int,
        body_height: int,
    ) -> tuple[int, int]:
        max_start = self._session_max_start(rendered_line_count)
        bottom_start = self._session_bottom_start(rendered_line_count, body_height)
        return bottom_start - max_start, bottom_start

    def _clamp_session_scroll(
        self,
        scroll: int,
        rendered_line_count: int,
        body_height: int,
    ) -> int:
        min_scroll, max_scroll = self._session_scroll_bounds(rendered_line_count, body_height)
        return max(min_scroll, min(scroll, max_scroll))

    def _session_view_start(
        self,
        scroll: int,
        rendered_line_count: int,
        body_height: int,
    ) -> int:
        clamped_scroll = self._clamp_session_scroll(scroll, rendered_line_count, body_height)
        bottom_start = self._session_bottom_start(rendered_line_count, body_height)
        max_start = self._session_max_start(rendered_line_count)
        return max(0, min(max_start, bottom_start - clamped_scroll))

    def _session_scroll_for_start(
        self,
        start: int,
        rendered_line_count: int,
        body_height: int,
    ) -> int:
        max_start = self._session_max_start(rendered_line_count)
        clamped_start = max(0, min(start, max_start))
        bottom_start = self._session_bottom_start(rendered_line_count, body_height)
        return self._clamp_session_scroll(
            bottom_start - clamped_start,
            rendered_line_count,
            body_height,
        )

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

    def _session_header_lines(
        self,
        session: SessionRecord,
        model: str,
    ) -> tuple[str, ...]:
        location_line = self._session_location_line(session)
        context_status = self._context_status(session, model)
        if context_status:
            return (f"{location_line} · {context_status}",)
        return (location_line,)

    def _session_location_line(self, session: SessionRecord) -> str:
        return f"Location: {session.cwd or self.cwd}"

    def _session_header_meta(self, session: SessionRecord, provider: str, model: str) -> str:
        model_label = self._model_header_label(provider, model)
        return f"{session.session_id[:8]} · {provider}/{model_label}"

    def _model_header_label(self, provider: str, model: str) -> str:
        marker = self._thinking_intensity_marker(provider, model)
        return f"{model} {marker}" if marker else model

    def _thinking_intensity_marker(self, provider: str, model: str) -> str:
        config = self.home.load_config()
        intensity = normalize_thinking_intensity(config.get("thinking_intensity"))
        supported = {option.value for option in thinking_intensity_options(provider, model)}
        if intensity not in supported:
            return ""
        return {
            "low": "(L)",
            "medium": "(M)",
            "high": "(H)",
            "xhigh": "(X)",
        }.get(intensity, "")

    def _context_status(self, session: SessionRecord, model: str) -> str:
        context_window = model_context_window(model)
        if context_window is None or not self._has_user_messages(session.path):
            return ""

        used_tokens = self.runtime.estimate_session_context_tokens(session.path)
        percent_used = context_usage_percent(used_tokens, context_window)
        return f"{percent_used}% Context"

    def _estimate_context_tokens(self, session_path: Path) -> int:
        return self.runtime.estimate_session_context_tokens(session_path)

    def _draw_working_line(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        width: int,
        frame: int,
    ) -> None:
        dots = "" if self._is_waiting_status_text(text) else "." * ((frame // 4) % 4)
        self._add(stdscr, y, x, f"{text}{dots}", width, self._attr("light"))

    def _is_waiting_status_text(self, text: str) -> bool:
        return text == "Waiting" or text.startswith("Waiting ")

    def _draw_work_box_line(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        width: int,
    ) -> None:
        self._add(stdscr, y, x, text.ljust(max(0, width)), width, self._attr("work_box"))

    def _draw_table_line(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        width: int,
        role: str,
    ) -> None:
        content_attr = self._attr("table_header") if role == "table_header" else curses.A_NORMAL
        border_attr = self._attr("table_border")
        start = 0
        current_is_border = bool(text) and text[0] in TABLE_BORDER_CHARS

        for index, character in enumerate(text):
            is_border = character in TABLE_BORDER_CHARS
            if is_border == current_is_border:
                continue
            attr = border_attr if current_is_border else content_attr
            self._add(stdscr, y, x + start, text[start:index], width - start, attr)
            start = index
            current_is_border = is_border

        if start < len(text):
            attr = border_attr if current_is_border else content_attr
            self._add(stdscr, y, x + start, text[start:], width - start, attr)

    def _draw_session_selection(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        line_index: int,
        text: str,
        width: int,
    ) -> None:
        span = self._selection_span_for_line(line_index, len(text))
        if span is None:
            return
        start, end = span
        if end <= start:
            return
        selected_width = min(end, width) - start
        if selected_width <= 0:
            return
        self._add(
            stdscr,
            y,
            x + start,
            text[start:end],
            selected_width,
            self._attr("selected"),
        )

    def _selection_span_for_line(
        self,
        line_index: int,
        text_length: int,
    ) -> tuple[int, int] | None:
        if self._session_selection is None:
            return None
        start, end = self._normalized_selection_points(self._session_selection)
        if line_index < start.line_index or line_index > end.line_index:
            return None
        if start.line_index == end.line_index:
            return (
                max(0, min(text_length, start.column)),
                max(0, min(text_length, end.column)),
            )
        if line_index == start.line_index:
            return max(0, min(text_length, start.column)), text_length
        if line_index == end.line_index:
            return 0, max(0, min(text_length, end.column))
        return 0, text_length

    def _normalized_selection_points(
        self,
        selection: SessionTextSelection,
    ) -> tuple[SessionSelectionPoint, SessionSelectionPoint]:
        if (
            selection.anchor.line_index,
            selection.anchor.column,
        ) <= (
            selection.focus.line_index,
            selection.focus.column,
        ):
            return selection.anchor, selection.focus
        return selection.focus, selection.anchor

    def _session_text_point_at(self, x: int, y: int) -> SessionSelectionPoint | None:
        row = self._session_text_rows.get(y)
        if row is None:
            return None
        return self._selection_point_for_row(row, x)

    def _nearest_session_text_point(self, x: int, y: int) -> SessionSelectionPoint | None:
        if not self._session_text_rows:
            return None
        row = self._session_text_rows.get(y)
        if row is None:
            row = min(
                self._session_text_rows.values(),
                key=lambda candidate: abs(candidate.y - y),
            )
        return self._selection_point_for_row(row, x)

    def _selection_point_for_row(
        self,
        row: SessionTextRow,
        x: int,
    ) -> SessionSelectionPoint:
        column = max(0, min(len(row.text), x - row.x))
        return SessionSelectionPoint(row.line_index, column)

    def _selected_session_text(self) -> str:
        if self._session_selection is None:
            return ""
        rows_by_index = {
            row.line_index: row.text
            for row in self._session_text_rows.values()
        }
        start, end = self._normalized_selection_points(self._session_selection)
        pieces: list[str] = []
        for line_index in range(start.line_index, end.line_index + 1):
            text = rows_by_index.get(line_index, "")
            if line_index == start.line_index == end.line_index:
                pieces.append(text[start.column : end.column])
            elif line_index == start.line_index:
                pieces.append(text[start.column :])
            elif line_index == end.line_index:
                pieces.append(text[: end.column])
            else:
                pieces.append(text)
        return "\n".join(pieces)

    def _clear_session_selection(self) -> None:
        self._session_selection = None
        self._session_selecting = False

    def _copy_to_clipboard(self, text: str) -> bool:
        if not text:
            return False
        clipboard_commands = (
            ("pbcopy",),
            ("wl-copy",),
            ("xclip", "-selection", "clipboard"),
            ("xsel", "--clipboard", "--input"),
        )
        for command in clipboard_commands:
            if shutil.which(command[0]) is None:
                continue
            try:
                subprocess.run(
                    command,
                    input=text,
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                    timeout=2,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            return True
        return False

    def _draw_running_workers(
        self,
        stdscr: CursesWindow,
        workers: tuple[WorkerAgentSnapshot, ...],
        start_y: int,
        frame: int,
        processes: tuple[AsyncProcessSnapshot, ...] = (),
    ) -> None:
        _, width = stdscr.getmaxyx()
        panel_width = max(1, width - 4)
        self._add(stdscr, start_y, 2, "─" * panel_width, panel_width, self._attr("light"))
        for offset, worker in enumerate(workers, start=1):
            right_text = self._worker_right_text(worker)
            right_x = max(4, width - len(right_text) - 4) if right_text else width
            left_width = max(1, right_x - 6)
            text = self._worker_left_text(worker, frame)
            self._add(stdscr, start_y + offset, 4, text, left_width, self._attr("light"))
            if right_text:
                self._add(
                    stdscr,
                    start_y + offset,
                    right_x,
                    right_text,
                    len(right_text),
                    self._attr("light"),
                )
        process_start = len(workers) + 1
        for offset, process in enumerate(processes, start=process_start):
            right_text = "Click to kill"
            right_x = max(4, width - len(right_text) - 4)
            left_width = max(1, right_x - 6)
            label = process.statement.strip() or process.command
            if process.source == "command":
                noun = "Command"
            elif process.source == "worker_command":
                owner = process.owner_name or process.owner_id or "Worker"
                noun = f"Worker Command · {owner}"
            else:
                noun = "Process"
            text = f"{noun} ({process.process_id}) · {label}"
            y = start_y + offset
            self._add(stdscr, y, 4, text, left_width, self._attr("light"))
            self._add(
                stdscr,
                y,
                right_x,
                right_text,
                len(right_text),
                self._attr("light"),
            )
            self._add_click_target(
                y,
                SessionMouseAction(
                    "kill_process",
                    0,
                    process.process_id,
                    right_x,
                    right_x + len(right_text),
                ),
            )

    def _add_click_target(self, y: int, action: SessionMouseAction) -> None:
        self._click_targets.setdefault(y, []).append(action)

    def _worker_left_text(self, worker: WorkerAgentSnapshot, frame: int) -> str:
        text = f"{worker.name} ({worker.worker_id})"
        if worker.status != WORKER_STATE_WORKING:
            return text
        statement = self._worker_display_statement(worker.statement)
        dots = "." * ((frame // 4) % 4)
        return f"{text} · {statement}{dots}"

    def _worker_right_text(self, worker: WorkerAgentSnapshot) -> str:
        context_text = self._worker_context_text(worker)
        if worker.status == WORKER_STATE_WORKING:
            state_text = self._worker_runtime_duration(worker)
            return self._join_worker_right_text(context_text, state_text)
        if worker.status == WORKER_STATE_READY:
            return self._join_worker_right_text(context_text, "Ready")
        if worker.status == WORKER_STATE_INTERRUPTED:
            return self._join_worker_right_text(context_text, "Interrupted")
        return self._join_worker_right_text(context_text, worker.status.title())

    def _worker_context_text(self, worker: WorkerAgentSnapshot) -> str:
        if worker.context_percent <= 0:
            return ""
        return f"{worker.context_percent}% Context"

    def _join_worker_right_text(self, context_text: str, state_text: str) -> str:
        if context_text and state_text:
            return f"{context_text} · {state_text}"
        return context_text or state_text

    def _worker_display_statement(self, statement: str) -> str:
        text = statement.strip()
        if not text or text.lower() == "thinking":
            return "Thinking"
        return text

    def _worker_runtime_duration(self, worker: WorkerAgentSnapshot) -> str:
        if not worker.started_at:
            return ""
        with suppress(ValueError):
            started = datetime.fromisoformat(worker.started_at.replace("Z", "+00:00"))
            seconds = max(0, int((datetime.now(tz=UTC) - started).total_seconds()))
            return self._format_duration(seconds)
        return ""

    def _command_bottom_panel(
        self,
        suggestions: list[CommandSpec],
        selected: int,
    ) -> BottomPanel | None:
        if not suggestions:
            return None
        return BottomPanel(
            "Commands",
            "Choose a command to run",
            tuple(
                MenuChoice(
                    label=command.command,
                    value=command.command,
                    detail=command.description,
                )
                for command in suggestions
            ),
            selected,
        )

    def _file_reference_bottom_panel(
        self,
        suggestions: list[MenuChoice],
        selected: int,
    ) -> BottomPanel | None:
        if not suggestions:
            return None
        return BottomPanel(
            "Files",
            "Choose a file to reference",
            tuple(suggestions),
            selected,
        )

    def _draw_bottom_panel(
        self,
        stdscr: CursesWindow,
        panel: BottomPanel,
        input_text: str = "",
    ) -> None:
        layout = self._prompt_layout(stdscr, input_text)
        _, width = stdscr.getmaxyx()
        viewport = self._bottom_panel_viewport(stdscr, panel, input_text)
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
        input_text: str = "",
    ) -> int:
        layout = self._prompt_layout(stdscr, input_text)
        return max(6, layout.top_line - self._bottom_panel_height(panel, subtitle_line_count))

    def _bottom_panel_viewport(
        self,
        stdscr: CursesWindow,
        panel: BottomPanel,
        input_text: str = "",
    ) -> BottomPanelViewport:
        layout = self._prompt_layout(stdscr, input_text)
        _, width = stdscr.getmaxyx()
        subtitle_lines = tuple(
            self._panel_text_lines(panel.subtitle, max(1, width - 8), max_lines=4)
        )
        start_y = self._bottom_panel_start(stdscr, panel, len(subtitle_lines), input_text)
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
        input_text: str = "",
    ) -> int | None:
        with suppress(curses.error):
            _, _x, y, _, button_state = curses.getmouse()
            if not self._is_left_click(button_state):
                return None
            viewport = self._bottom_panel_viewport(stdscr, panel, input_text)
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
        file_references: Mapping[str, str] | None = None,
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
            if input_text and file_references:
                line_start = (view_start + offset) * layout.input_width
                self._draw_prompt_text_line(
                    stdscr,
                    y,
                    layout.input_x,
                    line,
                    line_start,
                    layout.input_width,
                    attr,
                    file_references,
                )
            else:
                self._add(stdscr, y, layout.input_x, line, layout.input_width, attr)
        self._add(
            stdscr,
            layout.bottom_line,
            2,
            "─" * max(1, layout.input_width + 2),
            layout.input_width + 2,
            self._attr("light"),
        )
        show_notice = bool(notice and notice != RUNNING_NOTICE)
        hint_text = notice if show_notice else self.agent_mode.prompt_hint
        hint_attr = notice_role if show_notice else self._mode_hint_attr_name()
        self._add(
            stdscr,
            layout.hint_line,
            4,
            hint_text,
            layout.input_width,
            self._attr(hint_attr),
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

    def _draw_prompt_text_line(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        line_start: int,
        width: int,
        base_attr: int,
        file_references: Mapping[str, str],
    ) -> None:
        if not text:
            return
        cursor = 0
        for start, end in self._file_reference_spans(text, line_start, file_references):
            if start > cursor:
                self._add(stdscr, y, x + cursor, text[cursor:start], width - cursor, base_attr)
            self._add(stdscr, y, x + start, text[start:end], width - start, self._attr("accent"))
            cursor = end
        if cursor < len(text):
            self._add(stdscr, y, x + cursor, text[cursor:], width - cursor, base_attr)

    def _file_reference_spans(
        self,
        text: str,
        line_start: int,
        file_references: Mapping[str, str],
    ) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        full_start = line_start
        full_end = line_start + len(text)
        for label in file_references:
            if not label:
                continue
            for match in self._file_reference_label_pattern(label).finditer(text):
                absolute_start = full_start + match.start()
                absolute_end = full_start + match.end()
                if absolute_start < full_end and absolute_end > full_start:
                    spans.append((match.start(), match.end()))
        return self._merge_spans(spans)

    def _merge_spans(self, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not spans:
            return []
        merged: list[tuple[int, int]] = []
        for start, end in sorted(spans):
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                previous_start, previous_end = merged[-1]
                merged[-1] = (previous_start, max(previous_end, end))
        return merged

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

    def _move_prompt_cursor_row(
        self,
        stdscr: CursesWindow,
        input_text: str,
        cursor: int,
        direction: int,
    ) -> int:
        if not input_text:
            return cursor
        layout = self._prompt_layout(stdscr, input_text)
        return self._prompt_cursor_for_row_delta(
            input_text,
            cursor,
            layout.input_width,
            direction,
        )

    def _prompt_cursor_for_row_delta(
        self,
        input_text: str,
        cursor: int,
        width: int,
        direction: int,
    ) -> int:
        if not input_text or direction == 0:
            return cursor
        safe_width = max(1, width)
        line_count = len(self._prompt_lines(input_text, safe_width))
        current_line, current_column = self._prompt_cursor_position(
            input_text,
            cursor,
            safe_width,
        )
        target_line = current_line + direction
        if target_line < 0 or target_line >= line_count:
            return cursor
        line_start = target_line * safe_width
        line_end = min(len(input_text), line_start + safe_width)
        target_column = min(current_column, max(0, line_end - line_start))
        return min(len(input_text), line_start + target_column)

    def _previous_prompt_word(self, input_text: str, cursor: int) -> int:
        index = max(0, min(cursor, len(input_text)))
        while index > 0 and not self._is_prompt_word_char(input_text[index - 1]):
            index -= 1
        while index > 0 and self._is_prompt_word_char(input_text[index - 1]):
            index -= 1
        return index

    def _next_prompt_word(self, input_text: str, cursor: int) -> int:
        length = len(input_text)
        index = max(0, min(cursor, length))
        while index < length and not self._is_prompt_word_char(input_text[index]):
            index += 1
        while index < length and self._is_prompt_word_char(input_text[index]):
            index += 1
        return index

    def _is_prompt_word_char(self, char: str) -> bool:
        return char.isalnum() or char == "_"

    def _session_mouse_action(
        self,
        stdscr: CursesWindow,
        input_text: str,
        command_suggestions: list[CommandSpec],
        command_selected: int = 0,
        file_suggestions: list[MenuChoice] | None = None,
        file_selected: int = 0,
    ) -> SessionMouseAction | None:
        with suppress(curses.error):
            _, x, y, _, button_state = curses.getmouse()
            wheel_up = getattr(curses, "BUTTON4_PRESSED", 0)
            wheel_down = getattr(curses, "BUTTON5_PRESSED", 0)
            if wheel_up and button_state & wheel_up:
                return SessionMouseAction("scroll", 1)
            if wheel_down and button_state & wheel_down:
                return SessionMouseAction("scroll", -1)

            selection_action = self._session_selection_mouse_action(x, y, button_state)
            if selection_action is not None:
                return selection_action

            if self._is_left_click(button_state) and y in self._click_targets:
                for action in reversed(self._click_targets[y]):
                    if not action.x_end or action.x_start <= x < action.x_end:
                        return action

            active_file_suggestions = file_suggestions or []
            if active_file_suggestions and self._is_left_click(button_state):
                panel = self._file_reference_bottom_panel(
                    active_file_suggestions,
                    file_selected,
                )
                if panel is not None:
                    index = self._bottom_panel_mouse_choice(stdscr, panel, input_text)
                    if index is not None:
                        return SessionMouseAction("file_reference", index)

            if command_suggestions and self._is_left_click(button_state):
                panel = self._command_bottom_panel(
                    command_suggestions,
                    selected=command_selected,
                )
                if panel is not None:
                    index = self._bottom_panel_mouse_choice(stdscr, panel, input_text)
                    if index is not None:
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

    def _session_selection_mouse_action(
        self,
        x: int,
        y: int,
        button_state: int,
    ) -> SessionMouseAction | None:
        if self._is_left_press(button_state):
            if y in self._click_targets:
                return None
            point = self._session_text_point_at(x, y)
            if point is None:
                self._clear_session_selection()
                return None
            self._session_selection = SessionTextSelection(point, point)
            self._session_selecting = True
            return SessionMouseAction("selection", 0)

        if self._session_selecting and self._is_mouse_drag(button_state):
            point = self._nearest_session_text_point(x, y)
            if point is not None and self._session_selection is not None:
                self._session_selection = SessionTextSelection(
                    self._session_selection.anchor,
                    point,
                )
            return SessionMouseAction("selection", 0)

        if self._session_selecting and self._is_left_release(button_state):
            point = self._nearest_session_text_point(x, y)
            if point is not None and self._session_selection is not None:
                self._session_selection = SessionTextSelection(
                    self._session_selection.anchor,
                    point,
                )
            self._session_selecting = False
            selected_text = self._selected_session_text()
            if not selected_text:
                self._clear_session_selection()
                return SessionMouseAction("selection", 0)
            if self._copy_to_clipboard(selected_text):
                return SessionMouseAction(
                    "copy_selection",
                    len(selected_text),
                    "Copied selection to clipboard.",
                )
            return SessionMouseAction(
                "copy_selection",
                0,
                "Selected text, but automatic clipboard copy is unavailable.",
            )

        return None

    def _is_left_press(self, button_state: int) -> bool:
        pressed = getattr(curses, "BUTTON1_PRESSED", 0)
        return bool(pressed and button_state & pressed)

    def _is_left_release(self, button_state: int) -> bool:
        released = getattr(curses, "BUTTON1_RELEASED", 0)
        return bool(released and button_state & released)

    def _is_mouse_drag(self, button_state: int) -> bool:
        report = getattr(curses, "REPORT_MOUSE_POSITION", 0)
        return bool(
            (report and button_state & report)
            or self._is_left_press(button_state)
        )

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
        default: str = "",
    ) -> str | None:
        value = default
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

        def question_callback(request: QuestionRequest) -> QuestionResponse:
            response: queue.SimpleQueue[QuestionResponse] = queue.SimpleQueue()
            events.put(
                RuntimeUiEvent(
                    "question",
                    question_request=request,
                    question_response=response,
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
                    question=question_callback,
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
        command_selected = 0
        with suppress(curses.error):
            stdscr.nodelay(True)
        try:
            while worker.is_alive():
                command_suggestions = (
                    self._filtered_running_commands(input_text)
                    if input_text.startswith("/")
                    else []
                )
                if command_suggestions:
                    command_selected = min(command_selected, len(command_suggestions) - 1)
                else:
                    command_selected = 0
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
                        command_suggestions,
                        command_selected,
                    )
                    input_text = key_result.input_text
                    cursor = key_result.cursor
                    prompt_notice = key_result.notice
                    prompt_notice_role = key_result.notice_role
                    abort_key = key_result.abort_key
                    abort_deadline = key_result.abort_deadline
                    command_selected = key_result.command_selected
                    if key_result.clear_anchor:
                        running_anchor = None
                    if key_result.scroll_delta:
                        running_anchor = None
                        running_scroll += key_result.scroll_delta
                    if key_result.command:
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
                            running_anchor,
                            input_text,
                            cursor,
                            prompt_notice,
                            prompt_notice_role,
                            running_scroll,
                        )
                        with suppress(curses.error):
                            stdscr.nodelay(False)
                        try:
                            command_result = self._handle_command(
                                stdscr,
                                key_result.command,
                                session,
                                key_result.submitted,
                            )
                        finally:
                            with suppress(curses.error):
                                stdscr.nodelay(True)
                        prompt_notice = RUNNING_NOTICE
                        prompt_notice_role = "light"
                        if command_result == "exit":
                            self.runtime.abort_current_turn(session.path)
                            self._append_work_summary(
                                session,
                                turn_id,
                                started_at,
                                work_count,
                            )
                            return BackendTurnResult(
                                input_text,
                                cursor,
                                exit_requested=True,
                                anchor_line=running_anchor,
                                scroll=running_scroll,
                            )
                    if key_result.exit_requested:
                        self.runtime.abort_current_turn(session.path)
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
                            running_anchor,
                            input_text,
                            cursor,
                            prompt_notice,
                            prompt_notice_role,
                            running_scroll,
                        )
                        self._append_work_summary(session, turn_id, started_at, work_count)
                        self.home.append_session_event(
                            session.path,
                            "agent_message",
                            {"message": MANUAL_INTERRUPT_MESSAGE},
                        )
                        return BackendTurnResult(
                            input_text,
                            cursor,
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
                    running_anchor,
                    input_text,
                    cursor,
                    prompt_notice,
                    prompt_notice_role,
                    running_scroll,
                )
                viewport = self._draw_session(
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
                    command_suggestions=(
                        self._filtered_running_commands(input_text)
                        if input_text.startswith("/")
                        else []
                    ),
                    command_selected=command_selected,
                    prompt_notice=prompt_notice,
                    prompt_notice_role=prompt_notice_role,
                )
                if viewport is not None:
                    running_scroll = viewport.scroll
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
            running_anchor,
            input_text,
            cursor,
            prompt_notice,
            prompt_notice_role,
            running_scroll,
        )
        response = final_text or result.get("response", "")
        self._append_work_summary(session, turn_id, started_at, work_count)
        if response:
            if not final_text:
                self._fake_type_message(
                    stdscr,
                    session,
                    response,
                    anchor_line=running_anchor,
                    scroll=running_scroll,
                )
            self.home.append_session_event(session.path, "agent_message", {"message": response})
        viewport = self._draw_session(
            stdscr,
            session,
            self._read_message_lines(session.path),
            input_text,
            cursor,
            running_scroll,
            anchor_line=running_anchor,
        )
        if viewport is not None:
            running_scroll = viewport.scroll
        return BackendTurnResult(
            input_text,
            cursor,
            anchor_line=running_anchor,
            scroll=running_scroll,
        )

    def _append_work_summary(
        self,
        session: SessionRecord,
        turn_id: str,
        started_at: float,
        work_count: int,
    ) -> None:
        if not work_count:
            return
        duration = self._format_duration(time.monotonic() - started_at)
        self.home.append_session_event(
            session.path,
            "work_summary",
            {
                "turn_id": turn_id,
                "message": f"Worked for {duration} min",
            },
        )

    def _read_nonblocking_key(self, stdscr: CursesWindow) -> str | int | None:
        try:
            key = stdscr.get_wch()
        except curses.error:
            return None
        return self._complete_escape_key(stdscr, key, restore_blocking=False)

    def _read_prompt_key(self, stdscr: CursesWindow) -> str | int:
        key = stdscr.get_wch()
        return self._complete_escape_key(stdscr, key, restore_blocking=True)

    def _complete_escape_key(
        self,
        stdscr: CursesWindow,
        key: str | int,
        restore_blocking: bool,
    ) -> str | int:
        if not self._is_escape(key):
            return key
        suffix = self._read_escape_suffix(stdscr, restore_blocking)
        return f"\x1b{suffix}" if suffix else key

    def _read_escape_suffix(self, stdscr: CursesWindow, restore_blocking: bool) -> str:
        if not hasattr(stdscr, "nodelay"):
            return ""
        suffix = ""
        with suppress(curses.error):
            stdscr.nodelay(True)
        deadline = time.monotonic() + 0.04
        while time.monotonic() < deadline and len(suffix) < 12:
            try:
                next_key = stdscr.get_wch()
            except curses.error:
                time.sleep(0.001)
                continue
            if isinstance(next_key, str):
                suffix += next_key
                if self._is_complete_escape_suffix(suffix):
                    break
            else:
                break
        if restore_blocking:
            with suppress(curses.error):
                stdscr.nodelay(False)
        return suffix

    def _is_complete_escape_suffix(self, suffix: str) -> bool:
        if suffix in {"b", "B", "f", "F"}:
            return True
        return suffix.endswith(("C", "D", "~", "u", "Z"))

    def _handle_running_key(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        key: str | int,
        input_text: str,
        cursor: int,
        abort_key: str,
        abort_deadline: float,
        command_suggestions: list[CommandSpec] | None = None,
        command_selected: int = 0,
    ) -> RunningKeyResult:
        now = time.monotonic()
        suggestions = command_suggestions or []
        if self._is_shift_tab(key):
            self._cycle_agent_mode()
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
            )

        if self._is_option_left(key):
            return RunningKeyResult(
                input_text,
                self._previous_prompt_word(input_text, cursor),
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
            )
        if self._is_option_right(key):
            return RunningKeyResult(
                input_text,
                self._next_prompt_word(input_text, cursor),
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
            )

        if self._is_ctrl_c(key) and input_text:
            return RunningKeyResult(
                "",
                0,
                RUNNING_NOTICE,
                "light",
                "",
                0.0,
                0,
            )

        if self._is_escape(key) or self._is_ctrl_c(key) or self._is_ctrl_x(key):
            key_label = self._running_interrupt_key_label(key)
            if abort_key == key_label and now <= abort_deadline:
                return RunningKeyResult(
                    input_text,
                    cursor,
                    "Aborting agent",
                    "light",
                    key_label,
                    now,
                    command_selected,
                    exit_requested=True,
                )

            notice = ABORT_AGENT_CONFIRM_NOTICE.format(key_label=key_label)
            return RunningKeyResult(
                input_text,
                cursor,
                notice,
                "light",
                key_label,
                now + 3.0,
                command_selected,
            )

        if self._is_enter(key):
            submitted = input_text.strip()
            if submitted.startswith("/"):
                command = self._submitted_running_command(
                    submitted,
                    suggestions,
                    command_selected,
                )
                if self._is_running_session_command(command):
                    return RunningKeyResult(
                        "",
                        0,
                        RUNNING_NOTICE,
                        "light",
                        abort_key,
                        abort_deadline,
                        0,
                        command,
                        submitted,
                    )
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_COMMAND_BLOCKED_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_MESSAGE_BLOCKED_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
            )

        if key == curses.KEY_UP:
            if suggestions:
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    max(0, command_selected - 1),
                )
            moved_cursor = self._move_prompt_cursor_row(
                stdscr,
                input_text,
                cursor,
                direction=-1,
            )
            if moved_cursor != cursor:
                return RunningKeyResult(
                    input_text,
                    moved_cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
                scroll_delta=1,
                clear_anchor=True,
            )
        if key == curses.KEY_DOWN:
            if suggestions:
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    min(len(suggestions) - 1, command_selected + 1),
                )
            moved_cursor = self._move_prompt_cursor_row(
                stdscr,
                input_text,
                cursor,
                direction=1,
            )
            if moved_cursor != cursor:
                return RunningKeyResult(
                    input_text,
                    moved_cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
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
                command_selected,
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
                command_selected,
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
                command_selected,
            )
        if key == curses.KEY_RIGHT:
            return RunningKeyResult(
                input_text,
                min(len(input_text), cursor + 1),
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
            )
        if key == curses.KEY_HOME:
            return RunningKeyResult(
                input_text,
                0,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
            )
        if key == curses.KEY_END:
            return RunningKeyResult(
                input_text,
                len(input_text),
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
            )
        if key == curses.KEY_MOUSE:
            action = self._session_mouse_action(stdscr, input_text, suggestions)
            if action is not None and action.kind == "cursor":
                return RunningKeyResult(
                    input_text,
                    action.value,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            if action is not None and action.kind == "scroll":
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                    scroll_delta=action.value,
                    clear_anchor=True,
                )
            if action is not None and action.kind == "toggle_work":
                self._toggle_work_turn(action.text)
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            if action is not None and action.kind == "toggle_work_line":
                self._toggle_work_line(action.text)
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            if action is not None and action.kind == "kill_process":
                self.runtime.end_process(action.text, session.path)
                return RunningKeyResult(
                    input_text,
                    cursor,
                    "Process ended",
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            if action is not None and action.kind == "copy_selection":
                return RunningKeyResult(
                    input_text,
                    cursor,
                    action.text,
                    "ok" if action.value else "danger",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            if action is not None and action.kind == "command":
                command = suggestions[action.value].command
                return RunningKeyResult(
                    "",
                    0,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    0,
                    command,
                    command,
                )
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
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
                    command_selected,
                )
            updated = input_text[: cursor - 1] + input_text[cursor:]
            return RunningKeyResult(
                updated,
                cursor - 1,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                0,
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
                0,
            )
        return RunningKeyResult(
            input_text,
            cursor,
            RUNNING_NOTICE,
            "light",
            abort_key,
            abort_deadline,
            command_selected,
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
        scroll: int = 0,
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
                current_final = ""
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
                    scroll=scroll,
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
                approval_request = event.approval_request
                approval_response = event.approval_response
                if approval_request is not None and approval_response is not None:
                    choice = self._request_command_approval(
                        stdscr,
                        session,
                        approval_request,
                    )
                    approval_response.put(choice)
                    approval_message = self._approval_work_message(
                        approval_request,
                        choice,
                    )
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
            elif event.kind == "question":
                question_request = event.question_request
                question_response = event.question_response
                if question_request is not None and question_response is not None:
                    answer = self._request_question(
                        stdscr,
                        session,
                        question_request,
                        scroll=scroll,
                        anchor_line=anchor_line,
                    )
                    question_response.put(answer)
                    self._append_question_context(session, question_request, answer)
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
        scroll: int = 0,
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
                scroll,
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

    def _request_question(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        request: QuestionRequest,
        scroll: int = 0,
        anchor_line: int | None = None,
    ) -> QuestionResponse:
        if request.kind == "text":
            return self._bottom_text_question(
                stdscr,
                session,
                request,
                scroll=scroll,
                anchor_line=anchor_line,
            )
        if request.kind == "confirm":
            return self._bottom_select_question(
                stdscr,
                session,
                self._confirm_question_request(request),
                scroll=scroll,
                anchor_line=anchor_line,
            )
        return self._bottom_select_question(
            stdscr,
            session,
            request,
            scroll=scroll,
            anchor_line=anchor_line,
        )

    def _bottom_select_question(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        request: QuestionRequest,
        scroll: int = 0,
        anchor_line: int | None = None,
    ) -> QuestionResponse:
        if not request.options and request.allow_custom:
            return self._bottom_text_question(
                stdscr,
                session,
                request,
                scroll=scroll,
                anchor_line=anchor_line,
            )

        choices = [
            MenuChoice(option.label, option.value, option.description)
            for option in request.options
        ]
        custom_value = "__custom_question_answer__"
        if request.allow_custom:
            choices.append(
                MenuChoice(
                    "Custom response",
                    custom_value,
                    request.placeholder or "Type a custom answer",
                )
            )
        selected = self._bottom_menu(
            stdscr,
            session,
            "Question",
            request.question,
            tuple(choices),
            restore_nodelay=True,
            scroll=scroll,
            anchor_line=anchor_line,
        )
        if selected is None:
            return QuestionResponse(
                answered=False,
                kind=request.kind,
                cancelled=True,
            )
        if selected == custom_value:
            return self._bottom_text_question(
                stdscr,
                session,
                request,
                scroll=scroll,
                anchor_line=anchor_line,
            )

        option_by_value = {option.value: option for option in request.options}
        option = option_by_value.get(selected)
        return QuestionResponse(
            answered=True,
            answer=selected,
            selected_label=option.label if option is not None else selected,
            kind=request.kind,
        )

    def _bottom_text_question(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        request: QuestionRequest,
        scroll: int = 0,
        anchor_line: int | None = None,
    ) -> QuestionResponse:
        value = request.default
        cursor = len(value)
        current_scroll = scroll
        saved_placeholder = self._prompt_placeholder
        self._prompt_placeholder = request.placeholder or "Type your response"
        with suppress(curses.error):
            stdscr.nodelay(False)
        try:
            while True:
                panel = BottomPanel("Question", request.question, (), 0)
                viewport = self._draw_session(
                    stdscr,
                    session,
                    self._read_message_lines(session.path),
                    value,
                    cursor,
                    current_scroll,
                    bottom_panel=panel,
                    anchor_line=anchor_line,
                    prompt_notice="Enter Submit · Esc Cancel",
                    prompt_notice_role="light",
                )
                if viewport is not None:
                    current_scroll = viewport.scroll
                key = stdscr.get_wch()
                if self._is_escape(key) or self._is_ctrl_c(key):
                    return QuestionResponse(
                        answered=False,
                        kind=request.kind,
                        cancelled=True,
                    )
                if self._is_enter(key):
                    return QuestionResponse(
                        answered=True,
                        answer=value.strip(),
                        selected_label="",
                        kind=request.kind,
                    )
                if key == curses.KEY_LEFT:
                    cursor = max(0, cursor - 1)
                elif key == curses.KEY_RIGHT:
                    cursor = min(len(value), cursor + 1)
                elif key == curses.KEY_HOME:
                    cursor = 0
                elif key == curses.KEY_END:
                    cursor = len(value)
                elif self._is_backspace(key):
                    if cursor > 0:
                        value = value[: cursor - 1] + value[cursor:]
                        cursor -= 1
                elif isinstance(key, str) and key.isprintable():
                    value = value[:cursor] + key + value[cursor:]
                    cursor += len(key)
        finally:
            self._prompt_placeholder = saved_placeholder
            with suppress(curses.error):
                stdscr.nodelay(True)

    def _confirm_question_request(self, request: QuestionRequest) -> QuestionRequest:
        if request.options:
            return request
        return QuestionRequest(
            question=request.question,
            kind=request.kind,
            options=(
                QuestionOption("Yes", "yes", "Confirm and continue"),
                QuestionOption("No", "no", "Cancel or choose another path"),
            ),
            placeholder=request.placeholder,
            default=request.default,
            allow_custom=False,
        )

    def _append_question_context(
        self,
        session: SessionRecord,
        request: QuestionRequest,
        answer: QuestionResponse,
    ) -> None:
        if answer.cancelled:
            message = f"User cancelled question: {request.question}"
        else:
            message = f"Question: {request.question}\nAnswer: {answer.answer}"
        self.home.append_session_event(
            session.path,
            "system_message",
            {
                "message": message,
                "role": "question",
            },
        )

    def _read_message_lines(self, session_path: Path) -> list[MessageLine]:
        lines: list[MessageLine] = []
        pending_turn: dict[str, list[MessageLine]] = {}
        for event_index, event in enumerate(self.home.read_session_events(session_path)):
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            event_type = str(
                payload.get("type") if event.get("type") == "event_msg" else event.get("type")
            )
            message = str(payload.get("message", "")).strip()
            if event_type in {"user_message", "skill_invocation"} and message:
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
                if role in {"worker", "question"}:
                    continue
                turn_id = str(payload.get("turn_id", ""))
                expansion_key = self._session_work_line_key(role, turn_id, event_index)
                if turn_id:
                    pending_turn.setdefault(turn_id, []).append(
                        MessageLine(role, message, turn_id, expansion_key)
                    )
                else:
                    lines.append(MessageLine(role, message, expansion_key=expansion_key))
            elif event_type == "work_message" and message:
                turn_id = str(payload.get("turn_id", ""))
                role = str(payload.get("role", "tool"))
                expansion_key = self._session_work_line_key(role, turn_id, event_index)
                if turn_id:
                    pending_turn.setdefault(turn_id, []).append(
                        MessageLine(role, message, turn_id, expansion_key)
                    )
                else:
                    lines.append(MessageLine(role, message, expansion_key=expansion_key))
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

    def _session_work_line_key(self, role: str, turn_id: str, event_index: int) -> str:
        if not self._is_expandable_work_role(role):
            return ""
        namespace = turn_id or "session"
        return f"{namespace}:{event_index}"

    def _render_messages(self, messages: list[MessageLine], width: int) -> list[MessageLine]:
        rendered: list[MessageLine] = []
        previous_kind: str | None = None
        for message in messages:
            kind = self._message_kind(message.role)
            if rendered and previous_kind is not None and kind != previous_kind:
                rendered.append(MessageLine("meta", ""))
            if self._is_expandable_work_role(message.role):
                rendered.extend(self._render_work_message(message, width))
            else:
                for line in markdown_to_terminal_rendered_lines(
                    message.text,
                    width=max(20, width),
                ):
                    rendered.append(
                        MessageLine(
                            self._terminal_line_role(message.role, line.style),
                            line.text,
                            message.meta,
                        )
                    )
            previous_kind = kind
        return rendered

    def _terminal_line_role(self, fallback_role: str, style: str) -> str:
        if style in {"table_border", "table_header", "table_row"}:
            return style
        return fallback_role

    def _render_work_message(self, message: MessageLine, width: int) -> list[MessageLine]:
        safe_width = max(20, width)
        expansion_key = message.expansion_key or self._fallback_work_line_key(message)
        if expansion_key in self._expanded_work_lines:
            return self._expanded_work_box_lines(message, safe_width, expansion_key)

        display_text = self._single_line_work_text(message.text)
        approval_label = self._approval_work_label(display_text)
        if approval_label is not None:
            return [
                MessageLine(
                    message.role,
                    self._click_to_expand_text(approval_label, safe_width),
                    message.meta,
                    expansion_key,
                )
            ]
        if self._work_text_needs_expansion(message.text, safe_width):
            display_text = self._collapsed_work_text(display_text, safe_width)
            return [MessageLine(message.role, display_text, message.meta, expansion_key)]
        return [MessageLine(message.role, display_text, message.meta)]

    def _is_expandable_work_role(self, role: str) -> bool:
        return role in {"tool", "approved", "forbidden"}

    def _single_line_work_text(self, text: str) -> str:
        return " ".join(text.replace("\r", " ").replace("\n", " ").split())

    def _work_text_needs_expansion(self, text: str, width: int) -> bool:
        return "\n" in text or "\r" in text or len(self._single_line_work_text(text)) > width

    def _approval_work_label(self, text: str) -> str | None:
        label, separator, _command = text.partition(":")
        if not separator:
            return None
        if label in {"Approved command", "Rejected command"}:
            return label
        if label.startswith(("Always approved ", "Always rejected ")):
            return label
        return None

    def _collapsed_work_text(self, text: str, width: int) -> str:
        safe_width = max(1, width)
        if len(text) <= safe_width:
            return text
        return self._click_to_expand_text(text, safe_width)

    def _click_to_expand_text(self, text: str, width: int) -> str:
        safe_width = max(1, width)
        marker = "... click to expand"
        suffix = f" {marker}"
        if safe_width <= len(marker):
            return marker[:safe_width]
        prefix_width = safe_width - len(suffix)
        return f"{text[:prefix_width].rstrip()}{suffix}"

    def _expanded_work_box_lines(
        self,
        message: MessageLine,
        width: int,
        expansion_key: str,
    ) -> list[MessageLine]:
        safe_width = max(20, width)
        inner_width = max(1, safe_width - 4)
        border = "─" * max(1, safe_width - 2)
        lines = [MessageLine("work_box", f"╭{border}╮", message.meta, expansion_key)]
        for content_line in self._work_box_content_lines(message.text, inner_width):
            content = content_line[:inner_width].ljust(inner_width)
            lines.append(MessageLine("work_box", f"│ {content} │", message.meta, expansion_key))
        lines.append(MessageLine("work_box", f"╰{border}╯", message.meta, expansion_key))
        return lines

    def _work_box_content_lines(self, text: str, width: int) -> list[str]:
        safe_width = max(1, width)
        lines: list[str] = []
        for raw_line in text.splitlines() or [""]:
            line = raw_line.replace("\t", "    ").rstrip()
            if not line:
                lines.append("")
                continue
            lines.extend(
                textwrap.wrap(
                    line,
                    width=safe_width,
                    replace_whitespace=False,
                    drop_whitespace=False,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
                or [""]
            )
        return lines

    def _fallback_work_line_key(self, message: MessageLine) -> str:
        digest = hashlib.sha1(message.text.encode("utf-8", errors="replace")).hexdigest()
        return f"{message.role}:{message.meta}:{digest}"

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

    def _toggle_work_line(self, expansion_key: str) -> None:
        if expansion_key in self._expanded_work_lines:
            self._expanded_work_lines.remove(expansion_key)
        elif expansion_key:
            self._expanded_work_lines.add(expansion_key)

    def _cycle_agent_mode(self) -> None:
        self.agent_mode = self.agent_mode.next()
        self.runtime.set_mode(self.agent_mode)
        config = self.home.load_config()
        config["agent_mode"] = self.agent_mode.value
        self.home.save_config(config)

    def _mode_hint_attr_name(self) -> str:
        if self.agent_mode == AgentMode.FULL_CONTROL:
            return "danger"
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

    def _session_location_rows(self, session: SessionRecord) -> tuple[InfoRow, ...]:
        return (
            InfoRow("Path", session.cwd or str(self.cwd)),
            InfoRow("Workspace root", str(self.workspace_root)),
        )

    def _approved_command_rows(self, session: SessionRecord) -> tuple[InfoRow, ...]:
        subjects = self._session_command_subjects(self.session_allowed_commands)
        rows = [InfoRow("Always approved", subject) for subject in subjects]
        rows.extend(
            InfoRow("Approved", command)
            for command in self._approved_commands_from_session(session)
        )
        if not rows:
            return (InfoRow("Session", "No commands approved yet"),)
        return tuple(rows)

    def _approved_commands_from_session(self, session: SessionRecord) -> tuple[str, ...]:
        commands: list[str] = []
        seen: set[str] = set()
        for event in self.home.read_session_events(session.path):
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            event_type = str(
                payload.get("type") if event.get("type") == "event_msg" else event.get("type")
            )
            if event_type != "work_message":
                continue
            command = self._approved_command_from_message(str(payload.get("message", "")))
            if not command or command in seen:
                continue
            seen.add(command)
            commands.append(command)
        return tuple(commands)

    def _approved_command_from_message(self, message: str) -> str:
        text = message.strip()
        if text.startswith("Approved command:"):
            return text.partition(":")[2].strip()
        if text.startswith("Always approved "):
            return text.partition(":")[2].strip()
        return ""

    def _session_command_subjects(self, keys: set[str]) -> list[str]:
        return sorted({self._session_command_subject(key) for key in keys})

    def _session_command_subject(self, key: str) -> str:
        if key.startswith("cmd:"):
            subject = key.removeprefix("cmd:").strip()
            return subject or "this command"
        return key or "this command"

    def _all_skills(self) -> tuple[Skill, ...]:
        skills: list[Skill] = []
        seen: set[str] = set()
        for skill in (*load_builtin_skills(), *self._user_skills()):
            if skill.command in seen:
                continue
            seen.add(skill.command)
            skills.append(skill)
        return tuple(skills)

    def _user_skills(self) -> tuple[Skill, ...]:
        return load_user_skills(self.home.skills_dir)

    def _starter_skills(self) -> tuple[Skill, ...]:
        skills_by_command = {skill.command: skill for skill in self._all_skills()}
        return tuple(
            skills_by_command[command]
            for command in STARTER_SKILL_COMMANDS
            if command in skills_by_command
        )

    def _skill_for_command(self, command: str) -> Skill | None:
        command_name = command.removeprefix("/")
        return next(
            (skill for skill in self._all_skills() if skill.command == command_name),
            None,
        )

    def _user_skill_by_command(self, command: str) -> Skill | None:
        return next((skill for skill in self._user_skills() if skill.command == command), None)

    def _command_exists(self, command: str, exclude_command: str | None = None) -> bool:
        slash_command = f"/{command}"
        excluded = f"/{exclude_command}" if exclude_command else ""
        return any(
            spec.command == slash_command and spec.command != excluded
            for spec in self._command_specs()
        )

    def _command_specs(self) -> tuple[CommandSpec, ...]:
        skill_specs = tuple(
            CommandSpec(skill.slash_command, f"{skill.title} · {skill.description}")
            for skill in self._all_skills()
        )
        return (*COMMANDS, *skill_specs)

    def _invoke_skill(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        skill: Skill,
        submitted: str,
    ) -> str | None:
        arguments = self._skill_arguments(skill, submitted)
        display_message = f"/{skill.command} {arguments}".strip()
        self.home.append_session_event(
            session.path,
            "skill_invocation",
            {
                "message": display_message,
                "command": skill.command,
                "title": skill.title,
                "description": skill.description,
                "prompt": skill_invocation_prompt(skill, arguments),
            },
        )
        self._maybe_start_session_rename(session)
        anchor_line = self._latest_user_anchor_line(stdscr, session)
        self._animate_message_anchor(stdscr, session, anchor_line)
        turn_result = self._run_backend_turn(stdscr, session, anchor_line=anchor_line)
        return "exit" if turn_result.exit_requested else None

    def _skill_arguments(self, skill: Skill, submitted: str) -> str:
        parts = submitted.strip().split(maxsplit=1)
        if not parts:
            return ""
        if parts[0] != skill.slash_command:
            return ""
        return parts[1] if len(parts) > 1 else ""

    def _filtered_commands(self, input_text: str) -> list[CommandSpec]:
        return self._filtered_command_specs(input_text, self._command_specs())

    def _filtered_running_commands(self, input_text: str) -> list[CommandSpec]:
        return self._filtered_command_specs(input_text, self._running_command_specs())

    def _filtered_command_specs(
        self,
        input_text: str,
        command_specs: tuple[CommandSpec, ...],
    ) -> list[CommandSpec]:
        query = input_text.removeprefix("/").strip().lower()
        if not query:
            return list(command_specs)

        prefix_matches: list[CommandSpec] = []
        contains_matches: list[CommandSpec] = []
        for command in command_specs:
            command_name = command.command.removeprefix("/").lower()
            searchable = f"{command_name} {command.description}".lower()
            if command_name.startswith(query):
                prefix_matches.append(command)
            elif query in searchable:
                contains_matches.append(command)
        if prefix_matches:
            return prefix_matches[:5]
        return contains_matches[:5]

    def _active_file_reference_token(
        self,
        input_text: str,
        cursor: int,
    ) -> tuple[int, int, str] | None:
        bounded_cursor = max(0, min(cursor, len(input_text)))
        token_start = bounded_cursor
        while token_start > 0 and not input_text[token_start - 1].isspace():
            token_start -= 1
        token_end = bounded_cursor
        while token_end < len(input_text) and not input_text[token_end].isspace():
            token_end += 1
        token = input_text[token_start:bounded_cursor]
        if not token.startswith("@"):
            return None
        return (token_start, token_end, token.removeprefix("@"))

    def _filtered_file_references(self, query: str) -> list[MenuChoice]:
        normalized_query = query.strip().lower()
        matches: list[tuple[int, int, str, MenuChoice]] = []
        for path in self._workspace_reference_paths():
            relative = self._relative_workspace_path(path)
            if path.is_dir():
                relative = f"{relative}/"
            name = self._reference_label(path)
            name_search = name.lower()
            relative_search = relative.lower()
            if normalized_query:
                if name_search.startswith(normalized_query):
                    rank = 0
                elif relative_search.startswith(normalized_query):
                    rank = 1
                elif normalized_query in name_search:
                    rank = 2
                elif normalized_query in relative_search:
                    rank = 3
                else:
                    continue
            else:
                rank = 0

            matches.append(
                (
                    rank,
                    len(relative),
                    relative_search,
                    MenuChoice(name, relative, self._file_reference_detail(relative)),
                )
            )
        matches.sort(key=lambda match: (match[0], match[1], match[2]))
        return [match[3] for match in matches[:FILE_REFERENCE_LIMIT]]

    def _workspace_reference_paths(self) -> tuple[Path, ...]:
        now = time.monotonic()
        if (
            self._file_reference_cache
            and now - self._file_reference_cache_at < FILE_REFERENCE_CACHE_SECONDS
        ):
            return self._file_reference_cache

        paths: list[Path] = []
        for root, dirnames, filenames in os.walk(self.workspace_root):
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if not self._ignore_file_reference_dir(dirname)
            ]
            for dirname in dirnames:
                paths.append(Path(root) / dirname)
                if len(paths) >= FILE_REFERENCE_SCAN_LIMIT:
                    break
            if len(paths) >= FILE_REFERENCE_SCAN_LIMIT:
                break
            for filename in filenames:
                if self._ignore_file_reference_file(filename):
                    continue
                path = Path(root) / filename
                if not path.is_file():
                    continue
                paths.append(path)
                if len(paths) >= FILE_REFERENCE_SCAN_LIMIT:
                    break
            if len(paths) >= FILE_REFERENCE_SCAN_LIMIT:
                break

        self._file_reference_cache = tuple(
            sorted(paths, key=lambda path: self._relative_workspace_path(path).lower())
        )
        self._file_reference_cache_at = now
        return self._file_reference_cache

    def _reference_label(self, path: Path) -> str:
        suffix = "/" if path.is_dir() else ""
        return f"{path.name}{suffix}"

    def _ignore_file_reference_dir(self, dirname: str) -> bool:
        return dirname in IGNORED_FILE_REFERENCE_DIRS or dirname.startswith(".")

    def _ignore_file_reference_file(self, filename: str) -> bool:
        return filename.endswith((".pyc", ".pyo", ".DS_Store"))

    def _relative_workspace_path(self, path: Path) -> str:
        with suppress(ValueError):
            return path.resolve().relative_to(self.workspace_root).as_posix()
        return path.as_posix()

    def _file_reference_detail(self, relative_path: str) -> str:
        parent = Path(relative_path).parent.as_posix()
        return parent if parent != "." else relative_path

    def _insert_file_reference(
        self,
        input_text: str,
        cursor: int,
        token: tuple[int, int, str],
        choice: MenuChoice,
        file_references: dict[str, str],
    ) -> tuple[str, int]:
        del cursor
        start, end, _query = token
        suffix = "" if end < len(input_text) and input_text[end].isspace() else " "
        replacement = f"{choice.label}{suffix}"
        updated = input_text[:start] + replacement + input_text[end:]
        file_references[choice.label] = choice.value
        return updated, start + len(replacement)

    def _backend_message_for_prompt(
        self,
        prompt: str,
        file_references: Mapping[str, str],
    ) -> str:
        expanded = prompt
        for label, path in sorted(
            file_references.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if not label or not path:
                continue
            expanded = self._file_reference_label_pattern(label).sub(
                f"{label} [{path}]",
                expanded,
            )
        return expanded

    def _file_reference_label_pattern(self, label: str) -> re.Pattern[str]:
        escaped = re.escape(label)
        return re.compile(rf"(?<![\w./-]){escaped}(?=$|\s|[,.!?;:)\]}}])")

    def _submitted_command(
        self,
        submitted: str,
        suggestions: list[CommandSpec],
        selected: int,
    ) -> str:
        exact_command = submitted.split(maxsplit=1)[0]
        if any(command.command == exact_command for command in self._command_specs()):
            return exact_command
        if suggestions:
            return suggestions[min(selected, len(suggestions) - 1)].command
        return submitted

    def _submitted_running_command(
        self,
        submitted: str,
        suggestions: list[CommandSpec],
        selected: int,
    ) -> str:
        exact_command = submitted.split(maxsplit=1)[0]
        if self._is_running_session_command(exact_command):
            return exact_command
        if suggestions:
            return suggestions[min(selected, len(suggestions) - 1)].command
        return exact_command

    def _running_command_specs(self) -> tuple[CommandSpec, ...]:
        return tuple(
            command
            for command in COMMANDS
            if self._is_running_session_command(command.command)
        )

    def _is_running_session_command(self, command: str) -> bool:
        return command in RUNNING_SESSION_COMMANDS

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

    def _is_ctrl_x(self, key: str | int) -> bool:
        return key == "\x18" or key == 24

    def _is_ctrl_d(self, key: str | int) -> bool:
        return key == "\x04" or key == 4

    def _is_ctrl_e(self, key: str | int) -> bool:
        return key == "\x05" or key == 5

    def _is_ctrl_s(self, key: str | int) -> bool:
        return key == "\x13" or key == 19

    def _running_interrupt_key_label(self, key: str | int) -> str:
        if self._is_ctrl_c(key):
            return "Ctrl+C"
        if self._is_ctrl_x(key):
            return "Ctrl+X"
        return "Esc"

    def _is_shift_tab(self, key: str | int) -> bool:
        return key == "\x1b[Z" or key == getattr(curses, "KEY_BTAB", 353)

    def _is_option_left(self, key: str | int) -> bool:
        return key in {
            "\x1bb",
            "\x1bB",
            "\x1b[1;3D",
            "\x1b[1;9D",
            "\x1b[3D",
        }

    def _is_option_right(self, key: str | int) -> bool:
        return key in {
            "\x1bf",
            "\x1bF",
            "\x1b[1;3C",
            "\x1b[1;9C",
            "\x1b[3C",
        }

    def _is_shift_enter(self, key: str | int) -> bool:
        return key in {"\x1b[13;2u", "\x1b[13;2~"}

    def _is_enter(self, key: str | int) -> bool:
        return key in {"\n", "\r", curses.KEY_ENTER}

    def _is_backspace(self, key: str | int) -> bool:
        return key in {"\b", "\x7f", curses.KEY_BACKSPACE, 127, 8}
