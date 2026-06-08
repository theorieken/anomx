"""Full-screen terminal UI for the Anomx agent."""

from __future__ import annotations

import curses
import hashlib
import math
import os
import queue
import random
import re
import shlex
import shutil
import subprocess
import textwrap
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse
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
    IMAGE_FILE_EXTENSIONS,
    AgentRuntime,
    QuestionOption,
    QuestionRequest,
    QuestionResponse,
    RuntimeCallbacks,
    StatusCallback,
    backend_supports_image_input,
    context_usage_percent,
    image_mime_type,
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
    AsyncProcessSnapshot,
    PlanStep,
    latest_plan_steps,
    running_process_snapshots,
)
from anomx.agent.store import (
    AI_PROVIDERS,
    AnomxHome,
    ProjectRecord,
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
    PROJECT = "Project"
    CONTINUE_SESSION = "Continue Session"
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
    highlight: str = ""


@dataclass(frozen=True)
class MessageLine:
    """Rendered session transcript line."""

    role: str
    text: str
    meta: str = ""
    expansion_key: str = dataclass_field(default="", compare=False)
    detail_title: str = dataclass_field(default="", compare=False)
    detail_body: str = dataclass_field(default="", compare=False)


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
    command: str = ""
    approval_request: CommandApprovalRequest | None = None
    approval_response: queue.SimpleQueue[ApprovalChoice] | None = None
    question_request: QuestionRequest | None = None
    question_response: queue.SimpleQueue[QuestionResponse] | None = None


@dataclass
class ActiveSessionTurn:
    """A model turn that may keep running outside the focused session view."""

    session: SessionRecord
    runtime: AgentRuntime
    events: queue.SimpleQueue[RuntimeUiEvent]
    result: dict[str, str]
    turn_id: str
    started_at: float
    worker: threading.Thread
    mode: AgentMode
    working_text: str | None = "Thinking"
    working_deadline: float | None = None
    final_text: str = ""
    work_count: int = 0
    work_summary_appended: bool = False
    streamed_final_displayed: bool = False
    completed: bool = False
    pending_events: list[RuntimeUiEvent] = dataclass_field(default_factory=list)


@dataclass(frozen=True)
class StartupPreparation:
    """Initial session state prepared while the startup animation is visible."""

    previous_session: SessionRecord | None = None
    continuation_statement: str = ""
    project: ProjectRecord | None = None


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
    back_requested: bool = False


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
    back_requested: bool = False


@dataclass(frozen=True)
class ActivityDetailEntry:
    """A clickable line inside an expanded activity item."""

    key: str
    text: str
    detail_body: str = ""


@dataclass(frozen=True)
class ActivityDetailRow:
    """A rendered row inside an expanded activity item."""

    text: str
    role: str = "activity_detail"
    entry_key: str = ""


@dataclass(frozen=True)
class ActivityItem:
    """A bottom activity-panel row for workers and async processes."""

    key: str
    title: str
    right_text: str
    details: tuple[ActivityDetailEntry, ...]
    active: bool = False
    kill_process_id: str = ""
    marker: str = "•"


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
    "Agent is working · /commands available · messages blocked · Esc back · Ctrl+C/Ctrl+X interrupt"
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
ACTIVITY_DETAIL_MAX_LINES = 10
RAW_MOUSE_RE = re.compile(r"^\x1b\[<(?P<button>\d+);(?P<x>\d+);(?P<y>\d+)(?P<suffix>[mM])$")
RAW_MOUSE_SUFFIX_RE = re.compile(r"^\[<\d+;\d+;\d+[mM]$")
FILE_REFERENCE_LIMIT = 8
FILE_REFERENCE_SCAN_LIMIT = 500
FILE_REFERENCE_CACHE_SECONDS = 2.0
FILE_REFERENCE_FIRST_LEVEL_LIMIT = 200
IMAGE_DROP_EXTENSION_PATTERN = "|".join(re.escape(ext) for ext in IMAGE_FILE_EXTENSIONS)
IMAGE_DROP_CANDIDATE_PATTERN = re.compile(
    rf"(?P<path>(?:file://)?(?:~|/)[^\r\n]*?(?:{IMAGE_DROP_EXTENSION_PATTERN}))"
    r"(?=$|[\s\"'`),;!?])",
    re.IGNORECASE,
)
STARTUP_PHASE_SECONDS = 2.0
STARTUP_LOADING_SECONDS = STARTUP_PHASE_SECONDS * 3
STARTUP_REVEAL_SECONDS = 1.2
STARTUP_OVERLAY_DELAY_SECONDS = 0.35
STARTUP_LINE_REVEAL_SECONDS = 2.15
STARTUP_BRAND_REVEAL_SECONDS = 1.9
STARTUP_WIPE_SECONDS = 0.7
STARTUP_FRAME_SECONDS = 0.065
STARTUP_COLUMN_WIDTH = 2
STARTUP_MATRIX_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
START_HINT_REVEAL_SECONDS = 1.0
START_HINT_WIPE_SECONDS = 0.45
PLAN_STEP_REVEAL_SECONDS = 1.0
BRAND_NAME = "Anomx"
BRAND_DOT = "."
AGENT_DESCRIPTOR = "Data Analysis and Anomaly Detection Agent"
STARTUP_ANOMX_GLYPH = (
    "  #    #   #   ###   #   #  #   #",
    " # #   ##  #  #   #  ## ##  #   #",
    "#   #  # # #  #   #  # # #   # # ",
    "#####  #  ##  #   #  #   #    #  ",
    "#   #  #   #  #   #  #   #   # # ",
    "#   #  #   #  #   #  #   #  #   #",
    "#   #  #   #   ###   #   #  #   #",
)
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
        self.project_path = self.cwd
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
        self._expanded_plan_sessions: set[Path] = set()
        self._expanded_activity_items: set[str] = set()
        self._expanded_activity_entries: set[str] = set()
        self._activity_detail_scrolls: dict[str, int] = {}
        self._session_event_cache: dict[Path, tuple[int, int, list[dict[str, Any]]]] = {}
        self._message_line_cache: dict[
            Path,
            tuple[int, int, tuple[str, ...], list[MessageLine]],
        ] = {}
        self._rendered_message_cache: dict[
            Path,
            tuple[
                int,
                int,
                tuple[str, ...],
                tuple[str, ...],
                int,
                str,
                list[MessageLine],
            ],
        ] = {}
        self._context_status_cache: dict[Path, tuple[int, int, str, str]] = {}
        self._config_cache: tuple[int, int, dict[str, Any]] | None = None
        self._start_hint_reveal_started_at: float | None = None
        self._click_targets: dict[int, list[SessionMouseAction]] = {}
        self._session_text_rows: dict[int, SessionTextRow] = {}
        self._session_selection: SessionTextSelection | None = None
        self._session_selecting = False
        self._title_events: queue.SimpleQueue[tuple[str, str]] = queue.SimpleQueue()
        self._title_jobs: set[str] = set()
        self._file_reference_cache_at = 0.0
        self._file_reference_cache: tuple[Path, ...] = ()
        self._file_reference_full_cache: tuple[Path, ...] = ()
        self._file_reference_full_cache_at = 0.0
        self._prepare_startup_during_loading = False
        self._startup_preparation: StartupPreparation | None = None
        self._active_session_turns: dict[Path, ActiveSessionTurn] = {}

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
        self._prepare_startup_during_loading = True
        try:
            self._run_startup_loading(stdscr)
        finally:
            self._prepare_startup_during_loading = False

        while True:
            config = self.home.load_config()
            if not self._onboarding_complete(config):
                if not self._run_onboarding(stdscr):
                    return 1
                config = self.home.load_config()

            should_check_access = not self.home.is_repo_trusted(self.workspace_root)
            if should_check_access and not self._run_access_check(stdscr):
                return 1

            self.state = AgentState.PROJECT
            project = self._startup_project(stdscr, self._startup_preparation)
            self._cleanup_stale_project_sessions(project)
            try:
                return self._run_project(stdscr, project)
            finally:
                self._shutdown_active_session_turns()
                self.runtime.shutdown()

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
            brand_dot_pair = 3
            warning_badge_pair = 3
            if getattr(curses, "COLORS", 0) > 208 and getattr(curses, "COLOR_PAIRS", 0) > 8:
                with suppress(curses.error):
                    curses.init_pair(8, 208, curses.COLOR_BLACK)
                    brand_dot_pair = 8
            if getattr(curses, "COLOR_PAIRS", 0) > 9:
                with suppress(curses.error):
                    curses.init_pair(9, curses.COLOR_BLACK, curses.COLOR_YELLOW)
                    warning_badge_pair = 9
            self._colors = {
                "accent": curses.color_pair(1) | curses.A_BOLD,
                "selected": curses.color_pair(6) | curses.A_BOLD,
                "cursor": curses.color_pair(6) | curses.A_REVERSE,
                "background": curses.color_pair(7),
                "muted": curses.color_pair(7) | curses.A_DIM,
                "light": curses.color_pair(6) | curses.A_DIM,
                "warning": curses.color_pair(3) | curses.A_BOLD,
                "warning_badge": curses.color_pair(warning_badge_pair) | curses.A_BOLD,
                "ok": curses.color_pair(4) | curses.A_BOLD,
                "danger": curses.color_pair(5) | curses.A_BOLD,
                "bold": curses.color_pair(6) | curses.A_BOLD,
                "work_box": curses.color_pair(6) | curses.A_REVERSE,
                "table_header": curses.color_pair(6) | curses.A_REVERSE | curses.A_BOLD,
                "table_border": curses.color_pair(6) | curses.A_DIM,
                "matrix_dim": curses.color_pair(6) | curses.A_DIM,
                "matrix_function": curses.color_pair(1) | curses.A_BOLD,
                "matrix_blue": curses.color_pair(1) | curses.A_BOLD,
                "matrix_wave": curses.color_pair(6) | curses.A_BOLD,
                "matrix_hot": curses.color_pair(6) | curses.A_BOLD,
                "matrix_brand": curses.color_pair(1) | curses.A_BOLD,
                "brand_dot": curses.color_pair(brand_dot_pair) | curses.A_BOLD,
            }
        else:
            self._colors = {
                "accent": curses.A_BOLD,
                "selected": curses.A_REVERSE,
                "cursor": curses.A_REVERSE,
                "background": curses.A_NORMAL,
                "light": curses.A_DIM,
                "muted": curses.A_DIM,
                "warning": curses.A_BOLD,
                "warning_badge": curses.A_REVERSE | curses.A_BOLD,
                "ok": curses.A_BOLD,
                "danger": curses.A_BOLD,
                "bold": curses.A_BOLD,
                "work_box": curses.A_REVERSE,
                "table_header": curses.A_REVERSE | curses.A_BOLD,
                "table_border": curses.A_DIM,
                "matrix_dim": curses.A_DIM,
                "matrix_function": curses.A_BOLD,
                "matrix_blue": curses.A_NORMAL,
                "matrix_wave": curses.A_BOLD,
                "matrix_hot": curses.A_BOLD,
                "matrix_brand": curses.A_BOLD,
                "brand_dot": curses.A_BOLD,
            }
        with suppress(curses.error):
            stdscr.bkgd(" ", self._attr("background"))

    def _run_startup_loading(self, stdscr: CursesWindow) -> bool:
        """Show the startup matrix while the optional platform link warms up."""

        connection = self.home.platform_connection()
        results: queue.SimpleQueue[bool] = queue.SimpleQueue()
        worker: threading.Thread | None = None
        preparation_results: queue.SimpleQueue[StartupPreparation | None] = queue.SimpleQueue()
        preparation_worker: threading.Thread | None = None
        preparation_ready = not self._prepare_startup_during_loading
        self._startup_preparation = None

        frame = 0
        connected = False
        with suppress(curses.error):
            stdscr.nodelay(True)
        try:
            started_at = time.monotonic()
            now = started_at
            while (now - started_at) < STARTUP_LOADING_SECONDS or not preparation_ready:
                elapsed = now - started_at
                activity = self._startup_loading_activity(elapsed)
                if activity == "Connecting" and worker is None and connection is not None:

                    def run_heartbeat() -> None:
                        try:
                            results.put(heartbeat_platform_connection(self.home))
                        except Exception:
                            results.put(False)

                    worker = threading.Thread(target=run_heartbeat, daemon=True)
                    worker.start()
                if (
                    activity == "Screening"
                    and preparation_worker is None
                    and self._prepare_startup_during_loading
                ):

                    def run_startup_preparation() -> None:
                        try:
                            preparation_results.put(self._prepare_startup_state())
                        except Exception:
                            preparation_results.put(None)

                    preparation_worker = threading.Thread(
                        target=run_startup_preparation,
                        daemon=True,
                    )
                    preparation_worker.start()
                with suppress(queue.Empty):
                    connected = results.get_nowait() or connected
                if not preparation_ready:
                    with suppress(queue.Empty):
                        self._startup_preparation = preparation_results.get_nowait()
                        preparation_ready = True
                self._draw_startup_loading(
                    stdscr,
                    frame,
                    elapsed=now - started_at,
                    activity_text=activity,
                )
                frame += 1
                with suppress(curses.error):
                    stdscr.get_wch()
                time.sleep(STARTUP_FRAME_SECONDS)
                now = time.monotonic()
        finally:
            with suppress(curses.error):
                stdscr.nodelay(False)
        with suppress(queue.Empty):
            connected = results.get_nowait() or connected
        if not preparation_ready:
            with suppress(queue.Empty):
                self._startup_preparation = preparation_results.get_nowait()
                preparation_ready = True
        frame = self._run_startup_wipe(stdscr, frame)
        if worker is not None and connected:
            worker.join(timeout=0)
        if preparation_worker is not None and preparation_ready:
            preparation_worker.join(timeout=0)
        return connected

    def _startup_loading_activity(self, elapsed: float) -> str:
        if elapsed < STARTUP_PHASE_SECONDS:
            return "Booting"
        if elapsed < STARTUP_PHASE_SECONDS * 2:
            return "Connecting"
        return "Screening"

    def _prepare_startup_state(self) -> StartupPreparation:
        """Precompute project metadata before the loading animation ends."""

        config = self.home.load_config()
        if not self._onboarding_complete(config):
            return StartupPreparation()
        if not self.home.is_repo_trusted(self.workspace_root):
            return StartupPreparation()
        return StartupPreparation(project=self._ensure_project())

    def _draw_startup_loading(
        self,
        stdscr: CursesWindow,
        frame: int,
        *,
        elapsed: float | None = None,
        visible_rows: int | None = None,
        removal_progress: float = 0.0,
        show_overlays: bool | None = None,
        line_progress: float | None = None,
        brand_progress: float | None = None,
        activity_text: str = "",
    ) -> None:
        """Render the fullscreen alphanumeric startup matrix."""

        with suppress(curses.error):
            curses.curs_set(0)
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if height <= 0 or width <= 0:
            return

        column_count = math.ceil(width / STARTUP_COLUMN_WIDTH)
        column_heights = self._startup_column_heights(
            column_count,
            height,
            elapsed,
            visible_rows,
        )
        removal_progress = min(1.0, max(0.0, removal_progress))
        line_reveal, brand_reveal = self._startup_overlay_progress(
            elapsed,
            line_progress,
            brand_progress,
        )
        overlays_visible = (
            all(column_height >= height for column_height in column_heights)
            and removal_progress == 0.0
            if show_overlays is None
            else show_overlays
        )

        rng = random.Random((frame + 1) * 104_729 + height * 8_191 + width * 193)
        matrix_attr = self._attr("matrix_dim")
        background_attr = self._attr("background")
        for y in range(height):
            line = self._startup_matrix_line(
                rng,
                y,
                width,
                column_heights,
                removal_progress,
            )
            row_attr = matrix_attr if line.strip() else background_attr
            with suppress(curses.error):
                stdscr.addnstr(y, 0, line, width, row_attr)

        if overlays_visible:
            self._draw_startup_function(
                stdscr,
                height,
                width,
                frame,
                reveal_progress=line_reveal,
                removal_progress=removal_progress,
            )
            self._draw_startup_brand(
                stdscr,
                height,
                width,
                frame,
                reveal_progress=brand_reveal,
                removal_progress=removal_progress,
            )
        if activity_text:
            x = max(0, (width - len(activity_text)) // 2)
            y = max(0, height - 2)
            self._add(stdscr, y, x, activity_text, len(activity_text), self._attr("accent"))
        stdscr.refresh()

    def _run_startup_wipe(self, stdscr: CursesWindow, frame: int) -> int:
        height, _ = stdscr.getmaxyx()
        if height <= 0:
            return frame
        started_at = time.monotonic()
        now = started_at
        deadline = started_at + STARTUP_WIPE_SECONDS
        while now < deadline:
            progress = min(1.0, max(0.0, (now - started_at) / STARTUP_WIPE_SECONDS))
            self._draw_startup_loading(
                stdscr,
                frame,
                visible_rows=height,
                removal_progress=progress,
                show_overlays=True,
                line_progress=1.0,
                brand_progress=1.0,
            )
            frame += 1
            time.sleep(STARTUP_FRAME_SECONDS)
            now = time.monotonic()
        self._draw_startup_loading(
            stdscr,
            frame,
            visible_rows=height,
            removal_progress=1.0,
            show_overlays=True,
            line_progress=1.0,
            brand_progress=1.0,
        )
        return frame + 1

    def _startup_overlay_progress(
        self,
        elapsed: float | None,
        line_progress: float | None,
        brand_progress: float | None,
    ) -> tuple[float, float]:
        if line_progress is not None and brand_progress is not None:
            return (
                min(1.0, max(0.0, line_progress)),
                min(1.0, max(0.0, brand_progress)),
            )
        if elapsed is None:
            return (
                1.0 if line_progress is None else min(1.0, max(0.0, line_progress)),
                1.0 if brand_progress is None else min(1.0, max(0.0, brand_progress)),
            )
        overlay_elapsed = elapsed - STARTUP_REVEAL_SECONDS - STARTUP_OVERLAY_DELAY_SECONDS
        computed_line = min(1.0, max(0.0, overlay_elapsed / STARTUP_LINE_REVEAL_SECONDS))
        computed_brand = min(1.0, max(0.0, overlay_elapsed / STARTUP_BRAND_REVEAL_SECONDS))
        return (
            computed_line if line_progress is None else min(1.0, max(0.0, line_progress)),
            computed_brand if brand_progress is None else min(1.0, max(0.0, brand_progress)),
        )

    def _startup_column_heights(
        self,
        column_count: int,
        height: int,
        elapsed: float | None,
        visible_rows: int | None,
    ) -> tuple[int, ...]:
        if visible_rows is not None:
            visible_height = max(0, min(height, visible_rows))
            return tuple(visible_height for _ in range(column_count))
        if elapsed is None or elapsed >= STARTUP_REVEAL_SECONDS:
            return tuple(height for _ in range(column_count))
        return tuple(
            self._startup_column_height(column, height, elapsed)
            for column in range(column_count)
        )

    def _startup_column_height(self, column: int, height: int, elapsed: float) -> int:
        rng = random.Random((column + 1) * 7_919)
        start_delay = rng.uniform(0.0, STARTUP_REVEAL_SECONDS * 0.45)
        duration = rng.uniform(STARTUP_REVEAL_SECONDS * 0.35, STARTUP_REVEAL_SECONDS * 0.78)
        progress = min(1.0, max(0.0, (elapsed - start_delay) / duration))
        eased_progress = 1.0 - ((1.0 - progress) ** 2)
        return max(0, min(height, round(height * eased_progress)))

    def _startup_matrix_line(
        self,
        rng: random.Random,
        y: int,
        width: int,
        column_heights: tuple[int, ...],
        removal_progress: float,
    ) -> str:
        parts: list[str] = []
        for x in range(width):
            column = x // STARTUP_COLUMN_WIDTH
            column_height = column_heights[min(column, len(column_heights) - 1)]
            visible = y < column_height and not self._startup_cell_removed(
                x,
                y,
                removal_progress,
            )
            if visible:
                parts.append(rng.choice(STARTUP_MATRIX_ALPHABET))
            else:
                parts.append(" ")
        return "".join(parts)

    def _startup_cell_removed(
        self,
        x: int,
        y: int,
        removal_progress: float,
    ) -> bool:
        if removal_progress <= 0:
            return False
        if removal_progress >= 1:
            return True
        threshold = int(removal_progress * 10_000)
        return self._startup_cell_rank(x, y, 0xA11CE) < threshold

    def _startup_cell_rank(self, x: int, y: int, salt: int) -> int:
        value = ((x + 0x9E37_79B9) * 0x85EB_CA6B) & 0xFFFF_FFFF
        value ^= ((y + 0xC2B2_AE35) * 0x27D4_EB2D) & 0xFFFF_FFFF
        value ^= salt & 0xFFFF_FFFF
        value ^= value >> 16
        value = (value * 0x7FEB_352D) & 0xFFFF_FFFF
        value ^= value >> 15
        value = (value * 0x846C_A68B) & 0xFFFF_FFFF
        value ^= value >> 16
        return value % 10_000

    def _draw_startup_function(
        self,
        stdscr: CursesWindow,
        height: int,
        width: int,
        frame: int,
        *,
        reveal_progress: float,
        removal_progress: float,
    ) -> None:
        reveal_progress = min(1.0, max(0.0, reveal_progress))
        if height < 3 or width < 2 or reveal_progress <= 0:
            return
        attr = self._attr("matrix_function")
        char_rng = random.Random((frame // 3 + 1) * 65_537 + width * 97 + height)
        previous_y: int | None = None
        max_x = min(width - 1, math.floor((width - 1) * reveal_progress))
        for x in range(max_x + 1):
            y = self._startup_function_y(x, width, height, frame)
            ys: tuple[int, ...]
            if previous_y is None:
                ys = (y,)
            else:
                start = min(previous_y, y)
                stop = max(previous_y, y)
                ys = tuple(range(start, stop + 1))
            for point_y in ys:
                if self._startup_cell_removed(x, point_y, removal_progress):
                    continue
                self._add(
                    stdscr,
                    point_y,
                    x,
                    char_rng.choice(STARTUP_MATRIX_ALPHABET),
                    1,
                    attr,
                )
            previous_y = y

    def _startup_function_y(self, x: int, width: int, height: int, frame: int) -> int:
        t = x / max(1, width - 1)
        slow_time = frame * 0.032
        wave = (
            math.sin((t * math.tau * 1.17) + (slow_time * 0.83)) * 0.38
            + math.sin((t * math.tau * 2.71) - (slow_time * 0.47) + 1.9) * 0.25
            + math.sin((t * math.tau * 4.63) + (slow_time * 0.29) + 0.7) * 0.17
            + math.sin((t * math.tau * 0.53) - (slow_time * 0.19) + 2.6) * 0.20
        )
        center = (height - 1) / 2 + math.sin(slow_time * 0.41) * max(1.0, height * 0.08)
        amplitude = max(1.0, (height - 4) * 0.36)
        return max(0, min(height - 1, round(center - wave * amplitude)))

    def _draw_startup_brand(
        self,
        stdscr: CursesWindow,
        height: int,
        width: int,
        frame: int,
        *,
        reveal_progress: float,
        removal_progress: float,
    ) -> None:
        reveal_progress = min(1.0, max(0.0, reveal_progress))
        if reveal_progress <= 0:
            return
        scale_x = 2 if width >= 90 else 1
        glyph_width = len(STARTUP_ANOMX_GLYPH[0]) * scale_x
        dot_gap = 2 * scale_x
        dot_width = scale_x
        brand_width = glyph_width + dot_gap + dot_width
        y = 2 if height >= 14 else 0
        x = max(0, (width - brand_width) // 2)
        attr = self._attr("matrix_brand")
        rng = random.Random((frame + 1) * 131_071 + width * 17 + height * 31)
        for row_index, row in enumerate(STARTUP_ANOMX_GLYPH):
            draw_y = y + row_index
            if draw_y >= height:
                return
            draw_x = x
            for marker in row:
                if marker == "#":
                    for offset in range(scale_x):
                        cell_x = draw_x + offset
                        if self._startup_brand_cell_hidden(
                            cell_x,
                            draw_y,
                            reveal_progress,
                            removal_progress,
                        ):
                            continue
                        self._add(
                            stdscr,
                            draw_y,
                            cell_x,
                            rng.choice(STARTUP_MATRIX_ALPHABET),
                            1,
                            attr,
                        )
                draw_x += scale_x
        dot_y = y + len(STARTUP_ANOMX_GLYPH) - 1
        dot_x = x + glyph_width + dot_gap
        if dot_y < height:
            for offset in range(dot_width):
                cell_x = dot_x + offset
                if cell_x >= width or self._startup_brand_cell_hidden(
                    cell_x,
                    dot_y,
                    reveal_progress,
                    removal_progress,
                ):
                    continue
                self._add(
                    stdscr,
                    dot_y,
                    cell_x,
                    rng.choice(STARTUP_MATRIX_ALPHABET),
                    1,
                    self._attr("brand_dot"),
                )

    def _startup_brand_cell_hidden(
        self,
        x: int,
        y: int,
        reveal_progress: float,
        removal_progress: float,
    ) -> bool:
        if self._startup_cell_removed(x, y, removal_progress):
            return True
        threshold = int(reveal_progress * 10_000)
        return self._startup_cell_rank(x, y, 0xB4A3D) >= threshold

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

        user_name = self._prompt_text(
            stdscr,
            title="Your Name",
            label="Name",
            optional=False,
        )
        if not user_name:
            return False

        config = self.home.load_config()
        config["onboarding_complete"] = True
        config["provider"] = provider.key
        config["model"] = model
        config["thinking_intensity"] = thinking_intensity
        config["user_name"] = user_name.strip()
        self.home.save_config(config)
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

    def _startup_project(
        self,
        stdscr: CursesWindow,
        preparation: StartupPreparation | None = None,
    ) -> ProjectRecord:
        prepared_project = preparation.project if preparation is not None else None
        if prepared_project is not None and prepared_project.path == self.project_path:
            return prepared_project
        stored_project = self.home.project_for_path(self.project_path)
        if stored_project is not None:
            return stored_project
        return self._ensure_project_with_loading(stdscr)

    def _ensure_project_with_loading(self, stdscr: CursesWindow) -> ProjectRecord:
        result: queue.SimpleQueue[ProjectRecord] = queue.SimpleQueue()

        def run_project_setup() -> None:
            result.put(self._ensure_project())

        worker = threading.Thread(target=run_project_setup, daemon=True)
        worker.start()
        frame = 0
        with suppress(curses.error):
            stdscr.nodelay(True)
        try:
            while worker.is_alive():
                self._draw_startup_loading(
                    stdscr,
                    frame,
                    elapsed=min(STARTUP_LOADING_SECONDS, frame * STARTUP_FRAME_SECONDS),
                    activity_text="Screening",
                )
                frame += 1
                with suppress(curses.error):
                    stdscr.get_wch()
                time.sleep(STARTUP_FRAME_SECONDS)
        finally:
            with suppress(curses.error):
                stdscr.nodelay(False)
        worker.join(timeout=0)
        with suppress(queue.Empty):
            return result.get_nowait()
        return self.home.save_project(self.project_path, self._fallback_project_name())

    def _ensure_project(self) -> ProjectRecord:
        stored_project = self.home.project_for_path(self.project_path)
        if stored_project is not None:
            return stored_project
        name = ""
        outline = self._project_directory_outline(self.project_path)
        with suppress(Exception):
            name = self.runtime.suggest_project_name(self.project_path, outline) or ""
        return self.home.save_project(
            self.project_path,
            name or self._fallback_project_name(),
        )

    def _fallback_project_name(self) -> str:
        name = self.project_path.name.strip() or "Anomx Project"
        words = [
            word
            for word in re.split(r"[\s._-]+", name)
            if word.strip()
        ]
        if not words:
            return "Anomx Project"
        if len(words) == 1:
            words.append("Project")
        return " ".join(word[:1].upper() + word[1:] for word in words[:3])

    def _project_directory_outline(self, root: Path, max_depth: int = 3) -> str:
        lines: list[str] = []

        def visit(path: Path, depth: int, prefix: str) -> None:
            if depth > max_depth:
                return
            try:
                entries = sorted(
                    path.iterdir(),
                    key=lambda item: (not item.is_dir(), item.name.lower()),
                )
            except OSError:
                return
            visible = [
                entry
                for entry in entries
                if entry.name not in IGNORED_FILE_REFERENCE_DIRS
            ][:24]
            for entry in visible:
                marker = "/" if entry.is_dir() else ""
                lines.append(f"{prefix}{entry.name}{marker}")
                if entry.is_dir():
                    visit(entry, depth + 1, f"{prefix}  ")

        lines.append(f"{root.name or root}/")
        visit(root, 1, "  ")
        return "\n".join(lines[:160])

    def _run_project(self, stdscr: CursesWindow, project: ProjectRecord) -> int:
        input_text = ""
        cursor = 0
        selected = 0
        scroll = 0
        command_selected = 0
        file_references: dict[str, str] = {}
        file_selected = 0
        delete_pending_index: int | None = None
        frame = 0
        exit_confirm_deadline = 0.0
        prompt_notice = ""
        prompt_notice_role = "light"

        while True:
            if exit_confirm_deadline and time.monotonic() > exit_confirm_deadline:
                exit_confirm_deadline = 0.0
                prompt_notice = ""
            self._drain_active_session_turns(stdscr)
            sessions = self._project_sessions(project.path)
            selected = max(0, min(selected, len(sessions) - 1)) if sessions else 0
            if delete_pending_index is not None and delete_pending_index >= len(sessions):
                delete_pending_index = None
            command_suggestions = (
                self._filtered_commands(input_text) if input_text.startswith("/") else []
            )
            if command_suggestions:
                command_selected = min(command_selected, len(command_suggestions) - 1)
            else:
                command_selected = 0
            file_reference_token = self._active_file_reference_token(input_text, cursor)
            file_suggestions = (
                self._filtered_file_references(file_reference_token[2])
                if file_reference_token is not None
                else []
            )
            file_selected = (
                min(file_selected, len(file_suggestions) - 1) if file_suggestions else 0
            )
            scroll = self._draw_project(
                stdscr,
                project,
                sessions,
                selected,
                scroll,
                input_text,
                cursor,
                prompt_notice,
                prompt_notice_role,
                frame,
                command_suggestions=command_suggestions,
                command_selected=command_selected,
                delete_pending_index=delete_pending_index,
                file_suggestions=file_suggestions if file_reference_token else [],
                file_selected=file_selected,
                file_references=file_references,
                file_reference_active=file_reference_token is not None,
            )
            animated = self._project_animation_active(sessions)
            if animated:
                with suppress(curses.error, AttributeError):
                    stdscr.nodelay(True)
                key = self._read_nonblocking_key(stdscr)
                if key is None:
                    time.sleep(0.08)
                    frame += 1
                    continue
                with suppress(curses.error, AttributeError):
                    stdscr.nodelay(False)
            else:
                with suppress(curses.error, AttributeError):
                    stdscr.nodelay(False)
                key = self._read_prompt_key(stdscr)
            frame += 1

            if self._is_shift_tab(key):
                self._cycle_agent_mode()
                continue
            if self._is_ctrl_c(key):
                if input_text:
                    input_text = ""
                    cursor = 0
                    command_selected = 0
                    file_references = {}
                    file_selected = 0
                    prompt_notice = ""
                    continue
                now = time.monotonic()
                if exit_confirm_deadline and now <= exit_confirm_deadline:
                    return 0
                exit_confirm_deadline = now + 3.0
                prompt_notice = EXIT_ANOMX_CONFIRM_NOTICE
                prompt_notice_role = "light"
                continue
            if self._is_escape(key):
                delete_pending_index = None
                input_text = ""
                cursor = 0
                command_selected = 0
                file_references = {}
                file_selected = 0
                prompt_notice = ""
                continue
            if key == curses.KEY_UP:
                if file_suggestions:
                    file_selected = max(0, file_selected - 1)
                elif command_suggestions:
                    command_selected = max(0, command_selected - 1)
                elif input_text:
                    cursor = self._move_prompt_cursor_row(stdscr, input_text, cursor, -1)
                elif sessions:
                    selected = max(0, selected - 1)
                    delete_pending_index = None
                continue
            if key == curses.KEY_DOWN:
                if file_suggestions:
                    file_selected = min(len(file_suggestions) - 1, file_selected + 1)
                elif command_suggestions:
                    command_selected = min(len(command_suggestions) - 1, command_selected + 1)
                elif input_text:
                    cursor = self._move_prompt_cursor_row(stdscr, input_text, cursor, 1)
                elif sessions:
                    selected = min(len(sessions) - 1, selected + 1)
                    delete_pending_index = None
                continue
            if key == curses.KEY_PPAGE:
                scroll = max(0, scroll - 5)
                selected = max(0, selected - 5)
                delete_pending_index = None
                continue
            if key == curses.KEY_NPAGE:
                selected = min(max(0, len(sessions) - 1), selected + 5)
                scroll += 5
                delete_pending_index = None
                continue
            if self._is_option_left(key):
                cursor = self._previous_prompt_word(input_text, cursor)
                continue
            if self._is_option_right(key):
                cursor = self._next_prompt_word(input_text, cursor)
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
                action = self._project_mouse_action(
                    stdscr,
                    input_text,
                    command_suggestions,
                    command_selected,
                    file_suggestions=file_suggestions if file_reference_token else [],
                    file_selected=file_selected,
                )
                if action is None:
                    continue
                if action.kind == "cursor":
                    cursor = action.value
                elif action.kind == "scroll":
                    scroll = max(0, scroll + action.value)
                elif action.kind == "open_project_session":
                    delete_pending_index = None
                    opened = self._open_project_session(stdscr, sessions[action.value])
                    if isinstance(opened, int):
                        return opened
                    input_text = ""
                    cursor = 0
                    command_selected = 0
                elif action.kind == "file_reference":
                    if file_reference_token is not None:
                        input_text, cursor = self._insert_file_reference(
                            input_text,
                            cursor,
                            file_reference_token,
                            file_suggestions[action.value],
                            file_references,
                        )
                        file_selected = 0
                elif action.kind == "command":
                    command = command_suggestions[action.value].command
                    result = self._handle_project_command(
                        stdscr,
                        command,
                        command,
                        sessions,
                        selected,
                    )
                    if result == "exit":
                        return 0
                    if isinstance(result, SessionRecord) and command in {"/new", "/open"}:
                        opened = self._open_project_session(stdscr, result)
                        if isinstance(opened, int):
                            return opened
                    sessions = self._project_sessions(project.path)
                    selected = max(0, min(selected, len(sessions) - 1)) if sessions else 0
                    delete_pending_index = None
                    input_text = ""
                    cursor = 0
                    command_selected = 0
                continue
            if self._is_raw_mouse_fragment_key(key):
                continue
            if self._is_ctrl_d(key):
                if sessions:
                    delete_pending_index = selected
                    prompt_notice = ""
                continue
            if self._is_shift_enter(key):
                input_text = input_text[:cursor] + "\n" + input_text[cursor:]
                cursor += 1
                command_selected = 0
                delete_pending_index = None
                prompt_notice = ""
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
                if submitted:
                    if submitted.startswith("/"):
                        command = self._submitted_command(
                            submitted,
                            command_suggestions,
                            command_selected,
                        )
                        result = self._handle_project_command(
                            stdscr,
                            command,
                            submitted,
                            sessions,
                            selected,
                        )
                        if result == "exit":
                            return 0
                        if isinstance(result, SessionRecord) and command in {"/new", "/open"}:
                            opened = self._open_project_session(stdscr, result)
                            if isinstance(opened, int):
                                return opened
                    else:
                        backend_message = self._backend_message_for_prompt(
                            submitted,
                            file_references,
                        )
                        prompt_session = self._start_project_prompt_session(
                            submitted,
                            backend_message=backend_message,
                            file_references=dict(file_references),
                        )
                        opened = self._open_project_session(stdscr, prompt_session)
                        if isinstance(opened, int):
                            return opened
                        sessions = self._project_sessions(project.path)
                        selected = max(0, min(selected, len(sessions) - 1)) if sessions else 0
                        prompt_notice = ""
                    input_text = ""
                    cursor = 0
                    command_selected = 0
                    file_references = {}
                    file_selected = 0
                    delete_pending_index = None
                    continue
                if delete_pending_index == selected and sessions:
                    self._delete_project_session(sessions[selected])
                    sessions = self._project_sessions(project.path)
                    selected = max(0, min(selected, len(sessions) - 1)) if sessions else 0
                    delete_pending_index = None
                    prompt_notice = ""
                    continue
                if sessions:
                    delete_pending_index = None
                    opened = self._open_project_session(stdscr, sessions[selected])
                    if isinstance(opened, int):
                        return opened
                continue
            if self._is_option_delete(key):
                word_start = self._previous_prompt_word(input_text, cursor)
                input_text = input_text[:word_start] + input_text[cursor:]
                cursor = word_start
                command_selected = 0
                file_selected = 0
                delete_pending_index = None
                continue
            if self._is_backspace(key):
                if cursor > 0:
                    input_text = input_text[: cursor - 1] + input_text[cursor:]
                    cursor -= 1
                    command_selected = 0
                    file_selected = 0
                    delete_pending_index = None
                continue
            if isinstance(key, str) and key.isprintable():
                input_text = input_text[:cursor] + key + input_text[cursor:]
                cursor += len(key)
                command_selected = 0
                file_selected = 0
                delete_pending_index = None
                prompt_notice = ""

    def _open_project_session(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
    ) -> int | None:
        opened = self._prepare_session_for_opening(session)
        result = self._run_session(stdscr, opened)
        return None if result == "project" else int(result)

    def _start_project_prompt_session(
        self,
        message: str,
        backend_message: str = "",
        file_references: Mapping[str, str] | None = None,
    ) -> SessionRecord:
        session = self._create_session()
        payload: dict[str, object] = {"message": message}
        if backend_message:
            payload["backend_message"] = backend_message
        if file_references:
            payload["file_references"] = file_references
        self.home.append_session_event(session.path, "user_message", payload)
        self._maybe_start_session_rename(session)
        self._start_session_turn(session)
        return session

    def _handle_project_command(
        self,
        stdscr: CursesWindow,
        command: str,
        submitted: str,
        sessions: Sequence[SessionRecord],
        selected: int,
    ) -> str | SessionRecord | None:
        if command == "/exit":
            return "exit"
        if command == "/new":
            return self._create_session()
        current_session = self._project_command_session(sessions, selected)
        if current_session is None:
            current_session = self._create_session()
        if command == "/open":
            return current_session
        return self._handle_command(stdscr, command, current_session, submitted)

    def _project_command_session(
        self,
        sessions: Sequence[SessionRecord],
        selected: int,
    ) -> SessionRecord | None:
        if not sessions:
            return None
        index = max(0, min(selected, len(sessions) - 1))
        return sessions[index]

    def _delete_project_session(self, session: SessionRecord) -> None:
        turn = self._active_turn_for_session(session)
        if turn is not None:
            turn.runtime.shutdown(session.path)
            turn.completed = True
            self._active_session_turns.pop(self._session_turn_key(session), None)
        self.home.delete_session(session.path)
        with suppress(KeyError):
            self._session_event_cache.pop(session.path.expanduser().resolve())
        with suppress(KeyError):
            self._message_line_cache.pop(session.path.expanduser().resolve())
        with suppress(KeyError):
            self._rendered_message_cache.pop(session.path.expanduser().resolve())

    def _project_sessions(self, project_path: Path) -> list[SessionRecord]:
        project_key = str(project_path.expanduser().resolve())
        sessions = [
            session
            for session in self.home.list_sessions(limit=None)
            if session.cwd.strip() == project_key
        ]
        sessions.sort(
            key=lambda session: (session.last_user_at or session.created_at, session.created_at),
            reverse=True,
        )
        return sessions

    def _cleanup_stale_project_sessions(self, project: ProjectRecord) -> None:
        for session in self._project_sessions(project.path):
            if self._active_turn_for_session(session) is None:
                self.runtime.cleanup_session_runtime_state(session.path)

    def _project_animation_active(self, sessions: Sequence[SessionRecord]) -> bool:
        return any(self._active_turn_for_session(session) is not None for session in sessions)

    def _startup_session(
        self,
        stdscr: CursesWindow,
        preparation: StartupPreparation | None = None,
    ) -> SessionRecord:
        previous_session = (
            preparation.previous_session
            if preparation is not None and preparation.previous_session is not None
            else self._latest_continuable_session()
        )
        if previous_session is None:
            return self._create_session()

        statement = (
            preparation.continuation_statement
            if preparation is not None
            and preparation.previous_session is not None
            and preparation.previous_session.session_id == previous_session.session_id
            and preparation.continuation_statement
            else self._continue_session_statement(previous_session)
        )
        if self._run_continue_session_prompt(stdscr, previous_session, statement):
            return self._prepare_session_for_opening(previous_session)
        return self._create_session()

    def _prepare_session_for_opening(self, session: SessionRecord) -> SessionRecord:
        if session.unread:
            self.home.set_session_unread(session.path, False)
            session = replace(session, unread=False)
        if self._active_turn_for_session(session) is None:
            self.runtime.cleanup_session_runtime_state(session.path)
        return session

    def _latest_continuable_session(self) -> SessionRecord | None:
        for session in self.home.list_sessions(limit=None):
            if self._session_belongs_to_workspace(session):
                return session
        return None

    def _session_belongs_to_workspace(self, session: SessionRecord) -> bool:
        cwd = session.cwd.strip()
        if not cwd:
            return False
        with suppress(OSError, RuntimeError):
            return discover_workspace_root(Path(cwd)) == self.workspace_root
        return False

    def _continue_session_statement(self, session: SessionRecord) -> str:
        workspace_name = self.workspace_root.name or str(self.workspace_root)
        with suppress(Exception):
            return self.runtime.suggest_session_continuation(
                session.path,
                workspace_name,
            )
        return self._fallback_continue_session_statement(session)

    def _fallback_continue_session_statement(self, session: SessionRecord) -> str:
        title = session.title.strip()
        if title and title != "New session":
            return f"Do you want to continue the previous session titled {title}?"
        workspace_name = self.workspace_root.name or str(self.workspace_root)
        return f"Do you want to continue the previous Anomx session in {workspace_name}?"

    def _run_continue_session_prompt(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        statement: str,
    ) -> bool:
        self.state = AgentState.CONTINUE_SESSION
        selected = 0
        try:
            while True:
                self._draw_continue_session_prompt(stdscr, session, statement, selected)
                key = stdscr.get_wch()
                if self._is_escape(key) or self._is_ctrl_c(key):
                    return False
                if key == curses.KEY_UP:
                    selected = max(0, selected - 1)
                elif key == curses.KEY_DOWN:
                    selected = min(1, selected + 1)
                elif self._is_enter(key):
                    return selected == 0
        finally:
            self.state = AgentState.NEW_SESSION

    def _create_session(self) -> SessionRecord:
        config = self.home.load_config()
        return self.home.create_session(
            self.cwd,
            provider=str(config.get("provider", "openai")),
            model=str(config.get("model", "gpt-5.5")),
            mode=self.agent_mode,
        )

    def _run_session(self, stdscr: CursesWindow, session: SessionRecord) -> int | str:
        current_session = session
        self._activate_agent_mode(current_session.mode)
        input_text = ""
        cursor = 0
        file_references: dict[str, str] = {}
        image_attachments: dict[str, dict[str, str]] = {}
        scroll = 0
        command_selected = 0
        file_selected = 0
        pinned_anchor: int | None = None
        exit_confirm_deadline = 0.0
        exit_notice = ""
        frame = 0

        while True:
            if exit_confirm_deadline and time.monotonic() > exit_confirm_deadline:
                exit_confirm_deadline = 0.0
                exit_notice = ""
            current_session = self._process_title_events(stdscr, current_session)
            active_turn = self._active_turn_for_session(current_session)
            if active_turn is not None and active_turn.worker is not None and active_turn.worker.is_alive():
                turn_result = self._run_backend_turn(
                    stdscr,
                    current_session,
                    anchor_line=pinned_anchor,
                )
                if turn_result.exit_requested:
                    return 0
                if turn_result.back_requested:
                    return "project"
                input_text = turn_result.input_text
                cursor = turn_result.cursor
                pinned_anchor = turn_result.anchor_line
                scroll = turn_result.scroll
                continue
            if active_turn is not None:
                self._complete_session_turn(
                    stdscr,
                    active_turn,
                    anchor_line=pinned_anchor,
                    scroll=scroll,
                    input_text=input_text,
                    cursor=cursor,
                    render_final=True,
                )
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
                image_attachments=image_attachments,
                anchor_line=pinned_anchor,
                prompt_notice=exit_notice,
                prompt_notice_role="light",
                working_frame=frame,
            )
            if viewport is not None:
                scroll = viewport.scroll
            if self._session_animation_active(current_session.path):
                with suppress(curses.error, AttributeError):
                    stdscr.nodelay(True)
                key = self._read_nonblocking_key(stdscr)
                if key is None:
                    time.sleep(0.08)
                    frame += 1
                    continue
                with suppress(curses.error, AttributeError):
                    stdscr.nodelay(False)
            else:
                with suppress(curses.error, AttributeError):
                    stdscr.nodelay(False)
                key = self._read_prompt_key(stdscr)
            frame += 1

            if self._is_shift_tab(key):
                current_session = replace(
                    current_session,
                    mode=self._cycle_agent_mode(current_session),
                )
                continue
            if self._is_ctrl_c(key):
                if input_text:
                    input_text = ""
                    cursor = 0
                    command_selected = 0
                    file_selected = 0
                    file_references = {}
                    image_attachments = {}
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
                    image_attachments = {}
                elif pinned_anchor is not None:
                    pinned_anchor = None
                elif self._session_selection is not None:
                    self._clear_session_selection()
                else:
                    return "project"
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
                elif mouse_action.kind == "back_project":
                    return "project"
                elif mouse_action.kind == "scroll":
                    pinned_anchor = None
                    scroll += mouse_action.value
                elif mouse_action.kind == "toggle_work":
                    self._toggle_work_turn(mouse_action.text)
                elif mouse_action.kind == "toggle_plan":
                    self._toggle_plan(current_session.path)
                elif mouse_action.kind == "toggle_work_line":
                    self._toggle_work_line(mouse_action.text)
                elif mouse_action.kind == "toggle_activity_item":
                    self._toggle_activity_item(mouse_action.text)
                elif mouse_action.kind == "toggle_activity_entry":
                    self._toggle_activity_entry(mouse_action.text)
                elif mouse_action.kind == "scroll_activity_item":
                    self._scroll_activity_item(mouse_action.text, mouse_action.value)
                elif mouse_action.kind == "kill_process":
                    self._runtime_for_session(current_session).end_process(
                        mouse_action.text,
                        current_session.path,
                    )
                elif mouse_action.kind == "command":
                    command = command_suggestions[mouse_action.value].command
                    command_result = self._handle_command(stdscr, command, current_session)
                    if command_result == "exit":
                        return 0
                    if isinstance(command_result, SessionRecord):
                        current_session = command_result
                        self._activate_agent_mode(current_session.mode)
                        self._prompt_placeholder = random.choice(PROMPT_PLACEHOLDERS)
                        scroll = 0
                        pinned_anchor = None
                    input_text = ""
                    cursor = 0
                    command_selected = 0
                    file_selected = 0
                    file_references = {}
                    image_attachments = {}
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
                    image_attachments = {}
                continue
            raw_mouse_action = self._raw_mouse_action(
                key,
                stdscr,
                input_text,
                command_suggestions,
                command_selected,
                file_suggestions,
                file_selected,
            )
            if raw_mouse_action is not None:
                if raw_mouse_action.kind == "cursor":
                    cursor = raw_mouse_action.value
                elif raw_mouse_action.kind == "back_project":
                    return "project"
                elif raw_mouse_action.kind == "scroll":
                    pinned_anchor = None
                    scroll += raw_mouse_action.value
                elif raw_mouse_action.kind == "toggle_work":
                    self._toggle_work_turn(raw_mouse_action.text)
                elif raw_mouse_action.kind == "toggle_plan":
                    self._toggle_plan(current_session.path)
                elif raw_mouse_action.kind == "toggle_work_line":
                    self._toggle_work_line(raw_mouse_action.text)
                elif raw_mouse_action.kind == "toggle_activity_item":
                    self._toggle_activity_item(raw_mouse_action.text)
                elif raw_mouse_action.kind == "toggle_activity_entry":
                    self._toggle_activity_entry(raw_mouse_action.text)
                elif raw_mouse_action.kind == "scroll_activity_item":
                    self._scroll_activity_item(raw_mouse_action.text, raw_mouse_action.value)
                elif raw_mouse_action.kind == "kill_process":
                    self._runtime_for_session(current_session).end_process(
                        raw_mouse_action.text,
                        current_session.path,
                    )
                elif raw_mouse_action.kind == "command":
                    command = command_suggestions[raw_mouse_action.value].command
                    command_result = self._handle_command(stdscr, command, current_session)
                    if command_result == "exit":
                        return 0
                    if isinstance(command_result, SessionRecord):
                        current_session = command_result
                        self._activate_agent_mode(current_session.mode)
                        self._prompt_placeholder = random.choice(PROMPT_PLACEHOLDERS)
                        scroll = 0
                        pinned_anchor = None
                    input_text = ""
                    cursor = 0
                    command_selected = 0
                    file_selected = 0
                    file_references = {}
                    image_attachments = {}
                elif raw_mouse_action.kind == "file_reference":
                    if file_reference_token is None:
                        continue
                    input_text, cursor = self._insert_file_reference(
                        input_text,
                        cursor,
                        file_reference_token,
                        file_suggestions[raw_mouse_action.value],
                        file_references,
                    )
                    file_selected = 0
                elif raw_mouse_action.kind == "skill":
                    skill = self._skill_for_command(f"/{raw_mouse_action.text}")
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
                    image_attachments = {}
                continue
            if self._is_raw_mouse_fragment_key(key):
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
                submitted_image_attachments = self._active_image_attachments(
                    submitted,
                    image_attachments,
                )
                backend_message = self._backend_message_for_prompt(
                    submitted,
                    file_references,
                )
                submitted_file_references = dict(file_references)
                selected_command = command_selected
                if not submitted:
                    input_text = ""
                    cursor = 0
                    file_references = {}
                    image_attachments = {}
                    command_selected = 0
                    file_selected = 0
                    continue
                if submitted.startswith("/"):
                    input_text = ""
                    cursor = 0
                    file_references = {}
                    image_attachments = {}
                    command_selected = 0
                    file_selected = 0
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
                        self._activate_agent_mode(current_session.mode)
                        self._prompt_placeholder = random.choice(PROMPT_PLACEHOLDERS)
                        scroll = 0
                        pinned_anchor = None
                    continue
                if self._start_hints_visible_for_prompt(
                    current_session.path,
                    messages,
                    input_text,
                    command_suggestions,
                    file_suggestions,
                ):
                    self._animate_start_hints_exit(
                        stdscr,
                        current_session,
                        messages,
                        scroll,
                        pinned_anchor,
                    )
                input_text = ""
                cursor = 0
                file_references = {}
                image_attachments = {}
                command_selected = 0
                file_selected = 0
                self.home.append_session_event(
                    current_session.path,
                    "user_message",
                    {
                        "message": submitted,
                        "backend_message": backend_message,
                        "file_references": submitted_file_references,
                        "image_attachments": list(submitted_image_attachments.values()),
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
                if turn_result.back_requested:
                    return "project"
                input_text = turn_result.input_text
                cursor = turn_result.cursor
                pinned_anchor = turn_result.anchor_line
                scroll = turn_result.scroll
                continue

            if self._is_option_delete(key):
                word_start = self._previous_prompt_word(input_text, cursor)
                input_text = input_text[:word_start] + input_text[cursor:]
                cursor = word_start
                command_selected = 0
                file_selected = 0
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
                input_text, cursor, added_images = self._consume_dropped_images(
                    input_text,
                    cursor,
                    image_attachments,
                )
                if added_images:
                    self._append_unsupported_image_notice(
                        current_session,
                        added_images,
                    )
                command_selected = 0
                file_selected = 0

    def _session_animation_active(self, session_path: Path) -> bool:
        return (
            self._start_hint_reveal_active()
            or self._has_running_session_activity(session_path)
            or self._plan_reveal_active(session_path)
        )

    def _has_running_session_activity(self, session_path: Path) -> bool:
        events = self._session_events(session_path)
        if running_process_snapshots(events):
            return True
        return False

    def _plan_reveal_active(self, session_path: Path) -> bool:
        events = self._session_events(session_path)
        plan_steps = latest_plan_steps(events)
        if not plan_steps:
            return False
        created_at = self._latest_plan_create_timestamp(events)
        if created_at is None:
            return False
        elapsed = max(0.0, (datetime.now(tz=UTC) - created_at).total_seconds())
        visible_count = min(
            len(plan_steps),
            max(1, int(elapsed // PLAN_STEP_REVEAL_SECONDS) + 1),
        )
        return visible_count < len(plan_steps)

    def _start_hints_visible_for_prompt(
        self,
        session_path: Path,
        messages: list[MessageLine],
        input_text: str,
        command_suggestions: Sequence[CommandSpec],
        file_suggestions: Sequence[MenuChoice],
    ) -> bool:
        active_bottom_panel = (
            self._command_bottom_panel(list(command_suggestions), 0)
            if command_suggestions
            else None
        )
        if active_bottom_panel is None and file_suggestions:
            active_bottom_panel = self._file_reference_bottom_panel(list(file_suggestions), 0)
        return self._should_draw_start_hints(
            messages,
            input_text,
            active_bottom_panel,
            None,
            latest_plan_steps(self._session_events(session_path)),
        )

    def _animate_start_hints_exit(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        messages: list[MessageLine],
        scroll: int,
        anchor_line: int | None,
    ) -> None:
        started_at = time.monotonic()
        frame = 0
        while True:
            progress = min(1.0, (time.monotonic() - started_at) / START_HINT_WIPE_SECONDS)
            self._draw_session(
                stdscr,
                session,
                messages,
                "",
                0,
                scroll,
                anchor_line=anchor_line,
                working_frame=frame,
                force_start_hints=True,
                start_hint_reveal_progress=1.0,
                start_hint_removal_progress=progress,
            )
            if progress >= 1.0:
                self._start_hint_reveal_started_at = None
                return
            frame += 1
            time.sleep(STARTUP_FRAME_SECONDS)

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
        for event in self._session_events(session_path):
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

        selected = 0
        current_scroll = 0
        delete_pending_index: int | None = None
        with suppress(curses.error):
            stdscr.nodelay(False)
        try:
            while True:
                selected = min(selected, len(sessions) - 1)
                choices = self._open_session_choices(
                    sessions,
                    selected,
                    delete_pending_index,
                )
                messages = self._read_message_lines(current_session.path)
                panel = BottomPanel(
                    "Open Session",
                    self._open_session_subtitle(delete_pending_index),
                    choices,
                    selected,
                )
                viewport = self._draw_session(
                    stdscr,
                    current_session,
                    messages,
                    "",
                    0,
                    current_scroll,
                    bottom_panel=panel,
                )
                if viewport is not None:
                    current_scroll = viewport.scroll
                key = stdscr.get_wch()
                if self._is_escape(key):
                    if delete_pending_index is not None:
                        delete_pending_index = None
                        continue
                    return None
                if self._is_ctrl_c(key):
                    return None
                if self._is_shift_tab(key):
                    current_session = replace(
                        current_session,
                        mode=self._cycle_agent_mode(current_session),
                    )
                    continue
                if key == curses.KEY_UP:
                    selected = max(0, selected - 1)
                    delete_pending_index = None
                    continue
                if key == curses.KEY_DOWN:
                    selected = min(len(sessions) - 1, selected + 1)
                    delete_pending_index = None
                    continue
                if key == curses.KEY_PPAGE:
                    panel_viewport = self._bottom_panel_viewport(stdscr, panel)
                    page_size = max(1, len(panel_viewport.visible_indices))
                    selected = max(0, selected - page_size)
                    delete_pending_index = None
                    continue
                if key == curses.KEY_NPAGE:
                    panel_viewport = self._bottom_panel_viewport(stdscr, panel)
                    page_size = max(1, len(panel_viewport.visible_indices))
                    selected = min(len(sessions) - 1, selected + page_size)
                    delete_pending_index = None
                    continue
                if key == curses.KEY_MOUSE:
                    choice = self._bottom_panel_mouse_choice(stdscr, panel)
                    if choice is not None:
                        return self._prepare_session_for_opening(sessions[choice])
                    continue
                if self._is_ctrl_d(key):
                    delete_pending_index = selected
                    continue
                if self._is_enter(key):
                    if delete_pending_index == selected:
                        deleted_session = sessions[selected]
                        deleted_current = (
                            deleted_session.path.expanduser().resolve()
                            == current_session.path.expanduser().resolve()
                        )
                        self.home.delete_session(deleted_session.path)
                        sessions = self.home.list_sessions(limit=None)
                        delete_pending_index = None
                        if not sessions:
                            return self._create_session()
                        selected = min(selected, len(sessions) - 1)
                        if deleted_current:
                            return self._prepare_session_for_opening(sessions[selected])
                        continue
                    return self._prepare_session_for_opening(sessions[selected])
        finally:
            self.state = AgentState.NEW_SESSION

    def _open_session_choices(
        self,
        sessions: list[SessionRecord],
        selected: int,
        delete_pending_index: int | None = None,
    ) -> tuple[MenuChoice, ...]:
        del selected, delete_pending_index
        choices = tuple(
            MenuChoice(
                label=session.title,
                detail=self._open_session_detail(session),
                value=str(index),
            )
            for index, session in enumerate(sessions)
        )
        return choices

    def _open_session_subtitle(self, delete_pending_index: int | None = None) -> str:
        action = "Enter to confirm" if delete_pending_index is not None else "ctrl+d Delete"
        return f"Choose a stored session · {action}"

    def _open_session_detail(
        self,
        session: SessionRecord,
    ) -> str:
        return (
            f"{self._message_count_label(session.message_count)} · "
            f"{session.created_at} · {self._session_location_label(session)}"
        )

    def _session_location_label(self, session: SessionRecord) -> str:
        return session.cwd.strip() or "Unknown location"

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
            if selected == "debug":
                self._run_debug_panel(stdscr, current_session)
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
        config = self.home.load_config()
        return (
            MenuChoice("Choose backend", "backend", "Select provider and enter API key"),
            MenuChoice("Choose model", "model", "Pick the model for the selected backend"),
            platform_choice,
            MenuChoice("Manage Debug Mode", "debug", self._debug_config_detail(config)),
            MenuChoice("History persistence", "history_persistence", "Store all sessions or none"),
            MenuChoice(
                "Clear all sessions",
                "clear_sessions",
                "Delete stored sessions except this one",
            ),
            MenuChoice("Done", "done", "Same as Esc"),
        )

    def _debug_config_detail(self, config: Mapping[str, object]) -> str:
        debug_active = bool(config.get("debug_mode"))
        full_logs = bool(config.get("debug_full_session_logs"))
        if not debug_active:
            return "debug mode false"
        if full_logs:
            return "debug mode true · full session logs true"
        return "debug mode true · full session logs false"

    def _run_debug_panel(
        self,
        stdscr: CursesWindow,
        current_session: SessionRecord,
    ) -> None:
        while True:
            config = self.home.load_config()
            selected = self._bottom_menu(
                stdscr,
                current_session,
                "Debug",
                "Configure crash logs and backend request snapshots",
                self._debug_menu_choices(config),
            )
            if selected is None:
                self.state = AgentState.CONFIG
                return
            if selected == "debug_mode":
                config["debug_mode"] = not bool(config.get("debug_mode"))
                self.home.save_config(config)
                continue
            if selected == "full_session_logs":
                config["debug_full_session_logs"] = not bool(
                    config.get("debug_full_session_logs")
                )
                self.home.save_config(config)
                continue
            if selected == "full_session_logs_path":
                current_path = str(self.home.full_session_logs_dir(config))
                value = self._prompt_text(
                    stdscr,
                    "Full Session Logs Path",
                    "Directory path",
                    default=current_path,
                )
                if value is not None:
                    config["debug_full_session_logs_path"] = value.strip() or str(
                        self.home.root
                    )
                    self.home.save_config(config)

    def _debug_menu_choices(
        self,
        config: Mapping[str, object],
    ) -> tuple[MenuChoice, ...]:
        return (
            MenuChoice(
                "Debug mode active",
                "debug_mode",
                self._bool_config_detail(config.get("debug_mode")),
            ),
            MenuChoice(
                "Full session logs",
                "full_session_logs",
                self._bool_config_detail(config.get("debug_full_session_logs")),
            ),
            MenuChoice(
                "Full session logs path",
                "full_session_logs_path",
                str(self.home.full_session_logs_dir(config)),
            ),
        )

    def _bool_config_detail(self, value: object) -> str:
        return "true" if bool(value) else "false"

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
        plan_expanded: bool = False,
        title_suffix: str = "",
    ) -> tuple[int, int]:
        with suppress(curses.error):
            curses.curs_set(0)
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        self._draw_header_box(
            stdscr,
            title,
            subtitle,
            plan_steps,
            header_meta,
            plan_expanded,
            title_suffix,
        )
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
        plan_expanded: bool = False,
        title_suffix: str = "",
    ) -> None:
        _, width = stdscr.getmaxyx()
        right_text = self._header_right_text(header_meta)
        top = 1
        subtitle_lines = self._header_subtitle_lines(subtitle)
        bottom = self._header_bottom(plan_steps, len(subtitle_lines), plan_expanded)
        horizontal = "─" * max(1, width - 6)
        self._add(stdscr, top, 2, f"╭{horizontal}╮", width - 4, self._attr("accent"))
        for y in range(top + 1, bottom):
            self._add(stdscr, y, 2, "│", 1, self._attr("accent"))
            self._add(stdscr, y, max(2, width - 3), "│", 1, self._attr("accent"))
        self._add(stdscr, bottom, 2, f"╰{horizontal}╯", width - 4, self._attr("accent"))

        brand = BRAND_NAME
        descriptor = AGENT_DESCRIPTOR
        right_text = self._fit_header_right_text(right_text, max(1, width - 8))
        right_x = max(4, width - len(right_text) - 5)
        brand_dot_x = 4 + len(brand)
        descriptor_x = brand_dot_x + len(BRAND_DOT) + 2
        self._add(stdscr, top + 1, 4, brand, width - 8, self._attr("accent"))
        self._add(
            stdscr,
            top + 1,
            brand_dot_x,
            BRAND_DOT,
            1,
            self._attr("brand_dot"),
        )
        self._add(
            stdscr,
            top + 1,
            descriptor_x,
            descriptor,
            max(1, width - descriptor_x - 4),
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
        self._draw_header_title(
            stdscr,
            top + 2,
            4,
            title,
            plan_steps,
            plan_expanded,
            width - 8,
            title_suffix,
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
        if plan_steps and plan_expanded:
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
        plan_expanded: bool = False,
    ) -> int:
        base_bottom = 5 + max(0, subtitle_line_count - 1)
        if not plan_steps or not plan_expanded:
            return base_bottom
        return base_bottom + 1 + len(plan_steps)

    def _session_body_top(
        self,
        plan_steps: tuple[PlanStep, ...] = (),
        subtitle_line_count: int = 1,
        plan_expanded: bool = False,
    ) -> int:
        return self._header_bottom(plan_steps, subtitle_line_count, plan_expanded) + 2

    def _draw_header_title(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        title: str,
        plan_steps: tuple[PlanStep, ...],
        plan_expanded: bool,
        width: int,
        title_suffix: str = "",
    ) -> None:
        del plan_expanded
        title_text = self._header_title_text(title, plan_steps, width, title_suffix)
        self._add(stdscr, y, x, title_text, width, self._attr("bold"))
        if plan_steps and title_text:
            self._add_click_target(
                y,
                SessionMouseAction(
                    "toggle_plan",
                    0,
                    x_start=x,
                    x_end=x + min(len(title_text), width),
                ),
            )

    def _header_title_text(
        self,
        title: str,
        plan_steps: tuple[PlanStep, ...],
        width: int,
        title_suffix: str = "",
    ) -> str:
        text = title
        if plan_steps:
            text = f"{text} › {self._current_plan_step_title(plan_steps)}"
        suffix = title_suffix.strip()
        if suffix:
            text = f"{text} {suffix}"
        safe_width = max(1, width)
        if len(text) > safe_width:
            if safe_width <= 1:
                text = text[:safe_width]
            elif safe_width <= 4:
                text = f"{text[: safe_width - 1]}…"
            else:
                text = f"{text[: safe_width - 2].rstrip()} …"
        return text

    def _current_plan_step_title(self, plan_steps: tuple[PlanStep, ...]) -> str:
        for step in plan_steps:
            if not step.is_done:
                return step.title
        return plan_steps[-1].title if plan_steps else "Plan complete"

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
            marker = "›" if index == selected else "•"
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

    def _draw_continue_session_prompt(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        statement: str,
        selected: int,
    ) -> None:
        height, width = self._draw_shell(
            stdscr,
            "Continue session?",
            "Previous workspace session",
        )
        self._add(stdscr, 8, 4, str(self.workspace_root), width - 8, self._attr("bold"))
        y = 10
        self._add(stdscr, y, 4, session.title, width - 8, self._attr("bold"))
        y += 1
        detail = (
            f"{self._message_count_label(session.message_count)} · "
            f"{session.updated_at} · {session.provider}/{session.model}"
        )
        self._add(stdscr, y, 4, detail, width - 8, self._attr("light"))

        y += 3
        copy = statement.strip() or self._fallback_continue_session_statement(session)
        for line in textwrap.wrap(copy, width=max(24, width - 8)):
            self._add(stdscr, y, 4, line, width - 8)
            y += 1
        y += 1
        resume_copy = (
            "Anomx can resume the stored transcript and continue with the previous "
            "context, or start fresh in this workspace."
        )
        for line in textwrap.wrap(resume_copy, width=max(24, width - 8)):
            self._add(stdscr, y, 4, line, width - 8)
            y += 1

        y += 2
        choices = ("Yes, continue with session", "No, start new session")
        for index, choice in enumerate(choices):
            marker = "›" if index == selected else "•"
            attr = self._attr("accent") if index == selected else curses.A_NORMAL
            self._add(stdscr, y + index, 4, f"{marker} {index + 1}. {choice}", width - 8, attr)

        self._add(
            stdscr,
            min(height - 2, y + len(choices) + 2),
            4,
            "Enter to confirm · Esc to start new",
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

    def _draw_project(
        self,
        stdscr: CursesWindow,
        project: ProjectRecord,
        sessions: list[SessionRecord],
        selected: int,
        scroll: int,
        input_text: str,
        cursor: int,
        prompt_notice: str,
        prompt_notice_role: str,
        frame: int,
        command_suggestions: list[CommandSpec] | None = None,
        command_selected: int = 0,
        delete_pending_index: int | None = None,
        file_suggestions: list[MenuChoice] | None = None,
        file_selected: int = 0,
        file_references: Mapping[str, str] | None = None,
        file_reference_active: bool = False,
    ) -> int:
        config = self._load_config_cached()
        provider = str(config.get("provider", "openai"))
        model = str(config.get("model", "gpt-5.5"))
        self._click_targets = {}
        height, width = self._draw_shell(
            stdscr,
            project.name,
            str(project.path),
            header_meta=f"{provider}/{self._model_header_label(provider, model)}",
        )
        layout = self._prompt_layout(stdscr, input_text)
        body_top = self._session_body_top(subtitle_line_count=1)
        body_bottom = max(body_top + 1, layout.top_line)
        body_height = max(1, body_bottom - body_top)
        max_visible = max(1, body_height)
        if selected < scroll:
            scroll = selected
        elif selected >= scroll + max_visible:
            scroll = selected - max_visible + 1
        scroll = max(0, min(scroll, max(0, len(sessions) - 1)))

        y = body_top
        visible_sessions = sessions[scroll:]
        if not visible_sessions:
            self._add(
                stdscr,
                y,
                4,
                "No sessions yet.",
                width - 8,
                self._attr("light"),
            )
        for index, session in enumerate(visible_sessions, start=scroll):
            if y >= body_bottom:
                break
            y = self._draw_project_session_row(
                stdscr,
                y,
                width,
                session,
                selected=index == selected,
                index=index,
                body_bottom=body_bottom,
                delete_pending=delete_pending_index == index,
            )

        command_panel = (
            self._command_bottom_panel(command_suggestions or [], command_selected)
            if not file_suggestions
            else None
        )
        file_panel = self._file_reference_bottom_panel(
            file_suggestions or [],
            file_selected,
            active=file_reference_active,
        )
        active_panel = file_panel or command_panel
        if active_panel is not None:
            self._draw_bottom_panel(stdscr, active_panel, input_text)
        self._draw_prompt_bar(
            stdscr,
            input_text,
            cursor,
            prompt_notice,
            prompt_notice_role,
            self._prompt_reference_labels(file_references, None) if file_references else None,
            hint_suffix=self._project_prompt_hint_suffix(sessions, delete_pending_index),
        )
        stdscr.refresh()
        return scroll

    def _project_prompt_hint_suffix(
        self,
        sessions: Sequence[SessionRecord],
        delete_pending_index: int | None,
    ) -> str:
        if not sessions:
            return ""
        if delete_pending_index is not None:
            return " · Enter to confirm"
        return " · ctrl+d Delete"

    def _draw_project_session_row(
        self,
        stdscr: CursesWindow,
        y: int,
        width: int,
        session: SessionRecord,
        *,
        selected: bool,
        index: int,
        body_bottom: int,
        delete_pending: bool = False,
    ) -> int:
        del body_bottom
        running = self._session_is_running(session)
        needs_confirmation = self._project_session_needs_confirmation(session)
        quiet_attr = self._attr("light")
        active_attr = self._attr("accent")
        title_attr = (
            self._attr("selected") if selected else (active_attr if running else quiet_attr)
        )
        marker_attr = (
            self._attr("selected") if selected else (active_attr if running else quiet_attr)
        )
        right_text = self._project_session_right_text(
            session,
            running,
            selected=selected,
            delete_pending=delete_pending,
        )
        right_x = max(8, width - len(right_text) - 5) if right_text else width - 5
        title = self._project_session_title(session)
        if selected:
            self._add(stdscr, y, 4, "› ", 2, marker_attr)
            title_x = 6
        else:
            self._add(stdscr, y, 4, "•", 1, marker_attr)
            title_x = 6
        title_width = max(1, right_x - title_x)
        self._add(stdscr, y, title_x, title, title_width, title_attr)
        if running:
            statement = self._project_session_statement(session)
            if statement:
                statement_x = title_x + min(len(title), title_width)
                separator = " › "
                separator_width = max(0, min(len(separator), right_x - statement_x - 1))
                if separator_width > 0:
                    self._add(
                        stdscr,
                        y,
                        statement_x,
                        separator,
                        separator_width,
                        quiet_attr,
                    )
                    statement_x += separator_width
                if needs_confirmation and statement_x < right_x - 1:
                    badge_text = " Confirmation needed "
                    self._add(
                        stdscr,
                        y,
                        statement_x,
                        badge_text,
                        max(1, right_x - statement_x - 1),
                        self._attr("warning_badge"),
                    )
                elif statement_x < right_x - 1:
                    self._add(
                        stdscr,
                        y,
                        statement_x,
                        statement,
                        max(1, right_x - statement_x - 1),
                        quiet_attr,
                    )
        if right_text:
            if session.unread and not running and not delete_pending and selected:
                prefix = right_text.removesuffix("•")
                if prefix:
                    self._add(stdscr, y, right_x, prefix, len(prefix), self._attr("selected"))
                self._add(
                    stdscr,
                    y,
                    right_x + len(prefix),
                    "•",
                    1,
                    active_attr,
                )
            elif session.unread and not running and not delete_pending:
                right_attr = active_attr if not selected else self._attr("selected")
                self._add(stdscr, y, right_x, right_text, len(right_text), right_attr)
            else:
                right_attr = self._attr("selected") if selected else quiet_attr
                self._add(stdscr, y, right_x, right_text, len(right_text), right_attr)
        self._add_click_target(
            y,
            SessionMouseAction(
                "open_project_session",
                index,
                str(session.path),
                x_start=4,
                x_end=width - 4,
            ),
        )
        return y + 1

    def _project_session_title(
        self,
        session: SessionRecord,
    ) -> str:
        return session.title.strip() or "New session"

    def _project_session_right_text(
        self,
        session: SessionRecord,
        running: bool,
        *,
        selected: bool = False,
        delete_pending: bool = False,
    ) -> str:
        if delete_pending:
            return self._project_session_since_or_unread_text(session)
        if running:
            return self._project_session_running_duration(session) or "00:00"
        return self._project_session_since_or_unread_text(session)

    def _project_session_since_or_unread_text(self, session: SessionRecord) -> str:
        return "New •" if session.unread else self._project_session_since_text(session)

    def _project_session_running_duration(self, session: SessionRecord) -> str:
        turn = self._active_turn_for_session(session)
        if turn is not None:
            return "running " + self._format_duration(time.monotonic() - turn.started_at)
        events = self._session_events(session.path)

        processes = running_process_snapshots(events)
        if processes:
            return "running " + self._process_runtime_duration(processes[0].started_at)
        return ""

    def _project_session_since_text(self, session: SessionRecord) -> str:
        timestamp = session.updated_at.strip()
        if not timestamp:
            return ""
        with suppress(ValueError):
            updated_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            elapsed = max(0.0, (datetime.now(tz=UTC) - updated_at).total_seconds())
            if elapsed < 60:
                return "now ago"
            if elapsed < 3600:
                return f"{max(1, int(elapsed // 60))}min ago"
            if elapsed < 86400:
                return f"{max(1, int(elapsed // 3600))}h ago"
            return f"{max(1, int(elapsed // 86400))}d ago"
        return ""

    def _project_session_needs_confirmation(self, session: SessionRecord) -> bool:
        turn = self._active_turn_for_session(session)
        if turn is None:
            return False
        return any(event.kind == "approval" for event in turn.pending_events)

    def _project_session_statement(self, session: SessionRecord) -> str:
        turn = self._active_turn_for_session(session)
        events = self._session_events(session.path)
        statement = self._latest_project_session_statement(
            events,
            include_messages=turn is None,
        )
        if statement:
            return statement
        if turn is not None:
            if turn.final_text.strip():
                return self._ellipsized_statement_text(
                    self._single_line_work_text(turn.final_text),
                    120,
                )
            if turn.working_text and turn.working_text != "Thinking":
                return str(turn.working_text)
        statement = self._latest_project_session_statement(events, include_messages=True)
        if statement:
            return statement
        return ""

    def _latest_project_session_statement(
        self,
        events: Sequence[Mapping[str, object]],
        *,
        include_messages: bool = True,
    ) -> str:
        for event in reversed(events):
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            event_type = str(
                payload.get("type") if event.get("type") == "event_msg" else event.get("type")
            )
            if False:  # event_type == "worker_event" - removed
                pass
            if event_type == "process_event" and str(payload.get("status", "")) == "running":
                statement = str(payload.get("statement", "")).strip()
                if statement:
                    return self._ellipsized_statement_text(statement, 120)
            if include_messages and event_type in {"agent_message", "work_message"}:
                message = self._single_line_work_text(str(payload.get("message", "")))
                if message:
                    return self._ellipsized_statement_text(message, 120)
        return ""

    def _session_is_running(self, session: SessionRecord) -> bool:
        if self._active_turn_for_session(session) is not None:
            return True
        events = self._session_events(session.path)
        return bool(running_process_snapshots(events))

    def _session_mode_symbol(self, session: SessionRecord) -> str:
        turn = self._active_turn_for_session(session)
        if turn is not None:
            return turn.mode.symbol
        return AgentMode.parse(session.mode, self.agent_mode).symbol

    def _project_mouse_action(
        self,
        stdscr: CursesWindow,
        input_text: str,
        command_suggestions: Sequence[CommandSpec] | None = None,
        command_selected: int = 0,
        file_suggestions: Sequence[MenuChoice] | None = None,
        file_selected: int = 0,
    ) -> SessionMouseAction | None:
        with suppress(curses.error):
            _, x, y, _, button_state = curses.getmouse()
            if self._is_click_target_activation(button_state):
                action = self._click_target_action_at(x, y, button_state)
                if action is not None:
                    return action
            wheel_up = getattr(curses, "BUTTON4_PRESSED", 0)
            wheel_down = getattr(curses, "BUTTON5_PRESSED", 0)
            if wheel_up and button_state & wheel_up:
                return SessionMouseAction("scroll", -1)
            if wheel_down and button_state & wheel_down:
                return SessionMouseAction("scroll", 1)
            layout = self._prompt_layout(stdscr, input_text)
            clicked_prompt = layout.prompt_line <= y < layout.prompt_line + layout.prompt_height
            if clicked_prompt and self._is_left_click(button_state):
                view_start = self._prompt_view_start(input_text, len(input_text), layout)
                clicked_line = view_start + (y - layout.prompt_line)
                cursor = (clicked_line * layout.input_width) + (x - layout.input_x)
                cursor = max(0, min(len(input_text), cursor))
                return SessionMouseAction("cursor", cursor)
            if file_suggestions:
                file_panel = self._file_reference_bottom_panel(
                    list(file_suggestions),
                    file_selected,
                )
                if file_panel is not None and self._is_left_click(button_state):
                    choice = self._bottom_panel_mouse_choice_at(stdscr, file_panel, y, input_text)
                    if choice is not None:
                        return SessionMouseAction("file_reference", choice)
            else:
                command_panel = self._command_bottom_panel(
                    list(command_suggestions or ()),
                    command_selected,
                )
                if command_panel is not None and self._is_left_click(button_state):
                    choice = self._bottom_panel_mouse_choice_at(stdscr, command_panel, y, input_text)
                    if choice is not None:
                        return SessionMouseAction("command", choice)
        return None

    def _draw_back_to_project_link(self, stdscr: CursesWindow, width: int) -> None:
        text = "Back to Project"
        if width < len(text) + 8:
            return
        y = 3
        x = max(4, width - len(text) - 5)
        self._add(stdscr, y, x, text, len(text), self._attr("bold"))
        esc_text = "esc"
        esc_x = max(4, width - len(esc_text) - 5)
        self._add(stdscr, y + 1, esc_x, esc_text, len(esc_text), self._attr("light"))
        self._add_click_target(
            y,
            SessionMouseAction(
                "back_project",
                0,
                x_start=x,
                x_end=x + len(text),
            ),
        )

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
        image_attachments: Mapping[str, Mapping[str, str]] | None = None,
        bottom_panel: BottomPanel | None = None,
        working_text: str | None = None,
        working_deadline: float | None = None,
        working_frame: int = 0,
        anchor_line: int | None = None,
        prompt_notice: str = "",
        prompt_notice_role: str = "light",
        force_start_hints: bool = False,
        start_hint_reveal_progress: float | None = None,
        start_hint_removal_progress: float = 0.0,
        active_turn_elapsed: float | None = None,
        streaming_text: str = "",
    ) -> SessionViewportState:
        config = self._load_config_cached()
        provider = str(config.get("provider", session.provider))
        model = str(config.get("model", session.model))
        session_events = self._session_events(session.path)
        plan_steps = self._visible_plan_steps(
            session_events,
            latest_plan_steps(session_events),
        )
        plan_expanded = bool(plan_steps and session.path in self._expanded_plan_sessions)
        processes = running_process_snapshots(session_events) if bottom_panel is None else ()
        header_lines = self._session_header_lines(session, model)
        self._click_targets = {}
        height, width = self._draw_shell(
            stdscr,
            self._session_project_title(session),
            header_lines,
            plan_steps,
            header_meta=self._session_header_meta(session, provider, model),
            plan_expanded=plan_expanded,
            title_suffix=self._session_title_counter(active_turn_elapsed),
        )
        self._draw_back_to_project_link(stdscr, width)
        layout = self._prompt_layout(stdscr, input_text)
        suggestions = command_suggestions or []
        activity_items = self._activity_items(processes, session_events, working_frame)
        activity_panel_height = self._activity_panel_height(activity_items, width)
        body_top = self._session_body_top(
            plan_steps,
            subtitle_line_count=len(header_lines),
            plan_expanded=plan_expanded,
        )
        activity_panel_bottom = layout.prompt_line if activity_items else layout.top_line
        body_bottom = max(body_top + 1, activity_panel_bottom - activity_panel_height)
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
        display_messages = self._messages_with_transient_state(
            messages,
            active_turn_elapsed,
            streaming_text,
        )
        rendered = self._session_rendered_lines(
            session,
            display_messages,
            max(20, width - 8),
            None if streaming_text else self._working_status_text(working_text, working_deadline),
        )
        rendered_line_count = len(rendered)
        if anchor_line is None:
            scroll = self._clamp_session_scroll(scroll, rendered_line_count, body_height)
            start = self._session_view_start(scroll, rendered_line_count, body_height)
        else:
            start = max(0, min(anchor_line, self._session_max_start(rendered_line_count)))
            scroll = self._session_scroll_for_start(start, rendered_line_count, body_height)
        visible = rendered[start : start + body_height]
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
            if line.role in {"work_box", "work_box_danger"}:
                self._draw_work_box_line(stdscr, y, 4, line.text, width - 8, line.role)
                self._draw_session_selection(stdscr, y, 4, line_index, line.text, width - 8)
                continue
            if line.role in {"table_header", "table_row"}:
                self._draw_table_line(stdscr, y, 4, line.text, width - 8, line.role)
                self._draw_session_selection(stdscr, y, 4, line_index, line.text, width - 8)
                continue
            attr = self._line_attr(line.role)
            self._add(stdscr, y, 4, line.text, width - 8, attr)
            self._draw_session_selection(stdscr, y, 4, line_index, line.text, width - 8)

        should_draw_start_hints = self._should_draw_start_hints(
            messages,
            input_text,
            active_bottom_panel,
            working_text,
            plan_steps,
        )
        if force_start_hints or should_draw_start_hints:
            reveal_progress = (
                self._start_hint_reveal_progress()
                if start_hint_reveal_progress is None
                else start_hint_reveal_progress
            )
            self._draw_start_hints(
                stdscr,
                body_top,
                body_bottom,
                width,
                working_frame,
                reveal_progress,
                start_hint_removal_progress,
            )
        else:
            self._start_hint_reveal_started_at = None

        if activity_items:
            self._draw_activity_panel(
                stdscr,
                activity_items,
                body_bottom,
            )
        if active_bottom_panel is not None:
            self._draw_bottom_panel(stdscr, active_bottom_panel, input_text)
        self._draw_prompt_bar(
            stdscr,
            input_text,
            cursor,
            prompt_notice,
            prompt_notice_role,
            self._prompt_reference_labels(file_references, image_attachments),
            draw_top_rule=not activity_items,
        )
        stdscr.refresh()
        return SessionViewportState(start, scroll, body_height, rendered_line_count)

    def _session_title_counter(self, active_turn_elapsed: float | None) -> str:
        if active_turn_elapsed is None:
            return ""
        return self._format_duration(active_turn_elapsed)

    def _session_project_title(self, session: SessionRecord) -> str:
        project_name = self._current_project_name()
        title = session.title.strip() or "New session"
        return f"{project_name} › {title}" if project_name else title

    def _current_project_name(self) -> str:
        stored_project = self.home.project_for_path(self.project_path)
        if stored_project is not None and stored_project.name.strip():
            return stored_project.name.strip()
        return self._fallback_project_name()

    def _visible_plan_steps(
        self,
        events: Sequence[Mapping[str, object]],
        plan_steps: tuple[PlanStep, ...],
    ) -> tuple[PlanStep, ...]:
        if not plan_steps:
            return ()
        created_at = self._latest_plan_create_timestamp(events)
        if created_at is None:
            return plan_steps
        elapsed = max(0.0, (datetime.now(tz=UTC) - created_at).total_seconds())
        visible_count = min(
            len(plan_steps),
            max(1, int(elapsed // PLAN_STEP_REVEAL_SECONDS) + 1),
        )
        return plan_steps[:visible_count]

    def _latest_plan_create_timestamp(
        self,
        events: Sequence[Mapping[str, object]],
    ) -> datetime | None:
        created_at: datetime | None = None
        for event in events:
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            event_type = str(
                payload.get("type") if event.get("type") == "event_msg" else event.get("type")
            )
            if event_type != "plan_update":
                continue
            if str(payload.get("action", "")) != "create":
                created_at = None
                continue
            created_at = self._event_timestamp(event)
        return created_at

    def _event_timestamp(self, event: Mapping[str, object]) -> datetime | None:
        timestamp = str(event.get("timestamp", "")).strip()
        if not timestamp:
            return None
        with suppress(ValueError):
            return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return None

    def _start_hint_reveal_progress(self) -> float:
        now = time.monotonic()
        if self._start_hint_reveal_started_at is None:
            self._start_hint_reveal_started_at = now
        elapsed = now - self._start_hint_reveal_started_at
        return min(1.0, max(0.0, elapsed / START_HINT_REVEAL_SECONDS))

    def _start_hint_reveal_active(self) -> bool:
        if self._start_hint_reveal_started_at is None:
            return False
        return time.monotonic() - self._start_hint_reveal_started_at < START_HINT_REVEAL_SECONDS

    def _should_draw_start_hints(
        self,
        messages: list[MessageLine],
        input_text: str,
        active_bottom_panel: BottomPanel | None,
        working_text: str | None,
        plan_steps: tuple[PlanStep, ...],
    ) -> bool:
        del input_text
        return (
            not messages
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
        frame: int = 0,
        reveal_progress: float = 1.0,
        removal_progress: float = 0.0,
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
                    frame,
                    reveal_progress,
                    removal_progress,
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
                frame,
                reveal_progress,
                removal_progress,
            )

    def _draw_start_hint_card(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        width: int,
        height: int,
        skill: Skill,
        frame: int = 0,
        reveal_progress: float = 1.0,
        removal_progress: float = 0.0,
    ) -> None:
        inner_width = max(1, width - 4)
        horizontal = "─" * max(1, width - 2)
        self._draw_start_hint_line(
            stdscr,
            y,
            x,
            f"╭{horizontal}╮",
            width,
            self._attr("accent"),
            frame,
            1.0,
            removal_progress,
        )
        for offset in range(1, height - 1):
            self._draw_start_hint_line(
                stdscr,
                y + offset,
                x,
                "│",
                1,
                self._attr("accent"),
                frame,
                1.0,
                removal_progress,
            )
            self._draw_start_hint_line(
                stdscr,
                y + offset,
                x + width - 1,
                "│",
                1,
                self._attr("accent"),
                frame,
                1.0,
                removal_progress,
            )
        self._draw_start_hint_line(
            stdscr,
            y + height - 1,
            x,
            f"╰{horizontal}╯",
            width,
            self._attr("accent"),
            frame,
            1.0,
            removal_progress,
        )
        for offset in range(height):
            self._add_click_target(
                y + offset,
                SessionMouseAction("skill", 0, skill.command, x, x + width),
            )

        self._draw_start_hint_line(
            stdscr,
            y + 1,
            x + 2,
            skill.title,
            inner_width,
            self._attr("bold"),
            frame,
            reveal_progress,
            removal_progress,
        )
        description_lines = textwrap.wrap(
            skill.description,
            width=inner_width,
            break_long_words=True,
            break_on_hyphens=False,
        )[:3]
        for index, line in enumerate(description_lines):
            self._draw_start_hint_line(
                stdscr,
                y + 3 + index,
                x + 2,
                line,
                inner_width,
                self._attr("light"),
                frame,
                reveal_progress,
                removal_progress,
            )

    def _draw_start_hint_line(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        width: int,
        attr: int,
        frame: int,
        reveal_progress: float,
        removal_progress: float,
    ) -> None:
        reveal_progress = min(1.0, max(0.0, reveal_progress))
        removal_progress = min(1.0, max(0.0, removal_progress))
        if reveal_progress >= 1.0 and removal_progress <= 0.0:
            self._add(stdscr, y, x, text, width, attr)
            return
        char_rng = random.Random((frame + 1) * 262_147 + y * 97 + x * 31)
        for offset, character in enumerate(text[:width]):
            cell_x = x + offset
            if self._start_hint_cell_hidden(
                cell_x,
                y,
                reveal_progress,
                removal_progress,
            ):
                continue
            visible_character = (
                char_rng.choice(STARTUP_MATRIX_ALPHABET)
                if reveal_progress < 1.0 and character.strip()
                else character
            )
            self._add(stdscr, y, cell_x, visible_character, 1, attr)

    def _start_hint_cell_hidden(
        self,
        x: int,
        y: int,
        reveal_progress: float,
        removal_progress: float,
    ) -> bool:
        if removal_progress > 0:
            return self._startup_cell_removed(x, y, removal_progress)
        if reveal_progress >= 1:
            return False
        if reveal_progress <= 0:
            return True
        threshold = int(reveal_progress * 10_000)
        return self._startup_cell_rank(x, y, 0xC4AD5) >= threshold

    def _line_attr(self, role: str) -> int:
        if role == "user":
            return self._attr("accent")
        if role == "meta_accent":
            return self._attr("accent")
        if role in {"meta", "tool", "work_summary", "approved", "notice"}:
            return self._attr("light")
        if role == "warning":
            return self._attr("warning")
        if role == "work_box":
            return self._attr("work_box")
        if role == "work_box_danger":
            return self._attr("danger")
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
        cache_key = self._session_cache_key(session.path)
        expanded_turns = self._expanded_work_turns_key()
        expanded_lines = self._expanded_work_lines_key()
        working_key = "" if working_text is None else working_text
        message_cache = self._message_line_cache.get(session.path)
        if (
            cache_key is not None
            and message_cache is not None
            and message_cache[0] == cache_key[0]
            and message_cache[1] == cache_key[1]
            and message_cache[2] == expanded_turns
            and message_cache[3] is messages
        ):
            rendered_cache = self._rendered_message_cache.get(session.path)
            if (
                rendered_cache is not None
                and rendered_cache[0] == cache_key[0]
                and rendered_cache[1] == cache_key[1]
                and rendered_cache[2] == expanded_turns
                and rendered_cache[3] == expanded_lines
                and rendered_cache[4] == width
                and rendered_cache[5] == working_key
            ):
                return rendered_cache[6]
            rendered_messages = self._messages_with_working_status(messages, working_text)
            rendered = self._render_messages(rendered_messages, width)
            self._rendered_message_cache[session.path] = (
                cache_key[0],
                cache_key[1],
                expanded_turns,
                expanded_lines,
                width,
                working_key,
                rendered,
            )
            return rendered

        rendered_messages = self._messages_with_working_status(messages, working_text)
        return self._render_messages(rendered_messages, width)

    def _messages_with_working_status(
        self,
        messages: list[MessageLine],
        working_text: str | None,
    ) -> list[MessageLine]:
        if working_text is None:
            return messages
        return [*messages, MessageLine("working", working_text)]

    def _messages_with_transient_state(
        self,
        messages: list[MessageLine],
        active_turn_elapsed: float | None,
        streaming_text: str,
    ) -> list[MessageLine]:
        del active_turn_elapsed
        if not streaming_text:
            return messages
        rendered = list(messages)
        if streaming_text:
            rendered.append(MessageLine("agent", streaming_text))
        return rendered

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
        if working_deadline is None or self._is_waiting_status_text(working_text):
            return working_text
        current_time = time.monotonic() if now is None else now
        remaining = max(0, math.ceil(working_deadline - current_time))
        return f"{working_text} {remaining // 60:02d}:{remaining % 60:02d}"

    def _session_header_lines(
        self,
        session: SessionRecord,
        model: str,
    ) -> tuple[str, ...]:
        del model
        return (self._session_location_line(session),)

    def _session_location_line(self, session: SessionRecord) -> str:
        return str(session.cwd or self.cwd)

    def _session_header_meta(self, session: SessionRecord, provider: str, model: str) -> str:
        model_label = self._model_header_label(provider, model)
        parts = [session.session_id[:8], f"{provider}/{model_label}"]
        context_status = self._context_status(session, model)
        if context_status:
            parts.append(context_status)
        return " · ".join(parts)

    def _model_header_label(self, provider: str, model: str) -> str:
        marker = self._thinking_intensity_marker(provider, model)
        return f"{model} {marker}" if marker else model

    def _thinking_intensity_marker(self, provider: str, model: str) -> str:
        config = self._load_config_cached()
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
        if context_window is None:
            return ""
        cache_key = self._session_cache_key(session.path)
        if cache_key is not None:
            cached = self._context_status_cache.get(session.path)
            if (
                cached is not None
                and cached[0] == cache_key[0]
                and cached[1] == cache_key[1]
                and cached[2] == model
            ):
                return cached[3]

        if not self._has_user_messages(session.path):
            status = ""
        else:
            used_tokens = self.runtime.estimate_session_context_tokens(session.path)
            percent_used = context_usage_percent(used_tokens, context_window)
            status = f"{percent_used}% Context"

        if cache_key is not None:
            self._context_status_cache[session.path] = (
                cache_key[0],
                cache_key[1],
                model,
                status,
            )
        return status

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
        if self._is_waiting_status_text(text):
            dots = "." * (((frame // 4) % 3) + 1)
        else:
            dots = "." * ((frame // 4) % 4)
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
        role: str = "work_box",
    ) -> None:
        self._add(stdscr, y, x, text.ljust(max(0, width)), width, self._line_attr(role))

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

    def _activity_items(
        self,
        processes: tuple[AsyncProcessSnapshot, ...],
        events: Sequence[Mapping[str, object]],
        frame: int,
    ) -> tuple[ActivityItem, ...]:
        items: list[ActivityItem] = []
        items.extend(self._process_activity_item(process, frame) for process in processes)
        return tuple(items)

    
    def _process_activity_item(
        self,
        process: AsyncProcessSnapshot,
        frame: int,
    ) -> ActivityItem:
        active = process.status == "running"
        title = self._process_activity_title(process)
        if active:
            title = f"{title}{self._activity_dots(frame)}"
        return ActivityItem(
            key=f"process:{process.process_id}",
            title=title,
            right_text=self._process_right_text(process),
            details=self._process_activity_details(process),
            active=active,
            kill_process_id=process.process_id if active else "",
            marker=self._activity_marker(active, frame),
        )

    def _activity_panel_height(
        self,
        items: tuple[ActivityItem, ...],
        width: int,
    ) -> int:
        if not items:
            return 0
        height = 1
        for item in items:
            height += 2
            if item.key in self._expanded_activity_items:
                height += self._activity_detail_view_height(item, width)
        return height

    def _draw_activity_panel(
        self,
        stdscr: CursesWindow,
        items: tuple[ActivityItem, ...],
        start_y: int,
    ) -> None:
        _, width = stdscr.getmaxyx()
        panel_width = max(1, width - 4)
        separator = "─" * panel_width
        y = start_y
        self._add(stdscr, y, 2, separator, panel_width, self._attr("light"))
        for item in items:
            y += 1
            expanded = item.key in self._expanded_activity_items
            self._draw_activity_title_row(stdscr, y, item, width, expanded)
            self._add_click_target(y, SessionMouseAction("toggle_activity_item", 0, item.key))
            self._add_click_target(y, SessionMouseAction("scroll_activity_item", 0, item.key))
            y += 1
            if expanded:
                for row in self._visible_activity_detail_rows(item, width):
                    self._draw_activity_detail_row(stdscr, y, row, width)
                    self._add_click_target(
                        y,
                        SessionMouseAction("scroll_activity_item", 0, item.key),
                    )
                    if row.entry_key:
                        self._add_click_target(
                            y,
                            SessionMouseAction(
                                "toggle_activity_entry",
                                0,
                                row.entry_key,
                            ),
                        )
                    y += 1
            self._add(stdscr, y, 2, separator, panel_width, self._attr("light"))

    def _draw_activity_title_row(
        self,
        stdscr: CursesWindow,
        y: int,
        item: ActivityItem,
        width: int,
        expanded: bool,
    ) -> None:
        bullet_attr = self._attr("light")
        title_attr = self._attr("bold") if expanded else self._attr("light")
        right_attr = self._attr("bold") if expanded else self._attr("light")
        self._add(stdscr, y, 4, item.marker, 1, bullet_attr)
        right_text = self._activity_title_right_text(item, expanded)
        right_x = max(8, width - len(right_text) - 4) if right_text else width
        title_width = max(1, right_x - 7)
        self._add(stdscr, y, 6, item.title, title_width, title_attr)
        if right_text:
            self._add(stdscr, y, right_x, right_text, len(right_text), right_attr)

    def _activity_title_right_text(self, item: ActivityItem, expanded: bool) -> str:
        action = "Collapse" if expanded else "Expand"
        if item.right_text:
            return f"{action} · {item.right_text}"
        return action

    def _activity_marker(self, active: bool, frame: int) -> str:
        del frame
        if not active:
            return "⏸"
        return "▶"

    def _draw_activity_detail_row(
        self,
        stdscr: CursesWindow,
        y: int,
        row: ActivityDetailRow,
        width: int,
    ) -> None:
        detail_width = max(1, width - 10)
        role = "work_box" if row.role == "work_box" else "activity_detail"
        attr = self._attr("work_box") if role == "work_box" else self._attr("light")
        self._add(stdscr, y, 6, row.text, detail_width, attr)

    def _activity_detail_view_height(self, item: ActivityItem, width: int) -> int:
        return min(
            ACTIVITY_DETAIL_MAX_LINES,
            len(self._activity_detail_rows(item, self._activity_detail_width(width))),
        )

    def _visible_activity_detail_rows(
        self,
        item: ActivityItem,
        width: int,
    ) -> tuple[ActivityDetailRow, ...]:
        detail_width = self._activity_detail_width(width)
        rows = self._activity_detail_rows(item, detail_width)
        if not rows:
            return ()
        max_scroll = max(0, len(rows) - ACTIVITY_DETAIL_MAX_LINES)
        scroll = max(0, min(self._activity_detail_scrolls.get(item.key, 0), max_scroll))
        if scroll != self._activity_detail_scrolls.get(item.key, 0):
            self._activity_detail_scrolls[item.key] = scroll
        return rows[scroll : scroll + ACTIVITY_DETAIL_MAX_LINES]

    def _activity_detail_rows(
        self,
        item: ActivityItem,
        width: int,
    ) -> tuple[ActivityDetailRow, ...]:
        entries = item.details or (
            ActivityDetailEntry(f"{item.key}:empty", "No logs yet.", "No logs yet."),
        )
        rows: list[ActivityDetailRow] = []
        for entry in entries:
            rows.append(ActivityDetailRow(entry.text, entry_key=entry.key))
            if entry.key in self._expanded_activity_entries:
                rows.extend(self._activity_entry_box_rows(entry, width))
        return tuple(rows)

    def _activity_entry_box_rows(
        self,
        entry: ActivityDetailEntry,
        width: int,
    ) -> tuple[ActivityDetailRow, ...]:
        safe_width = max(20, width)
        inner_width = max(1, safe_width - 4)
        border = "─" * max(1, safe_width - 2)
        rows = [ActivityDetailRow(f"╭{border}╮", "work_box", entry.key)]
        body = entry.detail_body.strip() or entry.text
        for line in self._work_box_content_lines(body, inner_width):
            content = line[:inner_width].ljust(inner_width)
            rows.append(ActivityDetailRow(f"│ {content} │", "work_box", entry.key))
        rows.append(ActivityDetailRow(f"╰{border}╯", "work_box", entry.key))
        return tuple(rows)

    def _activity_detail_width(self, width: int) -> int:
        return max(1, width - 10)

    def _activity_dots(self, frame: int) -> str:
        dots = "." * ((frame // 4) % 4)
        return dots

    
    def _activity_command_display_text(self, statement: str, command: str) -> str:
        del command
        return self._single_line_work_text(statement) or "Command"

    def _activity_command_detail_body(self, command: str, output: str) -> str:
        del output
        return command.strip()

    
    def _process_activity_title(self, process: AsyncProcessSnapshot) -> str:
        label = process.statement.strip() or "Running command"
        if False:  # process.source == "worker_command"
            owner = process.owner_name or process.owner_id or "Process"
            return f"{owner} › Command {label}"
        if process.source == "command":
            return f"Command {label}"
        return f"Process {label}"

    def _short_command_label(self, command: str) -> str:
        return " ".join(command.split()) or "command"

    def _process_activity_details(
        self,
        process: AsyncProcessSnapshot,
    ) -> tuple[ActivityDetailEntry, ...]:
        output = process.output.strip()
        lines = output.splitlines() if output else ["No logs yet."]
        entries: list[ActivityDetailEntry] = []
        for line in lines:
            text = line.rstrip()
            if not text:
                continue
            entries.append(
                ActivityDetailEntry(
                    self._activity_entry_key(process.process_id, "log", len(entries), text),
                    text,
                    text,
                )
            )
        return tuple(entries)

    def _activity_entry_key(
        self,
        owner_id: str,
        kind: str,
        index: int,
        text: str,
    ) -> str:
        digest = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:12]
        return f"activity:{owner_id}:{kind}:{index}:{digest}"

    def _process_right_text(self, process: AsyncProcessSnapshot) -> str:
        if process.status == "running":
            return self._process_runtime_duration(process.started_at)
        return self._process_state_label(process)

    def _process_runtime_duration(self, started_at: str) -> str:
        if not started_at:
            return ""
        with suppress(ValueError):
            started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            seconds = max(0, int((datetime.now(tz=UTC) - started).total_seconds()))
            return self._format_duration(seconds)
        return ""

    def _process_state_label(self, process: AsyncProcessSnapshot) -> str:
        status = process.status.strip().lower()
        if status in {"ready", "complete", "completed", "done"}:
            return "Ready"
        if status == "ended":
            if process.exit_code is not None and process.exit_code < 0:
                return "Interrupted"
            if process.exit_code not in (None, 0):
                return "Failed"
            return "Ready"
        if status in {"interrupted", "killed", "stopped", "cancelled", "canceled"}:
            return "Interrupted"
        if status == "failed" or process.exit_code not in (None, 0):
            return "Failed"
        return status.title() if status else "Ready"

    def _add_click_target(self, y: int, action: SessionMouseAction) -> None:
        self._click_targets.setdefault(y, []).append(action)

    def _worker_left_text(self, worker: object, frame: int) -> str:
        return ""

    def _worker_right_text(self, worker: object) -> str:
        return ""

    def _worker_context_text(self, worker: object) -> str:
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
        active: bool = False,
    ) -> BottomPanel | None:
        if not suggestions:
            if not active:
                return None
            return BottomPanel(
                "Files",
                "No matches found",
                tuple(),
                0,
            )
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
            marker = "›" if choice_index == panel.selected else "•"
            attr = self._attr("accent") if choice_index == panel.selected else curses.A_NORMAL
            self._draw_bottom_panel_choice_label(
                stdscr,
                choice_y + row_offset,
                4,
                f"{marker} {choice.label}",
                max(1, detail_x - 6),
                attr,
                choice.highlight,
                choice_index == panel.selected,
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

    def _draw_bottom_panel_choice_label(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        width: int,
        attr: int,
        highlight: str = "",
        selected: bool = False,
    ) -> None:
        query = highlight.strip().lower()
        if not query:
            self._add(stdscr, y, x, text, width, attr)
            return
        visible = text[: max(0, width)]
        search = visible.lower()
        start = search.find(query)
        if start < 0:
            self._add(stdscr, y, x, visible, width, attr)
            return
        end = min(len(visible), start + len(query))
        if start > 0:
            self._add(stdscr, y, x, visible[:start], width, attr)
        highlight_attr = self._attr("selected") if selected else self._attr("accent")
        self._add(stdscr, y, x + start, visible[start:end], width - start, highlight_attr)
        if end < len(visible):
            self._add(stdscr, y, x + end, visible[end:], width - end, attr)

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
            return self._bottom_panel_mouse_choice_at(stdscr, panel, y, input_text)
        return None

    def _bottom_panel_mouse_choice_at(
        self,
        stdscr: CursesWindow,
        panel: BottomPanel,
        y: int,
        input_text: str = "",
    ) -> int | None:
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
        *,
        draw_top_rule: bool = True,
        hint_suffix: str = "",
    ) -> None:
        layout = self._prompt_layout(stdscr, input_text)
        _, width = stdscr.getmaxyx()
        panel_width = max(1, width - 4)
        clear_top = layout.top_line if draw_top_rule else layout.prompt_line
        for y in range(clear_top, layout.hint_line + 1):
            self._clear_row(stdscr, y)
        if draw_top_rule:
            self._add(
                stdscr,
                layout.top_line,
                2,
                "─" * panel_width,
                panel_width,
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
            self._add(stdscr, y, 4, marker, 1, self._attr("accent"))
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
        self._draw_prompt_cursor_cell(
            stdscr,
            layout,
            input_text,
            cursor,
            visible_lines,
            view_start,
        )
        self._add(
            stdscr,
            layout.bottom_line,
            2,
            "─" * panel_width,
            panel_width,
            self._attr("light"),
        )
        show_notice = bool(notice and notice != RUNNING_NOTICE)
        hint_text = notice if show_notice else self.agent_mode.prompt_hint
        hint_attr = notice_role if show_notice else self._mode_hint_attr_name()
        hint_width = layout.input_width + 2
        self._add(
            stdscr,
            layout.hint_line,
            4,
            hint_text,
            hint_width,
            self._attr(hint_attr),
        )
        if not show_notice and hint_suffix:
            suffix_x = 4 + min(len(hint_text), hint_width)
            suffix_width = max(0, hint_width - min(len(hint_text), hint_width))
            if suffix_width:
                self._add(
                    stdscr,
                    layout.hint_line,
                    suffix_x,
                    hint_suffix,
                    suffix_width,
                    self._attr("light"),
                )
        with suppress(curses.error):
            curses.curs_set(0)

    def _draw_prompt_cursor_cell(
        self,
        stdscr: CursesWindow,
        layout: PromptLayout,
        input_text: str,
        cursor: int,
        visible_lines: list[str],
        view_start: int,
    ) -> None:
        cursor_line, cursor_column = self._prompt_cursor_position(
            input_text,
            cursor,
            layout.input_width,
        )
        visible_cursor_line = cursor_line - view_start
        if not 0 <= visible_cursor_line < layout.prompt_height:
            return
        cursor_y = layout.prompt_line + visible_cursor_line
        cursor_x = layout.input_x + cursor_column
        line = (
            visible_lines[visible_cursor_line]
            if visible_cursor_line < len(visible_lines)
            else ""
        )
        character = line[cursor_column] if cursor_column < len(line) and input_text else " "
        self._add(stdscr, cursor_y, cursor_x, character, 1, self._attr("cursor"))

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
        input_width = max(1, width - 10)
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
            input_x=6,
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
            return self._session_mouse_action_from_event(
                stdscr,
                input_text,
                command_suggestions,
                command_selected,
                file_suggestions,
                file_selected,
                x,
                y,
                button_state,
            )
        return None

    def _raw_mouse_action(
        self,
        key: str | int,
        stdscr: CursesWindow,
        input_text: str,
        command_suggestions: list[CommandSpec],
        command_selected: int = 0,
        file_suggestions: list[MenuChoice] | None = None,
        file_selected: int = 0,
    ) -> SessionMouseAction | None:
        event = self._raw_mouse_event(key)
        if event is None:
            return None
        x, y, button_state = event
        return self._session_mouse_action_from_event(
            stdscr,
            input_text,
            command_suggestions,
            command_selected,
            file_suggestions,
            file_selected,
            x,
            y,
            button_state,
        )

    def _raw_mouse_event(self, key: str | int) -> tuple[int, int, int] | None:
        if not isinstance(key, str):
            return None
        match = RAW_MOUSE_RE.match(key)
        if match is None:
            return None
        button_code = int(match.group("button"))
        x = max(0, int(match.group("x")) - 1)
        y = max(0, int(match.group("y")) - 1)
        button_state = self._raw_mouse_button_state(button_code, match.group("suffix"))
        return x, y, button_state

    def _raw_mouse_button_state(self, button_code: int, suffix: str) -> int:
        if button_code == 64:
            return getattr(curses, "BUTTON4_PRESSED", 0)
        if button_code == 65:
            return getattr(curses, "BUTTON5_PRESSED", 0)
        if suffix == "m":
            return getattr(curses, "BUTTON1_RELEASED", 0)
        state = getattr(curses, "BUTTON1_PRESSED", 0)
        if button_code & 32:
            state |= getattr(curses, "REPORT_MOUSE_POSITION", 0)
        return state

    def _session_mouse_action_from_event(
        self,
        stdscr: CursesWindow,
        input_text: str,
        command_suggestions: list[CommandSpec],
        command_selected: int,
        file_suggestions: list[MenuChoice] | None,
        file_selected: int,
        x: int,
        y: int,
        button_state: int,
    ) -> SessionMouseAction | None:
        wheel_up = getattr(curses, "BUTTON4_PRESSED", 0)
        wheel_down = getattr(curses, "BUTTON5_PRESSED", 0)
        if wheel_up and button_state & wheel_up:
            activity_action = self._activity_scroll_action_at(y, 1)
            if activity_action is not None:
                return activity_action
            return SessionMouseAction("scroll", 1)
        if wheel_down and button_state & wheel_down:
            activity_action = self._activity_scroll_action_at(y, -1)
            if activity_action is not None:
                return activity_action
            return SessionMouseAction("scroll", -1)

        selection_action = self._session_selection_mouse_action(x, y, button_state)
        if selection_action is not None:
            return selection_action

        click_action = self._click_target_action_at(x, y, button_state)
        if click_action is not None:
            return click_action

        active_file_suggestions = file_suggestions or []
        if active_file_suggestions and self._is_left_click(button_state):
            panel = self._file_reference_bottom_panel(
                active_file_suggestions,
                file_selected,
            )
            if panel is not None:
                index = self._bottom_panel_mouse_choice_at(
                    stdscr,
                    panel,
                    y,
                    input_text,
                )
                if index is not None:
                    return SessionMouseAction("file_reference", index)

        if command_suggestions and self._is_left_click(button_state):
            panel = self._command_bottom_panel(
                command_suggestions,
                selected=command_selected,
            )
            if panel is not None:
                index = self._bottom_panel_mouse_choice_at(stdscr, panel, y, input_text)
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

    def _activity_scroll_action_at(
        self,
        y: int,
        delta: int,
    ) -> SessionMouseAction | None:
        for action in self._click_targets.get(y, ()):
            if action.kind == "scroll_activity_item":
                return SessionMouseAction("scroll_activity_item", delta, action.text)
        return None

    def _click_target_action_at(
        self,
        x: int,
        y: int,
        button_state: int,
    ) -> SessionMouseAction | None:
        if not self._is_click_target_activation(button_state):
            return None
        for action in reversed(self._click_targets.get(y, ())):
            if action.kind == "scroll_activity_item":
                continue
            if not action.x_end or action.x_start <= x < action.x_end:
                return action
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
            self._copy_to_clipboard(selected_text)
            return SessionMouseAction("selection", len(selected_text))

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

    def _is_click_target_activation(self, button_state: int) -> bool:
        clicked = getattr(curses, "BUTTON1_CLICKED", 0)
        pressed = getattr(curses, "BUTTON1_PRESSED", 0)
        return bool(
            (clicked and button_state & clicked)
            or (pressed and button_state & pressed)
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

    def _start_session_turn(
        self,
        session: SessionRecord,
        runtime: AgentRuntime | None = None,
    ) -> ActiveSessionTurn:
        existing = self._active_turn_for_session(session)
        if existing is not None:
            return existing

        events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
        result: dict[str, str] = {}
        turn_mode = AgentMode.parse(session.mode, self.agent_mode)
        turn_runtime = runtime or self._new_session_runtime(turn_mode)
        turn_runtime.set_mode(turn_mode)
        turn_id = uuid4().hex
        started_at = time.monotonic()

        def status_callback(message: str) -> None:
            events.put(RuntimeUiEvent("status", message))

        def message_callback(message: str) -> None:
            events.put(RuntimeUiEvent("message", message))

        def tool_message_callback(message: str) -> None:
            events.put(RuntimeUiEvent("tool_message", message))

        def command_callback(statement: str, command: str, _output: str) -> None:
            events.put(RuntimeUiEvent("command", statement, command=command))

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
            try:
                result["response"] = turn_runtime.backend_response(
                    session.path,
                    callbacks=RuntimeCallbacks(
                        status=status_callback,
                        message=message_callback,
                        tool_message=tool_message_callback,
                        command=command_callback,
                        delta=delta_callback,
                        approval=approval_callback,
                        system_message=system_message_callback,
                        question=question_callback,
                    ),
                )
            except Exception as error:  # pragma: no cover - defensive thread boundary
                crash_path: Path | None = None
                with suppress(OSError, TypeError, ValueError):
                    crash_path = self.home.write_crash_log(
                        error,
                        context={
                            "session_path": str(session.path),
                            "workspace_root": str(self.workspace_root),
                        },
                    )
                message = f"Agent crashed: {type(error).__name__}: {error}"
                if crash_path is not None:
                    message = f"{message}\nCrash log: {crash_path}"
                result["response"] = message

        worker = threading.Thread(target=run_backend, daemon=True)
        turn = ActiveSessionTurn(
            session=session,
            runtime=turn_runtime,
            events=events,
            result=result,
            turn_id=turn_id,
            started_at=started_at,
            worker=worker,
            mode=turn_mode,
        )
        self._active_session_turns[self._session_turn_key(session)] = turn
        worker.start()
        return turn

    def _new_session_runtime(self, mode: AgentMode | None = None) -> AgentRuntime:
        return AgentRuntime(
            self.home,
            self.cwd,
            self.session_allowed_commands,
            self.session_rejected_commands,
            self.agent_mode if mode is None else mode,
            workspace_root=self.workspace_root,
        )

    def _foreground_session_runtime(self, mode: AgentMode | None = None) -> AgentRuntime:
        for turn in self._active_session_turns.values():
            if turn.runtime is self.runtime and turn.worker is not None and turn.worker.is_alive():
                return self._new_session_runtime(mode)
        if mode is not None:
            self.runtime.set_mode(mode)
        return self.runtime

    def _active_turn_for_session(self, session: SessionRecord) -> ActiveSessionTurn | None:
        turn = self._active_session_turns.get(self._session_turn_key(session))
        if turn is None or turn.completed:
            return None
        return turn

    def _runtime_for_session(self, session: SessionRecord) -> AgentRuntime:
        turn = self._active_turn_for_session(session)
        return turn.runtime if turn is not None else self.runtime

    def _session_turn_key(self, session: SessionRecord) -> Path:
        return session.path.expanduser().resolve()

    def _drain_active_session_turns(self, stdscr: CursesWindow) -> None:
        for turn in list(self._active_session_turns.values()):
            self._drain_session_turn_events(
                stdscr,
                turn,
                render_events=False,
            )
            if turn.worker is not None and not turn.worker.is_alive():
                response = self._complete_session_turn(
                    stdscr,
                    turn,
                    render_final=False,
                )
                if response:
                    self.home.set_session_unread(turn.session.path, True)

    def _drain_session_turn_events(
        self,
        stdscr: CursesWindow,
        turn: ActiveSessionTurn,
        *,
        anchor_line: int | None = None,
        input_text: str = "",
        cursor: int = 0,
        prompt_notice: str = "",
        prompt_notice_role: str = "light",
        scroll: int = 0,
        active_turn_elapsed: float | None = None,
        render_events: bool = True,
    ) -> None:
        (
            turn.working_text,
            turn.working_deadline,
            turn.final_text,
            turn.work_count,
        ) = self._process_runtime_events(
            stdscr,
            turn.session,
            turn.events,
            turn.working_text,
            turn.working_deadline,
            turn.final_text,
            turn.turn_id,
            turn.work_count,
            anchor_line,
            input_text,
            cursor,
            prompt_notice,
            prompt_notice_role,
            scroll,
            active_turn_elapsed=active_turn_elapsed,
            render_events=render_events,
            deferred_events=turn.pending_events,
            defer_interactive=not render_events,
        )

    def _complete_session_turn(
        self,
        stdscr: CursesWindow,
        turn: ActiveSessionTurn,
        *,
        anchor_line: int | None = None,
        scroll: int = 0,
        input_text: str = "",
        cursor: int = 0,
        render_final: bool = True,
    ) -> str:
        if turn.completed:
            return ""
        turn.worker.join(timeout=0)
        self._drain_session_turn_events(
            stdscr,
            turn,
            anchor_line=anchor_line,
            input_text=input_text,
            cursor=cursor,
            scroll=scroll,
            active_turn_elapsed=time.monotonic() - turn.started_at,
            render_events=render_final,
        )
        response = turn.final_text or turn.result.get("response", "")
        if not turn.work_summary_appended:
            turn.work_summary_appended = self._append_work_summary(
                turn.session,
                turn.turn_id,
                turn.started_at,
                turn.work_count,
            )
        if response:
            if render_final and (not turn.final_text or not turn.streamed_final_displayed):
                self._fake_type_message(
                    stdscr,
                    turn.session,
                    response,
                    anchor_line=anchor_line,
                    scroll=scroll,
                    input_text=input_text,
                    cursor=cursor,
                    active_turn_elapsed=time.monotonic() - turn.started_at,
                )
            self.home.append_session_event(
                turn.session.path,
                "agent_message",
                {"message": response},
            )
        turn.completed = True
        self._active_session_turns.pop(self._session_turn_key(turn.session), None)
        return response

    def _shutdown_active_session_turns(self) -> None:
        for turn in list(self._active_session_turns.values()):
            turn.runtime.shutdown(turn.session.path)
            turn.completed = True
        self._active_session_turns.clear()

    def _run_backend_turn(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        anchor_line: int | None = None,
    ) -> BackendTurnResult:
        turn_mode = AgentMode.parse(session.mode, self.agent_mode)
        self._activate_agent_mode(turn_mode)
        turn = self._start_session_turn(
            replace(session, mode=turn_mode),
            runtime=self._foreground_session_runtime(turn_mode),
        )
        input_text = ""
        cursor = 0
        prompt_notice = RUNNING_NOTICE
        prompt_notice_role = "light"
        abort_key = ""
        abort_deadline = 0.0
        frame = 0
        running_anchor = anchor_line
        running_scroll = 0
        command_selected = 0
        with suppress(curses.error):
            stdscr.nodelay(True)
        try:
            while turn.worker is not None and turn.worker.is_alive():
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
                        self._drain_session_turn_events(
                            stdscr,
                            turn,
                            anchor_line=running_anchor,
                            input_text=input_text,
                            cursor=cursor,
                            prompt_notice=prompt_notice,
                            prompt_notice_role=prompt_notice_role,
                            scroll=running_scroll,
                            active_turn_elapsed=time.monotonic() - turn.started_at,
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
                            turn.runtime.abort_current_turn(session.path)
                            self._append_work_summary(
                                session,
                                turn.turn_id,
                                turn.started_at,
                                turn.work_count,
                            )
                            return BackendTurnResult(
                                input_text,
                                cursor,
                                exit_requested=True,
                                anchor_line=running_anchor,
                                scroll=running_scroll,
                            )
                    if key_result.exit_requested:
                        turn.runtime.abort_current_turn(session.path)
                        self._drain_session_turn_events(
                            stdscr,
                            turn,
                            anchor_line=running_anchor,
                            input_text=input_text,
                            cursor=cursor,
                            prompt_notice=prompt_notice,
                            prompt_notice_role=prompt_notice_role,
                            scroll=running_scroll,
                            active_turn_elapsed=time.monotonic() - turn.started_at,
                        )
                        self._append_work_summary(
                            session,
                            turn.turn_id,
                            turn.started_at,
                            turn.work_count,
                        )
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
                    if key_result.back_requested:
                        self._drain_session_turn_events(
                            stdscr,
                            turn,
                            anchor_line=running_anchor,
                            input_text=input_text,
                            cursor=cursor,
                            prompt_notice=prompt_notice,
                            prompt_notice_role=prompt_notice_role,
                            scroll=running_scroll,
                            active_turn_elapsed=time.monotonic() - turn.started_at,
                        )
                        return BackendTurnResult(
                            input_text,
                            cursor,
                            anchor_line=running_anchor,
                            scroll=running_scroll,
                            back_requested=True,
                        )
                elif abort_key and time.monotonic() > abort_deadline:
                    abort_key = ""
                    prompt_notice = RUNNING_NOTICE
                    prompt_notice_role = "light"

                self._drain_session_turn_events(
                    stdscr,
                    turn,
                    anchor_line=running_anchor,
                    input_text=input_text,
                    cursor=cursor,
                    prompt_notice=prompt_notice,
                    prompt_notice_role=prompt_notice_role,
                    scroll=running_scroll,
                    active_turn_elapsed=time.monotonic() - turn.started_at,
                )
                viewport = self._draw_session(
                    stdscr,
                    session,
                    self._read_message_lines(session.path),
                    input_text,
                    cursor,
                    running_scroll,
                    working_text=turn.working_text,
                    working_deadline=turn.working_deadline,
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
                    active_turn_elapsed=time.monotonic() - turn.started_at,
                    streaming_text=turn.final_text,
                )
                if turn.final_text:
                    turn.streamed_final_displayed = True
                if viewport is not None:
                    running_scroll = viewport.scroll
                time.sleep(0.08)
                frame += 1
        finally:
            with suppress(curses.error):
                stdscr.nodelay(False)
        self._complete_session_turn(
            stdscr,
            turn,
            anchor_line=running_anchor,
            scroll=running_scroll,
            input_text=input_text,
            cursor=cursor,
            render_final=True,
        )
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
    ) -> bool:
        if not work_count:
            return False
        duration = self._format_duration(time.monotonic() - started_at)
        self.home.append_session_event(
            session.path,
            "work_summary",
            {
                "turn_id": turn_id,
                "message": f"Worked for {duration} min",
            },
        )
        return True

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
        while time.monotonic() < deadline and len(suffix) < 24:
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
        if RAW_MOUSE_SUFFIX_RE.match(suffix):
            return True
        if suffix in {"b", "B", "f", "F"}:
            return True
        return suffix.endswith(("C", "D", "~", "u", "Z"))

    def _is_raw_mouse_fragment_key(self, key: str | int) -> bool:
        if not isinstance(key, str):
            return False
        return bool(
            RAW_MOUSE_RE.match(key)
            or RAW_MOUSE_SUFFIX_RE.match(key)
            or re.match(r"^<?\d+;\d+;\d+[mM]$", key)
        )

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
            new_mode = self._cycle_agent_mode(session)
            turn = self._active_turn_for_session(session)
            if turn is not None:
                turn.mode = new_mode
                turn.runtime.set_mode(new_mode)
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

        if self._is_escape(key):
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
                back_requested=True,
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

        if self._is_ctrl_c(key) or self._is_ctrl_x(key):
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
            if action is not None and action.kind == "back_project":
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                    back_requested=True,
                )
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
            if action is not None and action.kind == "toggle_plan":
                self._toggle_plan(session.path)
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
            if action is not None and action.kind == "toggle_activity_item":
                self._toggle_activity_item(action.text)
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            if action is not None and action.kind == "toggle_activity_entry":
                self._toggle_activity_entry(action.text)
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            if action is not None and action.kind == "scroll_activity_item":
                self._scroll_activity_item(action.text, action.value)
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
                self._runtime_for_session(session).end_process(action.text, session.path)
                return RunningKeyResult(
                    input_text,
                    cursor,
                    "Process ended",
                    "light",
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
        raw_mouse_action = self._raw_mouse_action(
            key,
            stdscr,
            input_text,
            suggestions,
            command_selected,
        )
        if raw_mouse_action is not None:
            if raw_mouse_action.kind == "back_project":
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                    back_requested=True,
                )
            if raw_mouse_action.kind == "cursor":
                return RunningKeyResult(
                    input_text,
                    raw_mouse_action.value,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            if raw_mouse_action.kind == "scroll":
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                    scroll_delta=raw_mouse_action.value,
                    clear_anchor=True,
                )
            if raw_mouse_action.kind == "toggle_work":
                self._toggle_work_turn(raw_mouse_action.text)
            elif raw_mouse_action.kind == "toggle_plan":
                self._toggle_plan(session.path)
            elif raw_mouse_action.kind == "toggle_work_line":
                self._toggle_work_line(raw_mouse_action.text)
            elif raw_mouse_action.kind == "toggle_activity_item":
                self._toggle_activity_item(raw_mouse_action.text)
            elif raw_mouse_action.kind == "toggle_activity_entry":
                self._toggle_activity_entry(raw_mouse_action.text)
            elif raw_mouse_action.kind == "scroll_activity_item":
                self._scroll_activity_item(raw_mouse_action.text, raw_mouse_action.value)
            elif raw_mouse_action.kind == "kill_process":
                self._runtime_for_session(session).end_process(
                    raw_mouse_action.text,
                    session.path,
                )
                return RunningKeyResult(
                    input_text,
                    cursor,
                    "Process ended",
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            elif raw_mouse_action.kind == "command":
                command = suggestions[raw_mouse_action.value].command
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
        if self._is_raw_mouse_fragment_key(key):
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
            )
        if self._is_option_delete(key):
            word_start = self._previous_prompt_word(input_text, cursor)
            updated_input = input_text[:word_start] + input_text[cursor:]
            return RunningKeyResult(
                updated_input,
                word_start,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                0,
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
        *,
        active_turn_elapsed: float | None = None,
        render_events: bool = True,
        deferred_events: list[RuntimeUiEvent] | None = None,
        defer_interactive: bool = False,
    ) -> tuple[str | None, float | None, str, int]:
        current_working = working_text
        current_deadline = working_deadline
        current_final = final_text
        current_work_count = work_count
        while True:
            if deferred_events:
                event = deferred_events.pop(0)
            else:
                try:
                    event = events.get_nowait()
                except queue.Empty:
                    return current_working, current_deadline, current_final, current_work_count

            if defer_interactive and event.kind in {"approval", "question"}:
                if deferred_events is not None:
                    deferred_events.insert(0, event)
                return current_working, current_deadline, current_final, current_work_count

            if event.kind == "status":
                status_text, status_seconds = self._parse_runtime_status(event.text)
                current_working = status_text
                current_deadline = (
                    time.monotonic() + status_seconds if status_seconds is not None else None
                )
                if self._should_persist_status_statement(status_text):
                    self.home.append_session_event(
                        session.path,
                        "work_message",
                        {
                            "message": status_text,
                            "role": "tool",
                            "turn_id": turn_id,
                        },
                    )
                    current_work_count += 1
            elif event.kind == "delta":
                current_final += event.text
            elif event.kind in {"message", "tool_message", "command"} and event.text:
                role = "tool" if event.kind == "tool_message" else "agent"
                if event.kind == "command":
                    role = "tool"
                current_final = ""
                current_working = None
                current_deadline = None
                if render_events:
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
                        animate=event.kind == "message",
                        active_turn_elapsed=active_turn_elapsed,
                    )
                if event.kind == "message":
                    self.home.append_session_event(
                        session.path,
                        "agent_message",
                        {
                            "message": event.text,
                            "turn_id": turn_id,
                            "intermediate": True,
                        },
                    )
                    current_work_count += 1
                else:
                    payload = {
                        "message": event.text,
                        "role": role,
                        "turn_id": turn_id,
                    }
                    if event.kind == "command" and event.command:
                        payload["command"] = event.command
                    self.home.append_session_event(session.path, "work_message", payload)
                    current_work_count += 1
            elif event.kind == "approval":
                approval_request = event.approval_request
                approval_response = event.approval_response
                if approval_request is not None and approval_response is not None:
                    choice = self._request_command_approval(
                        stdscr,
                        session,
                        approval_request,
                        scroll=scroll,
                        anchor_line=anchor_line,
                    )
                    approval_response.put(choice)
                    approval_message = self._approval_work_message(
                        approval_request,
                        choice,
                    )
                    if approval_message:
                        approval_role = (
                            "forbidden"
                            if choice
                            in {ApprovalChoice.REJECT, ApprovalChoice.ALWAYS_REJECT}
                            else "tool"
                        )
                        self.home.append_session_event(
                            session.path,
                            "work_message",
                            {
                                "message": approval_message,
                                "role": approval_role,
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

    def _should_persist_status_statement(self, status_text: str) -> bool:
        normalized = status_text.strip()
        if not normalized:
            return False
        return normalized not in {"Thinking", "Waiting", "Loading model"}

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
        if choice in {ApprovalChoice.ALLOW, ApprovalChoice.ALWAYS_ALLOW}:
            return ""

        statement = request.statement.strip() or "command"
        if choice == ApprovalChoice.ALWAYS_REJECT:
            reason = (
                f"{request.allowance_label or 'Matching commands'} blocked for this session "
                "by user policy."
            )
        else:
            reason = "Rejected by user."
        return self._blocked_statement_message(statement, request.command, reason)

    def _blocked_statement_message(self, statement: str, command: str, reason: str) -> str:
        return f"Blocked: {statement}\nCommand: {command}\nReason: {reason}"

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
        animate: bool = True,
        active_turn_elapsed: float | None = None,
    ) -> None:
        if not animate:
            self._draw_session(
                stdscr,
                session,
                [
                    *self._read_message_lines(session.path),
                    MessageLine(role, message),
                ],
                input_text,
                cursor,
                scroll,
                anchor_line=anchor_line,
                prompt_notice=prompt_notice,
                prompt_notice_role=prompt_notice_role,
                active_turn_elapsed=active_turn_elapsed,
            )
            return
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
                active_turn_elapsed=active_turn_elapsed,
            )
            time.sleep(0.003)

    def _request_command_approval(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        request: CommandApprovalRequest,
        scroll: int = 0,
        anchor_line: int | None = None,
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
            scroll=scroll,
            anchor_line=anchor_line,
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
        cache_key = self._session_cache_key(session_path)
        expanded_turns = self._expanded_work_turns_key()
        if cache_key is not None:
            cached = self._message_line_cache.get(session_path)
            if (
                cached is not None
                and cached[0] == cache_key[0]
                and cached[1] == cache_key[1]
                and cached[2] == expanded_turns
            ):
                return cached[3]

        lines: list[MessageLine] = []
        turn_lines: dict[str, list[MessageLine]] = {}
        turn_summaries: dict[str, str] = {}

        def append_turn_line(turn_id: str, line: MessageLine) -> None:
            if not turn_id:
                lines.append(line)
                return
            if turn_id not in turn_lines:
                turn_lines[turn_id] = []
                lines.append(MessageLine("__turn_placeholder__", turn_id))
            turn_lines[turn_id].append(line)

        def append_turn_summary(turn_id: str, message: str) -> None:
            if not turn_id:
                lines.append(MessageLine("work_summary", f"{message} · expand"))
                return
            if turn_id not in turn_lines:
                turn_lines[turn_id] = []
                lines.append(MessageLine("__turn_placeholder__", turn_id))
            turn_summaries[turn_id] = message

        for event_index, event in enumerate(self._session_events(session_path)):
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
                role = "agent_intermediate" if payload.get("intermediate") else "agent"
                append_turn_line(turn_id, MessageLine(role, message, turn_id))
            elif event_type == "system_message" and message:
                role = str(payload.get("role", "system"))
                if role == "question":
                    continue
                turn_id = str(payload.get("turn_id", ""))
                expansion_key = self._session_work_line_key(role, turn_id, event_index)
                message, detail_body, detail_title = self._message_display_parts(
                    role,
                    message,
                    str(payload.get("command", "")),
                )
                append_turn_line(
                    turn_id,
                    MessageLine(
                        role,
                        message,
                        turn_id,
                        expansion_key,
                        detail_title,
                        detail_body,
                    )
                    if turn_id
                    else MessageLine(
                        role,
                        message,
                        expansion_key=expansion_key,
                        detail_title=detail_title,
                        detail_body=detail_body,
                    ),
                )
            elif event_type == "work_message" and message:
                turn_id = str(payload.get("turn_id", ""))
                role = str(payload.get("role", "tool"))
                expansion_key = self._session_work_line_key(role, turn_id, event_index)
                message, detail_body, detail_title = self._message_display_parts(
                    role,
                    message,
                    str(payload.get("command", "")),
                )
                append_turn_line(
                    turn_id,
                    MessageLine(
                        role,
                        message,
                        turn_id,
                        expansion_key,
                        detail_title,
                        detail_body,
                    )
                    if turn_id
                    else MessageLine(
                        role,
                        message,
                        expansion_key=expansion_key,
                        detail_title=detail_title,
                        detail_body=detail_body,
                    ),
                )
            elif event_type == "work_summary" and message:
                turn_id = str(payload.get("turn_id", ""))
                append_turn_summary(turn_id, message)
        rendered_lines: list[MessageLine] = []
        for line in lines:
            if line.role != "__turn_placeholder__":
                rendered_lines.append(line)
                continue
            turn_id = line.text
            summary = turn_summaries.get(turn_id)
            if summary:
                if turn_id in self._expanded_work_turns:
                    rendered_lines.extend(turn_lines.get(turn_id, []))
                    rendered_lines.append(
                        MessageLine("work_summary", f"{summary} · collapse", turn_id)
                    )
                else:
                    rendered_lines.append(
                        MessageLine("work_summary", f"{summary} · expand", turn_id)
                    )
            else:
                rendered_lines.extend(turn_lines.get(turn_id, []))
        if cache_key is not None:
            self._message_line_cache[session_path] = (
                cache_key[0],
                cache_key[1],
                expanded_turns,
                rendered_lines,
            )
        return rendered_lines

    def _session_events(self, session_path: Path) -> list[dict[str, Any]]:
        cache_key = self._session_cache_key(session_path)
        if cache_key is None:
            return self.home.read_session_events(session_path)
        cached = self._session_event_cache.get(session_path)
        if (
            cached is not None
            and cached[0] == cache_key[0]
            and cached[1] == cache_key[1]
        ):
            return cached[2]
        events = self.home.read_session_events(session_path)
        self._session_event_cache[session_path] = (cache_key[0], cache_key[1], events)
        return events

    def _session_cache_key(self, session_path: Path) -> tuple[int, int] | None:
        try:
            stat = session_path.stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _expanded_work_turns_key(self) -> tuple[str, ...]:
        return tuple(sorted(self._expanded_work_turns))

    def _expanded_work_lines_key(self) -> tuple[str, ...]:
        return tuple(sorted(self._expanded_work_lines))

    def _load_config_cached(self) -> dict[str, Any]:
        cache_key = self._config_cache_key()
        if cache_key is None:
            return self.home.load_config()
        if (
            self._config_cache is not None
            and self._config_cache[0] == cache_key[0]
            and self._config_cache[1] == cache_key[1]
        ):
            return self._config_cache[2]
        config = self.home.load_config()
        self._config_cache = (cache_key[0], cache_key[1], config)
        return config

    def _config_cache_key(self) -> tuple[int, int] | None:
        try:
            stat = self.home.config_path.stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _message_display_parts(
        self,
        role: str,
        message: str,
        command: str = "",
    ) -> tuple[str, str, str]:
        if role != "forbidden":
            return message, command, ""

        lines = message.splitlines()
        statement = lines[0].strip() if lines else message.strip()
        detail_command = command.strip()
        reason = ""
        for line in lines[1:]:
            if line.startswith("Command: "):
                detail_command = line.removeprefix("Command: ").strip()
            elif line.startswith("Reason: "):
                reason = line.removeprefix("Reason: ").strip()
        if statement.startswith("Blocked: "):
            detail_title = f"Reason: {reason}" if reason else ""
            return statement, detail_command, detail_title

        if message.startswith("Blocked command: "):
            command, separator, reason = message.removeprefix("Blocked command: ").partition(
                " · "
            )
            detail_title = f"Reason: {reason.strip()}" if separator else ""
            detail_command = command.strip()
            return f"Blocked: {detail_command}", detail_command, detail_title

        return message, detail_command, ""

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

        display_text = self._ellipsized_statement_text(
            self._work_message_display_text(message),
            safe_width,
        )
        return [
            MessageLine(
                message.role,
                display_text,
                message.meta,
                expansion_key,
                message.detail_title,
                message.detail_body,
            )
        ]

    def _is_expandable_work_role(self, role: str) -> bool:
        return role in {"tool", "approved", "forbidden"}

    def _work_message_display_text(self, message: MessageLine) -> str:
        return self._single_line_work_text(message.text)

    def _single_line_work_text(self, text: str) -> str:
        return " ".join(text.replace("\r", " ").replace("\n", " ").split())

    def _ellipsized_statement_text(self, text: str, width: int) -> str:
        safe_width = max(1, width)
        if len(text) <= safe_width:
            return text
        if safe_width <= 3:
            return "." * safe_width
        return f"{text[: safe_width - 3].rstrip()}..."

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
        for title_line in self._work_box_content_lines(message.text, inner_width):
            title = title_line[:inner_width].ljust(inner_width)
            lines.append(MessageLine("work_box", f"│ {title} │", message.meta, expansion_key))
        if message.detail_title:
            for title_line in self._work_box_content_lines(message.detail_title, inner_width):
                title = title_line[:inner_width].ljust(inner_width)
                lines.append(MessageLine("work_box", f"│ {title} │", message.meta, expansion_key))
        if message.detail_body:
            lines.append(self._empty_work_box_line(inner_width, message.meta, expansion_key))
            for content_line in self._work_box_content_lines(message.detail_body, inner_width):
                content = content_line[:inner_width].ljust(inner_width)
                lines.append(MessageLine("work_box", f"│ {content} │", message.meta, expansion_key))
        lines.append(MessageLine("work_box", f"╰{border}╯", message.meta, expansion_key))
        return lines

    def _empty_work_box_line(
        self,
        inner_width: int,
        meta: str,
        expansion_key: str,
    ) -> MessageLine:
        return MessageLine(
            "work_box",
            f"│ {' ' * max(1, inner_width)} │",
            meta,
            expansion_key,
        )

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
        identity = f"{message.text}\n{message.detail_body}\n{message.detail_title}"
        digest = hashlib.sha1(identity.encode("utf-8", errors="replace")).hexdigest()
        return f"{message.role}:{message.meta}:{digest}"

    def _message_kind(self, role: str) -> str:
        if role == "user":
            return "user"
        if role == "agent":
            return "agent"
        if role == "agent_intermediate":
            return "agent"
        return "working"

    def _toggle_work_turn(self, turn_id: str) -> None:
        if turn_id in self._expanded_work_turns:
            self._expanded_work_turns.remove(turn_id)
        elif turn_id:
            self._expanded_work_turns.add(turn_id)

    def _toggle_plan(self, session_path: Path) -> None:
        if session_path in self._expanded_plan_sessions:
            self._expanded_plan_sessions.remove(session_path)
        else:
            self._expanded_plan_sessions.add(session_path)

    def _toggle_work_line(self, expansion_key: str) -> None:
        if expansion_key in self._expanded_work_lines:
            self._expanded_work_lines.remove(expansion_key)
        elif expansion_key:
            self._expanded_work_lines.add(expansion_key)

    def _toggle_activity_item(self, activity_key: str) -> None:
        if activity_key in self._expanded_activity_items:
            self._expanded_activity_items.remove(activity_key)
            self._remove_activity_entry_expansions(activity_key)
        elif activity_key:
            self._expanded_activity_items = {activity_key}
            self._expanded_activity_entries.clear()
            self._activity_detail_scrolls.setdefault(activity_key, 0)

    def _toggle_activity_entry(self, entry_key: str) -> None:
        if entry_key in self._expanded_activity_entries:
            self._expanded_activity_entries.remove(entry_key)
        elif entry_key:
            self._expanded_activity_entries.add(entry_key)

    def _scroll_activity_item(self, activity_key: str, delta: int) -> None:
        if not activity_key:
            return
        current = self._activity_detail_scrolls.get(activity_key, 0)
        self._activity_detail_scrolls[activity_key] = max(0, current - delta)

    def _remove_activity_entry_expansions(self, activity_key: str) -> None:
        owner_id = activity_key.partition(":")[2]
        if not owner_id:
            return
        prefix = f"activity:{owner_id}:"
        self._expanded_activity_entries = {
            entry_key
            for entry_key in self._expanded_activity_entries
            if not entry_key.startswith(prefix)
        }

    def _activate_agent_mode(self, mode: AgentMode | str) -> AgentMode:
        agent_mode = AgentMode.parse(mode, self.agent_mode)
        self.agent_mode = agent_mode
        self.runtime.set_mode(agent_mode)
        return agent_mode

    def _cycle_agent_mode(self, session: SessionRecord | None = None) -> AgentMode:
        base_mode = (
            AgentMode.parse(session.mode, self.agent_mode)
            if session is not None
            else self.agent_mode
        )
        next_mode = base_mode.next()
        self._activate_agent_mode(next_mode)
        if session is not None:
            self.home.update_session_mode(session.path, next_mode)
        else:
            config = self.home.load_config()
            config["agent_mode"] = next_mode.value
            self.home.save_config(config)
        return next_mode

    def _mode_hint_attr_name(self) -> str:
        if self.agent_mode == AgentMode.AUTONOMOUS:
            return "danger"
        if self.agent_mode == AgentMode.AUTO:
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
        for event in self._session_events(session.path):
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
        ends_with_slash = normalized_query.endswith("/")
        query_has_slash = "/" in normalized_query
        # Terminal name after the last "/". Empty when the query ends with "/".
        query_name = normalized_query.rsplit("/", 1)[-1] if query_has_slash else normalized_query
        matches: list[tuple[int, int, str, MenuChoice]] = []
        for path in self._workspace_reference_paths(query=query):
            relative = self._relative_workspace_path(path)
            if path.is_dir():
                relative = f"{relative}/"
            name = self._reference_label(path)
            name_search = name.lower()
            relative_search = relative.lower()

            if not normalized_query:
                matches.append((0, len(relative), relative_search, MenuChoice(relative, relative, "", "")))
                continue

            # Determine relevance: how well does this path match the query?
            if ends_with_slash:
                # Trailing-slash query: show children of this path.
                # When _workspace_reference_paths sees a trailing slash it scans
                # the directory's children directly, so all paths here are direct
                # children — show them all.
                rank = 0
            elif query_has_slash:
                # Multi-segment query: check if the full relative path contains the
                # query as a subsequence (e.g. "anomx/agent" in "src/anomx/agent/").
                if normalized_query in relative_search:
                    rank = 1
                elif query_name and query_name in relative_search:
                    rank = 2
                else:
                    continue
            else:
                # Single-segment query: match against the terminal name.
                if name_search.startswith(normalized_query):
                    rank = 0
                elif normalized_query in name_search:
                    rank = 2
                elif normalized_query in relative_search:
                    rank = 3
                else:
                    continue

            matches.append((rank, len(relative), relative_search, MenuChoice(relative, relative, "", normalized_query)))

        matches.sort(key=lambda match: (match[0], match[1], match[2]))
        return [match[3] for match in matches[:FILE_REFERENCE_LIMIT]]

    def _workspace_reference_paths(self, query: str = "") -> tuple[Path, ...]:
        now = time.monotonic()

        # Empty query (just "@"): show first-level entries only (instant, no full walk)
        if not query:
            if (
                self._file_reference_cache
                and now - self._file_reference_cache_at < FILE_REFERENCE_CACHE_SECONDS
            ):
                return self._file_reference_cache
            paths: list[Path] = []
            try:
                for entry in sorted(
                    Path(self.workspace_root).iterdir(),
                    key=lambda e: (not e.is_dir(), e.name.lower()),
                ):
                    if entry.is_dir():
                        if self._ignore_file_reference_dir(entry.name):
                            continue
                        paths.append(entry)
                    elif entry.is_file():
                        if self._ignore_file_reference_file(entry.name):
                            continue
                        paths.append(entry)
                    if len(paths) >= FILE_REFERENCE_FIRST_LEVEL_LIMIT:
                        break
            except OSError:
                pass
            self._file_reference_cache = tuple(
                sorted(paths, key=lambda path: self._relative_workspace_path(path).lower())
            )
            self._file_reference_cache_at = now
            return self._file_reference_cache

        # Trailing-slash query (e.g. "src/", "anomx/agent/"): scan that
        # directory's children directly so the user sees what's "behind"
        # the path. Supports partial paths (e.g. "anomx/" finds
        # "src/anomx/" by searching the walk cache).
        stripped = query.rstrip("/")
        if query.endswith("/") and stripped:
            # Try resolving as a literal workspace path first.
            literal_dir = (self.workspace_root / stripped).resolve()
            try:
                literal_dir.relative_to(self.workspace_root)
            except ValueError:
                literal_dir = None
            if literal_dir and literal_dir.is_dir():
                paths: list[Path] = []
                try:
                    for entry in sorted(
                        literal_dir.iterdir(),
                        key=lambda e: (not e.is_dir(), e.name.lower()),
                    ):
                        if entry.is_dir():
                            if self._ignore_file_reference_dir(entry.name):
                                continue
                            paths.append(entry)
                        elif entry.is_file():
                            if self._ignore_file_reference_file(entry.name):
                                continue
                            paths.append(entry)
                        if len(paths) >= FILE_REFERENCE_FIRST_LEVEL_LIMIT:
                            break
                except OSError:
                    pass
                return tuple(
                    sorted(paths, key=lambda p: self._relative_workspace_path(p).lower())
                )
            # Not found as literal path — fall through to full walk.

        # Non-empty query: search across the full workspace tree (cached).
        # Walks all paths so that nested and partial-path queries like
        # "anomx/agent" or "src/anomx/agent/ui" find matches anywhere.
        if (
            self._file_reference_full_cache
            and now - self._file_reference_full_cache_at < FILE_REFERENCE_CACHE_SECONDS
        ):
            pass
        else:
            _paths: list[Path] = []
            for root, dirnames, filenames in os.walk(self.workspace_root):
                dirnames[:] = [
                    dirname
                    for dirname in dirnames
                    if not self._ignore_file_reference_dir(dirname)
                ]
                for dirname in dirnames:
                    _paths.append(Path(root) / dirname)
                    if len(_paths) >= FILE_REFERENCE_SCAN_LIMIT:
                        break
                if len(_paths) >= FILE_REFERENCE_SCAN_LIMIT:
                    break
                for filename in filenames:
                    if self._ignore_file_reference_file(filename):
                        continue
                    path = Path(root) / filename
                    if not path.is_file():
                        continue
                    _paths.append(path)
                    if len(_paths) >= FILE_REFERENCE_SCAN_LIMIT:
                        break
                if len(_paths) >= FILE_REFERENCE_SCAN_LIMIT:
                    break
            self._file_reference_full_cache = tuple(
                sorted(_paths, key=lambda path: self._relative_workspace_path(path).lower())
            )
            self._file_reference_full_cache_at = now

        return self._file_reference_full_cache

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

    def _prompt_reference_labels(
        self,
        file_references: Mapping[str, str] | None,
        image_attachments: Mapping[str, Mapping[str, str]] | None,
    ) -> dict[str, str]:
        labels = dict(file_references or {})
        for token in image_attachments or {}:
            labels[token] = ""
        return labels

    def _consume_dropped_images(
        self,
        input_text: str,
        cursor: int,
        image_attachments: dict[str, dict[str, str]],
    ) -> tuple[str, int, tuple[dict[str, str], ...]]:
        matches = list(IMAGE_DROP_CANDIDATE_PATTERN.finditer(input_text))
        if not matches:
            return input_text, cursor, ()

        updated = input_text
        updated_cursor = cursor
        added: list[dict[str, str]] = []
        for match in reversed(matches):
            raw_path = match.group("path")
            path = self._dropped_image_path(raw_path)
            if path is None:
                continue
            label = self._image_attachment_label(path, image_attachments)
            token = f"[image: {label}]"
            if token in image_attachments:
                continue
            payload = self._image_attachment_payload(path, label, token)
            if payload is None:
                continue

            start, end = match.span("path")
            suffix = "" if end < len(updated) and updated[end].isspace() else " "
            replacement = f"{token}{suffix}"
            updated = updated[:start] + replacement + updated[end:]
            delta = len(replacement) - (end - start)
            if updated_cursor >= end:
                updated_cursor += delta
            elif updated_cursor > start:
                updated_cursor = start + len(replacement)
            image_attachments[token] = payload
            added.append(payload)

        return updated, max(0, min(updated_cursor, len(updated))), tuple(reversed(added))

    def _dropped_image_path(self, raw_path: str) -> Path | None:
        for candidate in self._dropped_path_candidates(raw_path):
            path = Path(candidate).expanduser()
            if not path.is_absolute():
                path = self.cwd / path
            with suppress(OSError):
                resolved = path.resolve()
                if resolved.is_file() and image_mime_type(resolved) is not None:
                    return resolved
        return None

    def _dropped_path_candidates(self, raw_path: str) -> tuple[str, ...]:
        stripped = raw_path.strip().strip("\"'")
        if not stripped:
            return ()
        if stripped.startswith("file://"):
            parsed = urlparse(stripped)
            path = unquote(parsed.path)
            if parsed.netloc and not path.startswith(f"//{parsed.netloc}"):
                path = f"//{parsed.netloc}{path}"
            return (path,)

        candidates = [stripped, stripped.replace("\\ ", " ")]
        with suppress(ValueError):
            parts = shlex.split(stripped)
            if len(parts) == 1:
                candidates.append(parts[0])
        unique: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in unique:
                unique.append(candidate)
        return tuple(unique)

    def _image_attachment_label(
        self,
        path: Path,
        image_attachments: Mapping[str, Mapping[str, str]],
    ) -> str:
        base = path.name or "image"
        existing = {
            str(payload.get("label", ""))
            for payload in image_attachments.values()
            if isinstance(payload, Mapping)
        }
        if base not in existing:
            return base
        stem = path.stem or "image"
        suffix = path.suffix
        index = 2
        while f"{stem}-{index}{suffix}" in existing:
            index += 1
        return f"{stem}-{index}{suffix}"

    def _image_attachment_payload(
        self,
        path: Path,
        label: str,
        token: str,
    ) -> dict[str, str] | None:
        mime_type = image_mime_type(path)
        if mime_type is None:
            return None
        return {
            "label": label,
            "token": token,
            "path": path.as_posix(),
            "mime_type": mime_type,
        }

    def _active_image_attachments(
        self,
        prompt: str,
        image_attachments: Mapping[str, dict[str, str]],
    ) -> dict[str, dict[str, str]]:
        return {
            token: payload
            for token, payload in image_attachments.items()
            if self._file_reference_label_pattern(token).search(prompt)
        }

    def _append_unsupported_image_notice(
        self,
        session: SessionRecord,
        images: tuple[dict[str, str], ...],
    ) -> None:
        if not images:
            return
        config = self.home.load_config()
        provider_key = str(config.get("provider", session.provider))
        model = str(config.get("model", session.model))
        if backend_supports_image_input(provider_key, model):
            return
        provider = provider_by_key(provider_key)
        provider_label = provider.label if provider is not None else provider_key
        labels = ", ".join(image["label"] for image in images if image.get("label"))
        self.home.append_session_event(
            session.path,
            "system_message",
            {
                "message": (
                    f"{provider_label}/{model} does not support image input. "
                    f"Dropped image attachments will be kept in the prompt but sent "
                    f"to this backend as text only: {labels}."
                ),
                "role": "warning",
            },
        )

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
            replacement = path if label == path else f"{label} [{path}]"
            expanded = self._file_reference_label_pattern(label).sub(replacement, expanded)
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


    def _is_option_delete(self, key: str | int) -> bool:
        return key in {
            "[3;3~",      # Option+Delete/Backspace in iTerm2, many terminals
            "[3;5~",      # Ctrl+Delete (also common)
        }
