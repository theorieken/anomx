"""Quick ``anomx "<request>"`` mode: one agent turn that ends in a command or an answer."""

from __future__ import annotations

import json
import os
import platform
import queue
import re
import shutil
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TextIO, cast
from uuid import uuid4

from anomx.agent.base.backends import strip_thinking_tags
from anomx.agent.exceptions import BackendFailure
from anomx.agent.helpers.extract_json import extract_json_object
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.helpers.terminal import terminal_cell_spans
from anomx.agent.helpers.terminal_colors import (
    Rgb,
    blend_color,
    read_terminal_theme,
    terminal_color_index,
)
from anomx.agent.helpers.tool_manager import ApprovalChoice, CommandApprovalRequest
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.store import AnomxHome, provider_by_key
from anomx.agent.ui.activity import ActivityPresentation, is_activity_statement
from anomx.agent.ui.constants import ACTIVITY_FRAME_SECONDS, ACTIVITY_WAVE_SHADE_COUNT
from anomx.agent.ui.wave import activity_wave_strengths

QuickResponseKind = Literal["command", "answer"]

QUICK_MODE_INSTRUCTIONS = """\
# Identity

You are Anomx in quick mode. The user typed `anomx <request>` in their terminal and is \
waiting at their shell prompt for your result. Each request names the directory it was \
issued from; relative paths and phrases like "here" or "this folder" refer to it.

# Outcome

Finish with exactly one of two outcomes:

1. command: one shell command that does what the user asked. It is placed on the user's \
prompt line; they review it and run it with Enter. Choose this whenever the request is \
something to do in the shell, including opening, editing, moving, searching, building or \
running things.
2. answer: a short plain-text answer, for questions a command would not answer better.

# Working

- Inspect before you answer when it makes the result exact: list directories, read or \
search files, run read-only commands. For example, resolve "the first file" to a real path.
- Never change anything yourself. No writes, edits, deletions, installs, network changes or \
process control: the user runs the final command. Commands with side effects are rejected.
- Do not ask questions or request approvals; make a sensible assumption instead.
- Be quick: inspect only what the request needs.
- Commands must suit this system: {system}. Prefer one line; chain with && or pipes and \
quote paths that contain spaces.
- Earlier quick requests from this machine and your final responses to them are included \
as conversation history, so follow-ups like "again for the other folder" work. You do not \
know whether the user ran an earlier command or what it printed.

# Final response

Reply with exactly one JSON object and nothing else:
{{"type": "command", "command": "<shell command>"}}
or
{{"type": "answer", "answer": "<answer text>"}}
The value is a JSON string: escape line breaks as \\n and double quotes as \\"."""

# Recovers an outcome whose string value broke JSON, e.g. with unescaped quotes.
LOOSE_RESPONSE_RE = re.compile(
    r'^\{\s*"type"\s*:\s*"(command|answer)"\s*,\s*"(?:command|answer)"\s*:\s*"(.*)"\s*\}$',
    re.DOTALL,
)


@dataclass(frozen=True)
class QuickResponse:
    """The final outcome of a quick request."""

    kind: QuickResponseKind
    text: str

    def to_message(self) -> str:
        """Encode the outcome the way the agent is asked to reply."""

        return json.dumps({"type": self.kind, self.kind: self.text}, ensure_ascii=False)


@dataclass(frozen=True)
class _ActivityEvent:
    kind: str
    text: str = ""
    payload: dict[str, Any] | None = None


RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
CLEAR_LINE = "\r\x1b[K"

# Set by the ``--shell-init`` wrapper: the command goes to this file and the
# shell itself places it on the prompt line.
COMMAND_FILE_ENV = "ANOMX_COMMAND_FILE"

SHELL_INIT_SCRIPTS = {
    "zsh": """\
anomx() {
  local anomx_file anomx_status
  anomx_file="$(mktemp "${TMPDIR:-/tmp}/anomx.XXXXXX")" || { command anomx "$@"; return; }
  ANOMX_COMMAND_FILE="$anomx_file" command anomx "$@"
  anomx_status=$?
  [[ -s "$anomx_file" ]] && print -z -- "$(<"$anomx_file")"
  command rm -f -- "$anomx_file"
  return $anomx_status
}
""",
    "bash": """\
anomx() {
  local anomx_file anomx_status anomx_line
  anomx_file="$(mktemp "${TMPDIR:-/tmp}/anomx.XXXXXX")" || { command anomx "$@"; return; }
  ANOMX_COMMAND_FILE="$anomx_file" command anomx "$@"
  anomx_status=$?
  if [[ ! -s "$anomx_file" ]]; then
    :
  elif (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
    printf '%s\n' "$(<"$anomx_file")"  # Prefilled input needs bash 4.4+.
  elif read -r -e -i "$(<"$anomx_file")" -p "${PS1@P}" anomx_line && [[ -n "$anomx_line" ]]; then
    history -s -- "$anomx_line"
    eval -- "$anomx_line"
    anomx_status=$?
  fi
  command rm -f -- "$anomx_file"
  return $anomx_status
}
""",
}


def run_command_mode(
    home: AnomxHome,
    request: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    use_color: bool = True,
    cwd: Path | None = None,
) -> int:
    """Run one agent turn for a terminal request and hand over its outcome.

    The agent may inspect the system first. A command is placed on the shell's
    prompt line, ready to edit or run with Enter (or printed when that is not
    possible); an answer is printed and the prompt stays empty. The live activity
    wave goes to stderr. Every exchange joins the machine-wide quick history.
    """

    cwd = (Path.cwd() if cwd is None else cwd).expanduser().resolve()
    config = home.load_config()
    # Before onboarding, the stored provider is only a default, not a selection.
    selected = provider is not None or bool(config.get("onboarding_complete"))
    provider_key = provider or (str(config.get("provider") or "").strip() if selected else "")
    selected_model = model or str(config.get("model") or "").strip()
    if provider_key and not selected_model:
        option = provider_by_key(provider_key)
        selected_model = option.models[0] if option is not None and option.models else ""
    if not provider_key or not selected_model:
        _error("No model selected yet. Run `anomx` once to choose a backend and model.")
        return 1
    if provider_key not in home.connected_backend_keys():
        _error(f"The {provider_key} backend is not connected. Run `anomx` to connect it.")
        return 1

    run_dir = home.root / "cli_runs" / uuid4().hex
    session_path = run_dir / "session.jsonl"
    run_dir.mkdir(parents=True)
    try:
        _seed_session(home, session_path, request, cwd)
        runtime = AgentRuntime(
            home,
            cwd,
            mode=AgentMode.AUTOMATIC,
            task_instructions=QUICK_MODE_INSTRUCTIONS.format(system=_system_description()),
            brain_sync=lambda: None,
        )
        events: queue.SimpleQueue[_ActivityEvent] = queue.SimpleQueue()
        result: list[str] = []

        def run_turn() -> None:
            try:
                result.append(
                    runtime.backend_response(
                        session_path,
                        _activity_callbacks(runtime, session_path, events),
                        provider=provider_key,
                        model=selected_model,
                    )
                )
            except Exception as error:  # pragma: no cover - defensive thread boundary
                result.append(BackendFailure(f"Agent crashed: {type(error).__name__}: {error}"))

        worker = threading.Thread(target=run_turn, name="anomx-quick-turn", daemon=True)
        worker.start()
        activity = ActivityPresentation()
        try:
            _animate_until_done(
                worker,
                sys.stderr,
                use_color=use_color,
                label=lambda: _drain_activity(activity, events),
            )
        except KeyboardInterrupt:
            runtime.abort_current_turn(session_path)
            return 130
    finally:
        if not home.full_session_logs_enabled():
            shutil.rmtree(run_dir, ignore_errors=True)

    response = result[0] if result else ""
    if isinstance(response, BackendFailure) or not response.strip():
        _error(str(response).strip() or "The model returned no response.")
        return 1
    outcome = parse_quick_response(response)
    home.append_cli_history(
        {"cwd": str(cwd), "request": request, "type": outcome.kind, "content": outcome.text}
    )
    if outcome.kind == "command":
        _deliver_command(outcome.text, use_color=use_color)
    else:
        print(outcome.text)
    return 0


def parse_quick_response(response: str) -> QuickResponse:
    """Read the agent's JSON outcome; anything else is shown as an answer."""

    visible = strip_thinking_tags(response).strip()
    payload = extract_json_object(visible) or _loose_response(visible)
    kind = str(payload.get("type") or "").strip().lower()
    if kind == "command":
        command = _clean_command(str(payload.get("command") or ""))
        if command:
            return QuickResponse("command", command)
    if kind == "answer":
        answer = str(payload.get("answer") or "").strip()
        if answer:
            return QuickResponse("answer", answer)
    return QuickResponse("answer", visible)


def shell_init_script(shell: str) -> str:
    """Return the shell function that puts suggested commands on the prompt line."""

    return SHELL_INIT_SCRIPTS[shell]


def _deliver_command(command: str, *, use_color: bool) -> None:
    command_file = os.environ.get(COMMAND_FILE_ENV)
    if command_file:
        Path(command_file).write_text(command, encoding="utf-8")
        return
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if interactive and _insert_into_prompt(command):
        return
    styled = use_color and sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    print(f"{BOLD}{command}{RESET}" if styled else command)
    if interactive and "\n" not in command:
        shell = Path(os.environ.get("SHELL") or "").name
        shell = shell if shell in SHELL_INIT_SCRIPTS else "zsh"
        tip = (
            f'Tip: add eval "$(anomx --shell-init {shell})" to your shell config '
            "to get commands on the prompt line."
        )
        print(f"{DIM}{tip}{RESET}" if styled else tip, file=sys.stderr)


def _insert_into_prompt(command: str) -> bool:
    """Queue the command as terminal input so the shell's next prompt shows it.

    Uses TIOCSTI, which macOS allows but Linux 6.2+ disables by default. A
    multi-line command is never queued because its newlines would run it.
    """

    if "\n" in command:
        return False
    try:
        import fcntl
        import termios
    except ImportError:
        return False
    request = getattr(termios, "TIOCSTI", None)
    if request is None:
        return False
    data = command.encode("utf-8")
    try:
        saved = termios.tcgetattr(0)
    except (termios.error, OSError):
        return False
    quiet = termios.tcgetattr(0)
    quiet[3] &= ~termios.ECHO  # The shell redraws the line itself; avoid a second copy.
    try:
        termios.tcsetattr(0, termios.TCSANOW, quiet)
        try:
            fcntl.ioctl(0, request, data[:1])
        except OSError:
            return False
        for byte in data[1:]:
            fcntl.ioctl(0, request, bytes([byte]))
    except (termios.error, OSError):
        return False
    finally:
        termios.tcsetattr(0, termios.TCSANOW, saved)
    return True


def _loose_response(text: str) -> dict[str, str]:
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    match = LOOSE_RESPONSE_RE.match(fenced.group(1) if fenced else text)
    if match is None:
        return {}
    kind, raw = match.groups()
    try:
        value = json.loads(f'"{raw}"', strict=False)
    except json.JSONDecodeError:
        value = raw.replace('\\"', '"').replace("\\n", "\n")
    return {"type": kind, kind: str(value)}


def _seed_session(home: AnomxHome, session_path: Path, request: str, cwd: Path) -> None:
    """Replay the quick history as plain exchanges, then add the current request."""

    for entry in home.load_cli_history():
        kind = str(entry.get("type") or "")
        content = str(entry.get("content") or "")
        if kind not in {"command", "answer"} or not content:
            continue
        home.append_session_event(
            session_path,
            "user_message",
            {"message": _request_message(str(entry.get("request") or ""), entry.get("cwd"))},
        )
        home.append_session_event(
            session_path,
            "agent_message",
            {"message": QuickResponse(cast(QuickResponseKind, kind), content).to_message()},
        )
    home.append_session_event(
        session_path, "user_message", {"message": _request_message(request, cwd)}
    )


def _request_message(request: str, cwd: object) -> str:
    return f"Directory: {cwd}\nRequest: {request.strip()}"


def _system_description() -> str:
    shell = Path(os.environ.get("SHELL") or "sh").name
    return f"{platform.system()} {platform.release()} ({platform.machine()}), {shell} shell"


def _clean_command(value: str) -> str:
    lines = [line.rstrip() for line in value.strip().strip("`").strip().splitlines()]
    if lines and lines[0].strip() in {"sh", "bash", "zsh", "shell", "console"}:
        lines = lines[1:]  # Language tag left from a code fence.
    return "\n".join(line.removeprefix("$ ") for line in lines if line.strip()).strip()


def _activity_callbacks(
    runtime: AgentRuntime,
    session_path: Path,
    events: queue.SimpleQueue[_ActivityEvent],
) -> RuntimeCallbacks:
    """Forward live work to the terminal and keep approvals to read-only commands."""

    def approve(request: CommandApprovalRequest) -> ApprovalChoice:
        evaluation = runtime.evaluate_command_request(session_path, request)
        if evaluation is not None and evaluation.allows_action and evaluation.read_only:
            return ApprovalChoice.ALLOW
        return ApprovalChoice.REJECT

    def statement(text: str, *_args: object) -> None:
        events.put(_ActivityEvent("statement", text))

    return RuntimeCallbacks(
        status=lambda text: events.put(_ActivityEvent("status", text)),
        thought=lambda text: events.put(_ActivityEvent("thought", text)),
        tool_message=statement,
        command=statement,
        delta=lambda text: events.put(_ActivityEvent("delta", text)),
        tool_activity=lambda payload: events.put(_ActivityEvent("tool", payload=payload)),
        approval=approve,
    )


def _drain_activity(
    activity: ActivityPresentation,
    events: queue.SimpleQueue[_ActivityEvent],
) -> str:
    """Apply pending runtime events and return the label to show, like the full UI."""

    while True:
        try:
            event = events.get_nowait()
        except queue.Empty:
            break
        if event.kind == "tool" and event.payload is not None:
            key = str(event.payload.get("id"))
            if event.payload.get("status") == "running":
                activity.running[key] = str(event.payload.get("statement") or "")
                activity.push(activity.running[key])
            else:
                activity.running.pop(key, None)
        elif event.kind == "status":
            text = event.text.partition(":")[0] if event.text.startswith("Waiting:") else event.text
            text = "Thinking" if text in {"Writing", "Executing", ""} else text
            activity.phase = text
            if is_activity_statement(text):
                activity.push(text)
        elif event.kind == "thought" and event.text:
            activity.push("Created a thought", thought=True)
        elif event.kind == "statement" and event.text.strip() and not activity.running:
            activity.push(event.text.strip())
        elif event.kind == "delta":
            activity.phase = "Thinking"
    return " ".join(activity.display(time.monotonic()).split())


def _animate_until_done(
    worker: threading.Thread,
    stream: TextIO,
    *,
    use_color: bool,
    label: Callable[[], str],
) -> None:
    """Draw the CLI's activity wave over the live label until the worker finishes."""

    if not stream.isatty():
        worker.join()
        return
    shades = _wave_shades() if use_color else None
    stream.write(HIDE_CURSOR)
    try:
        while worker.is_alive():
            width = max(1, shutil.get_terminal_size().columns - 1)
            spans = terminal_cell_spans(label(), width)
            frame = int(time.monotonic() / ACTIVITY_FRAME_SECONDS)
            strengths = activity_wave_strengths(spans, frame)
            parts = [CLEAR_LINE]
            for (_column, glyph, _cells), strength in zip(spans, strengths, strict=True):
                if shades:
                    parts.append(shades[round(strength * (len(shades) - 1))] + glyph)
                else:
                    parts.append((RESET if strength >= 0.5 else DIM) + glyph)
            parts.append(RESET)
            stream.write("".join(parts))
            stream.flush()
            worker.join(ACTIVITY_FRAME_SECONDS)
    finally:
        stream.write(CLEAR_LINE + SHOW_CURSOR)
        stream.flush()


def _wave_shades() -> tuple[str, ...] | None:
    """Blend the terminal's own foreground into its background, like the full UI."""

    if os.environ.get("NO_COLOR"):
        return None
    truecolor = os.environ.get("COLORTERM", "").lower() in {"truecolor", "24bit"}
    if not truecolor and "256" not in os.environ.get("TERM", ""):
        return None
    with _quiet_terminal():
        theme = read_terminal_theme()
    if theme is None:
        return None
    foreground, background = theme
    shades = []
    for index in range(ACTIVITY_WAVE_SHADE_COUNT):
        progress = index / (ACTIVITY_WAVE_SHADE_COUNT - 1)
        color = blend_color(background, foreground, 0.55 + 0.3 * progress)
        shades.append(_foreground_sequence(color, truecolor))
    return tuple(shades)


def _foreground_sequence(color: Rgb, truecolor: bool) -> str:
    if truecolor:
        return f"\x1b[38;2;{color[0]};{color[1]};{color[2]}m"
    return f"\x1b[38;5;{terminal_color_index(color, 256)}m"


@contextmanager
def _quiet_terminal() -> Iterator[None]:
    """Keep the terminal's color-query reply from echoing onto the screen."""

    try:
        import termios
    except ImportError:
        yield
        return
    try:
        saved = termios.tcgetattr(0)
    except (termios.error, OSError):
        yield
        return
    quiet = termios.tcgetattr(0)
    quiet[3] &= ~(termios.ICANON | termios.ECHO)
    try:
        termios.tcsetattr(0, termios.TCSANOW, quiet)
        yield
    finally:
        termios.tcsetattr(0, termios.TCSANOW, saved)


def _error(message: str) -> None:
    print(f"anomx: {message}", file=sys.stderr)


__all__ = ["QuickResponse", "parse_quick_response", "run_command_mode", "shell_init_script"]
