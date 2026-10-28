"""One-shot ``anomx "<request>"`` mode that turns a request into a shell command."""

from __future__ import annotations

import os
import platform
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TextIO

from anomx.agent.backends import backend_for_provider
from anomx.agent.helpers.terminal import terminal_cell_spans
from anomx.agent.helpers.terminal_colors import (
    Rgb,
    blend_color,
    read_terminal_theme,
    terminal_color_index,
)
from anomx.agent.runtime import AgentRuntime
from anomx.agent.store import AnomxHome, provider_by_key
from anomx.agent.ui.constants import ACTIVITY_FRAME_SECONDS, ACTIVITY_WAVE_SHADE_COUNT
from anomx.agent.ui.wave import activity_wave_strengths

COMMAND_MODE_LABEL = "Thinking"
COMMAND_MODE_EFFORT = "low"
# Some providers occasionally answer with an empty message; ask again before failing.
COMMAND_MODE_ATTEMPTS = 3
CONTEXT_ENTRY_LIMIT = 40

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
    """Ask the selected model for a shell command and hand it to the user.

    In an interactive terminal the command is placed on the shell's prompt line,
    ready to edit or run with Enter. Otherwise it is printed, so stdout holds only
    the command and stays usable in ``$(anomx "...")``. The wave animation goes
    to stderr while the model works.
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
    backend = backend_for_provider(provider_key, AgentRuntime(home, cwd))
    if backend is None:
        _error(f"Unknown backend: {provider_key}")
        return 1
    backend.background_effort = COMMAND_MODE_EFFORT

    result: list[str | None] = []
    failure: list[Exception] = []

    def suggest() -> None:
        context = _system_context(cwd)
        try:
            for _attempt in range(COMMAND_MODE_ATTEMPTS):
                command = backend.suggest_shell_command(request, context, selected_model)
                if command:
                    result.append(command)
                    return
        except Exception as error:
            failure.append(error)

    worker = threading.Thread(target=suggest, name="anomx-command-mode", daemon=True)
    worker.start()
    try:
        _animate_until_done(worker, sys.stderr, use_color=use_color)
    except KeyboardInterrupt:
        return 130
    if failure:
        _error(str(failure[0]))
        return 1
    command = result[0] if result else None
    if not command:
        _error("The model returned no command.")
        return 1
    if command.startswith("#"):
        _error(command.removeprefix("#").strip())
        return 1
    _deliver_command(command, use_color=use_color)
    return 0


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


def _system_context(cwd: Path) -> str:
    """Describe where the command will run so the model picks matching syntax."""

    shell = Path(os.environ.get("SHELL") or "sh").name
    lines = [
        f"Operating system: {platform.system()} {platform.release()} ({platform.machine()})",
        f"Shell: {shell}",
        f"Working directory: {cwd}",
    ]
    with suppress(OSError):
        entries = sorted(cwd.iterdir(), key=lambda path: path.name.lower())
        names = [path.name + ("/" if path.is_dir() else "") for path in entries]
        listing = ", ".join(names[:CONTEXT_ENTRY_LIMIT])
        if len(names) > CONTEXT_ENTRY_LIMIT:
            listing += f", ... ({len(names) - CONTEXT_ENTRY_LIMIT} more)"
        lines.append(f"Directory entries: {listing or '(empty)'}")
    return "\n".join(lines)


def _animate_until_done(worker: threading.Thread, stream: TextIO, *, use_color: bool) -> None:
    """Draw the CLI's activity wave on one line until the worker finishes."""

    if not stream.isatty():
        worker.join()
        return
    shades = _wave_shades() if use_color else None
    spans = terminal_cell_spans(COMMAND_MODE_LABEL, len(COMMAND_MODE_LABEL))
    stream.write(HIDE_CURSOR)
    try:
        while worker.is_alive():
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


__all__ = ["run_command_mode", "shell_init_script"]
