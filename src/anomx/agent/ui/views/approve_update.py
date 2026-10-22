"""Version update approval view."""

from __future__ import annotations

import curses
import os
import subprocess
import sys
import textwrap

from anomx import __version__
from anomx.agent.startup import check_for_update, check_latest_version
from anomx.agent.ui.models import (
    AgentState,
    CursesWindow,
)


class ApproveUpdateViewMixin:
    """Version update approval view."""

    @staticmethod
    def _check_latest_version() -> str | None:
        """Fetch the latest anomx version from PyPI.
        Returns the version string if newer, or None if up-to-date / unreachable.
        """
        return check_latest_version(__version__)

    def _startup_update_pending(self) -> bool:
        tasks = self._startup_tasks
        return bool(
            tasks is not None and not self._startup_version_checked and not tasks.version_ready
        )

    def _maybe_prompt_startup_update(self, stdscr: CursesWindow, *, allow_prompt: bool) -> None:
        tasks = self._startup_tasks
        if tasks is None or not tasks.version_ready or self._startup_version_checked:
            return
        if not allow_prompt or any(
            turn.worker.is_alive() for turn in self._active_session_turns.values()
        ):
            return
        self._startup_version_checked = True
        if tasks.latest_version:
            previous_state = self.state
            try:
                self._run_version_check(stdscr)
            finally:
                self.state = previous_state

    def _run_manual_update(self, stdscr: CursesWindow) -> None:
        """Always perform a fresh check, then install an available update."""
        if any(turn.worker.is_alive() for turn in self._active_session_turns.values()):
            self._message(
                stdscr, "Update", "Wait for running sessions to finish before updating Anomx."
            )
            return
        self._draw_overlay(stdscr, title="Update", body_lines=("Checking for updates…",), footer="")
        result = check_for_update(__version__)
        self._startup_version_checked = True
        if not result.checked:
            self._message(
                stdscr,
                "Update",
                "Could not check for updates. Check your connection and try /update again.",
            )
        elif result.available_version:
            self._install_update(stdscr, result.available_version)
        else:
            self._message(
                stdscr, "Update", f"No updates available. Anomx {__version__} is up to date."
            )

    def _run_version_check(self, stdscr: CursesWindow) -> bool:
        """Offer a ready startup update without waiting for the network."""
        startup_tasks = self._startup_tasks
        if (
            startup_tasks is None
            or not startup_tasks.version_ready
            or not startup_tasks.latest_version
        ):
            return True
        latest = startup_tasks.latest_version
        self.state = AgentState.VERSION_CHECK
        selected = 0
        stdscr.nodelay(False)
        while True:
            self._draw_version_check(stdscr, selected, latest)
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key):
                return True
            if key == curses.KEY_UP:
                selected = max(0, selected - 1)
            elif key == curses.KEY_DOWN:
                selected = min(1, selected + 1)
            elif self._is_enter(key):
                if selected == 0:
                    self._install_update(stdscr, latest)
                return True

    def _install_update(self, stdscr: CursesWindow, latest: str) -> None:
        stdscr.erase()
        stdscr.refresh()
        curses.endwin()
        print(f"Updating anomx from {__version__} to {latest}...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--upgrade", f"anomx=={latest}"],
            )
            print("Update successful. Restarting...")
        except (OSError, subprocess.CalledProcessError):
            input("Update failed. Press Enter to continue...")
        if self._startup_tasks is not None:
            self._startup_tasks.close()
        self._file_reference_search.close()
        close_runtime = getattr(self.runtime, "close", None)
        if callable(close_runtime):
            close_runtime()
        os.execve(
            sys.executable,
            [sys.executable, "-m", "anomx"] + sys.argv[1:],
            dict(os.environ),
        )

    def _draw_version_check(
        self,
        stdscr: CursesWindow,
        selected: int,
        latest: str,
    ) -> None:
        height, width = self._draw_shell(
            stdscr,
            "Version Check",
            f"v{__version__}",
        )
        y = 8
        self._add(
            stdscr,
            y,
            4,
            f"Local version:  {__version__}",
            width - 8,
            self._attr("bold"),
        )
        y += 1
        self._add(
            stdscr,
            y,
            4,
            f"Latest version: {latest}",
            width - 8,
            self._attr("accent"),
        )
        y += 3
        copy = "There is a new version of anomx available. Would you like to update now?"
        for line in textwrap.wrap(copy, width=max(24, width - 8)):
            self._add(stdscr, y, 4, line, width - 8)
            y += 1
        y += 2
        choices = ("Update now", "Update later")
        for index, choice in enumerate(choices):
            marker = "›" if index == selected else "•"
            attr = self._attr("accent") if index == selected else curses.A_NORMAL
            self._add(stdscr, y + index, 4, f"{marker} {index + 1}. {choice}", width - 8, attr)
        self._add(
            stdscr,
            min(height - 2, y + len(choices) + 2),
            4,
            "Enter to confirm · Esc to skip",
            width - 8,
            self._attr("light"),
        )
        stdscr.refresh()
