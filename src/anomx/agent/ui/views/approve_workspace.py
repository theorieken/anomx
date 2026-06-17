"""Workspace trust approval view."""

from __future__ import annotations

import curses
import textwrap

from anomx.agent.ui.models import (
    AgentState,
    CursesWindow,
)


class ApproveWorkspaceViewMixin:
    """Workspace trust approval view."""

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
