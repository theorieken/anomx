"""Project-level session list view."""

from __future__ import annotations

import curses
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime

from anomx.agent.helpers.state import (
    running_process_snapshots,
    running_subagent_snapshots,
)
from anomx.agent.store import (
    ProjectRecord,
    SessionRecord,
)
from anomx.agent.ui.constants import ACTIVITY_FRAME_SECONDS
from anomx.agent.ui.models import (
    BottomPanel,
    CommandSpec,
    CursesWindow,
    MenuChoice,
    PromptPasteSpan,
    SessionMouseAction,
)


class ProjectViewMixin:
    """Project-level session list view."""

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
        file_reference_searching: bool = False,
        bottom_panel: BottomPanel | None = None,
        prompt_hint_suffix: str = "",
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
        sessions_focused: bool = False,
    ) -> int:
        self._prompt_feature_session_path = None
        self._prompt_feature_notes = self._feature_notes_for_prompt(input_text)
        config = self._load_config_cached()
        provider = str(config.get("provider", "openai"))
        model = str(config.get("model", "gpt-5.5"))
        self._click_targets = {}
        height, width = self._draw_shell(
            stdscr,
            project.name,
            header_meta=self._model_header_label(provider, model),
            header_location=str(project.path),
        )
        layout = self._prompt_layout(stdscr, input_text, pasted_spans=pasted_spans)
        body_top = self._session_body_top()
        body_bottom = max(body_top + 1, layout.top_line)
        body_height = max(1, body_bottom - body_top)
        max_visible = max(1, body_height)
        if sessions_focused and selected < scroll:
            scroll = selected
        elif sessions_focused and selected >= scroll + max_visible:
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
                selected=sessions_focused and index == selected,
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
            searching=file_reference_searching,
        )
        active_panel = bottom_panel or file_panel or command_panel
        if active_panel is not None:
            self._draw_bottom_panel(stdscr, active_panel, input_text, pasted_spans)
        hint_suffix = prompt_hint_suffix or self._project_prompt_hint_suffix(
            sessions if sessions_focused else (),
            delete_pending_index,
        )
        self._draw_prompt_bar(
            stdscr,
            input_text,
            cursor,
            prompt_notice,
            prompt_notice_role,
            self._prompt_reference_labels(file_references, None) if file_references else None,
            hint_suffix=hint_suffix,
            pasted_spans=pasted_spans,
            focused=not sessions_focused,
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
        title_attr = self._attr("bold") & ~curses.A_BOLD if selected or running else quiet_attr
        right_text = self._project_session_right_text(
            session,
            running,
            selected=selected,
            delete_pending=delete_pending,
        )
        right_x = max(8, width - len(right_text) - 5) if right_text else width - 5
        title = self._project_session_title(session)
        if selected:
            self._add(stdscr, y, 2, "●", 1, self._attr("cursor") & ~curses.A_REVERSE)
        title_x = 4
        badge_text = " Confirmation needed " if needs_confirmation else ""
        title_width = max(1, right_x - title_x - len(badge_text) - 1)
        self._add(stdscr, y, title_x, title, title_width, title_attr)
        if running and not needs_confirmation:
            self._draw_activity_wave(
                stdscr, y, title_x, title, title_width,
                int(time.monotonic() / ACTIVITY_FRAME_SECONDS), title_attr, primary=True,
            )
        if badge_text:
            badge_x = title_x + min(len(title), title_width)
            self._add(
                stdscr, y, badge_x, badge_text, max(0, right_x - badge_x - 1),
                self._attr("warning_badge"),
            )
        if right_text:
            if session.unread and not running and not delete_pending:
                prefix = right_text.removesuffix("•")
                if prefix:
                    self._add(stdscr, y, right_x, prefix, len(prefix), quiet_attr)
                self._add(
                    stdscr,
                    y,
                    right_x + len(prefix),
                    "•",
                    1,
                    active_attr,
                )
            else:
                self._add(stdscr, y, right_x, right_text, len(right_text), quiet_attr)
        self._add_click_target(
            y,
            SessionMouseAction(
                "open_project_session",
                index,
                str(session.path),
                x_start=2,
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
        subagents = running_subagent_snapshots(events)
        if subagents:
            return "running " + self._process_runtime_duration(subagents[0].started_at)
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

    def _session_is_running(self, session: SessionRecord) -> bool:
        if self._active_turn_for_session(session) is not None:
            return True
        events = self._session_events(session.path)
        return bool(running_process_snapshots(events) or running_subagent_snapshots(events))

    def _session_mode_symbol(self, session: SessionRecord) -> str:
        turn = self._active_turn_for_session(session)
        if turn is not None:
            return turn.agent_symbol or turn.mode.symbol
        return session.mode.symbol

    def _project_mouse_action(
        self,
        stdscr: CursesWindow,
        input_text: str,
        command_suggestions: Sequence[CommandSpec] | None = None,
        command_selected: int = 0,
        file_suggestions: Sequence[MenuChoice] | None = None,
        file_selected: int = 0,
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
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
            layout = self._prompt_layout(stdscr, input_text, pasted_spans=pasted_spans)
            clicked_prompt = layout.prompt_line <= y < layout.prompt_line + layout.prompt_height
            if clicked_prompt and self._is_left_click(button_state):
                display_text = self._prompt_display_text(input_text, pasted_spans)
                display_cursor = self._prompt_display_cursor(
                    input_text,
                    len(input_text),
                    pasted_spans,
                )
                view_start = self._prompt_view_start(display_text, display_cursor, layout)
                clicked_line = view_start + (y - layout.prompt_line)
                display_click = (clicked_line * layout.input_width) + (x - layout.input_x)
                cursor = self._prompt_real_cursor(input_text, display_click, pasted_spans)
                return SessionMouseAction("cursor", cursor)
            if file_suggestions:
                file_panel = self._file_reference_bottom_panel(
                    list(file_suggestions),
                    file_selected,
                )
                if file_panel is not None and self._is_left_click(button_state):
                    choice = self._bottom_panel_mouse_choice_at(
                        stdscr,
                        file_panel,
                        y,
                        input_text,
                        pasted_spans,
                    )
                    if choice is not None:
                        return SessionMouseAction("file_reference", choice)
            else:
                command_panel = self._command_bottom_panel(
                    list(command_suggestions or ()),
                    command_selected,
                )
                if command_panel is not None and self._is_left_click(button_state):
                    choice = self._bottom_panel_mouse_choice_at(
                        stdscr,
                        command_panel,
                        y,
                        input_text,
                        pasted_spans,
                    )
                    if choice is not None:
                        return SessionMouseAction("command", choice)
        return None
