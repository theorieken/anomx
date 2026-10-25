"""Reusable menus, overlays, text popovers, and approval/question popups."""

from __future__ import annotations

import curses
import textwrap
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import replace

from anomx.agent.helpers.terminal import TerminalLine
from anomx.agent.helpers.tool_manager import (
    ApprovalChoice,
    CommandApprovalRequest,
    approval_choice_for_mode,
)
from anomx.agent.runtime import (
    QuestionOption,
    QuestionRequest,
    QuestionResponse,
)
from anomx.agent.store import (
    ProjectRecord,
    SessionRecord,
)
from anomx.agent.ui.models import (
    BottomPanel,
    CursesWindow,
    MenuChoice,
    PromptPasteEvent,
)


class PopupComponentMixin:
    """Reusable menus, overlays, text popovers, and approval/question popups."""

    @staticmethod
    def _filter_menu_choices(
        choices: tuple[MenuChoice, ...],
        query: str,
    ) -> tuple[MenuChoice, ...]:
        """Filter menu choices by a case-insensitive substring of the label.

        Matching choices carry the query as ``highlight`` so the renderer can
        mark the matched span. An empty query returns the choices unchanged.
        """
        needle = query.strip().lower()
        if not needle:
            return choices
        return tuple(
            replace(choice, highlight=query)
            for choice in choices
            if needle in choice.label.lower()
        )

    @staticmethod
    def _menu_search_subtitle(subtitle: str, query: str, searchable: bool) -> str:
        if not searchable:
            return subtitle
        if query:
            return f"Search: {query}"
        return f"{subtitle} · type to search" if subtitle else "Type to search"

    @staticmethod
    def _apply_menu_search_key(key: object, query: str) -> tuple[str, bool]:
        """Return the updated query and whether the key was consumed as search."""
        if isinstance(key, str) and key and key.isprintable() and key not in {"\n", "\r"}:
            return query + key, True
        return query, False

    @staticmethod
    def _insert_single_line_paste(value: str, cursor: int, text: str) -> tuple[str, int]:
        """Insert a terminal paste event into a single-line input.

        Clipboard tools commonly include a trailing newline.  A text popover is
        intentionally single-line, so keep the pasted secret or value intact
        while discarding line separators rather than treating them as submit
        keystrokes.
        """
        paste = text.replace("\r", "").replace("\n", "")
        return value[:cursor] + paste + value[cursor:], cursor + len(paste)

    def _menu(
        self,
        stdscr: CursesWindow,
        title: str,
        subtitle: str,
        choices: tuple[MenuChoice, ...],
        *,
        initial_value: str | None = None,
    ) -> str | None:
        return self._run_overlay_menu(stdscr, title, subtitle, choices, initial_value=initial_value)

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
        searchable: bool = False,
        replaces_prompt: bool = False,
    ) -> str | None:
        selected = 0
        current_scroll = scroll
        query = ""
        question_scroll = 0
        with suppress(curses.error):
            stdscr.nodelay(False)
        try:
            while True:
                if autonomous_value is not None:
                    automatic_choice = approval_choice_for_mode(self.agent_mode, None)
                    if automatic_choice is not None:
                        return automatic_choice.value
                messages = self._read_message_lines(session.path)
                active_choices = (
                    self._filter_menu_choices(choices, query) if searchable else choices
                )
                selected = max(0, min(selected, len(active_choices) - 1)) if active_choices else 0
                panel = BottomPanel(
                    title,
                    self._menu_search_subtitle(subtitle, query, searchable),
                    active_choices,
                    selected,
                    highlight_attr="match" if searchable else "accent",
                    selected_highlight_attr="match" if searchable else "selected",
                    replaces_prompt=replaces_prompt,
                    subtitle_max_lines=0 if replaces_prompt else 4,
                    subtitle_scroll=question_scroll,
                )
                viewport = self._draw_session(
                    stdscr,
                    session,
                    messages,
                    self._modal_prompt_text,
                    self._modal_prompt_cursor,
                    current_scroll,
                    bottom_panel=panel,
                    anchor_line=anchor_line,
                    prompt_notice=prompt_notice,
                    prompt_notice_role=prompt_notice_role,
                    file_references=self._modal_prompt_file_references,
                    image_attachments=self._modal_prompt_image_attachments,
                    pasted_spans=self._modal_prompt_pasted_spans,
                    hide_plan=replaces_prompt,
                )
                if viewport is not None:
                    current_scroll = viewport.scroll
                if replaces_prompt:
                    panel_viewport = self._bottom_panel_viewport(stdscr, panel)
                    question_page = max(1, len(panel_viewport.subtitle_lines))
                    max_question_scroll = max(0, panel_viewport.subtitle_total - question_page)
                    question_scroll = panel_viewport.subtitle_offset
                key = self._read_prompt_key(stdscr) if replaces_prompt else stdscr.get_wch()
                if isinstance(key, PromptPasteEvent):
                    continue
                if replaces_prompt:
                    mouse = self._question_mouse_event(key)
                    if mouse is not None:
                        _x, y, button_state = mouse
                        delta = self._question_mouse_scroll(button_state)
                        question_scroll = max(0, min(question_scroll + delta, max_question_scroll))
                        if not delta and self._is_left_click(button_state):
                            choice = self._bottom_panel_mouse_choice_at(stdscr, panel, y)
                            if choice is not None:
                                return active_choices[choice].value
                        continue
                if self._is_escape(key) or self._is_ctrl_c(key):
                    return None
                if self._is_shift_tab(key):
                    self._cycle_agent_mode(session)
                    continue
                if replaces_prompt and key in {curses.KEY_PPAGE, curses.KEY_NPAGE}:
                    delta = -question_page if key == curses.KEY_PPAGE else question_page
                    question_scroll = max(0, min(question_scroll + delta, max_question_scroll))
                    continue
                if key == curses.KEY_UP:
                    selected = max(0, selected - 1)
                elif key == curses.KEY_DOWN:
                    selected = min(len(active_choices) - 1, selected + 1)
                elif key == curses.KEY_PPAGE:
                    panel_viewport = self._bottom_panel_viewport(stdscr, panel)
                    page_size = max(1, len(panel_viewport.visible_indices))
                    selected = max(0, selected - page_size)
                elif key == curses.KEY_NPAGE:
                    panel_viewport = self._bottom_panel_viewport(stdscr, panel)
                    page_size = max(1, len(panel_viewport.visible_indices))
                    selected = min(len(active_choices) - 1, selected + page_size)
                elif key == curses.KEY_MOUSE:
                    choice = self._bottom_panel_mouse_choice(stdscr, panel)
                    if choice is not None and choice < len(active_choices):
                        return active_choices[choice].value
                elif self._is_enter(key):
                    if active_choices:
                        return active_choices[selected].value
                elif searchable and self._is_backspace(key):
                    query = query[:-1]
                    selected = 0
                elif searchable:
                    query, consumed = self._apply_menu_search_key(key, query)
                    if consumed:
                        selected = 0
        finally:
            if restore_nodelay:
                with suppress(curses.error):
                    stdscr.nodelay(True)

    def _project_bottom_menu(
        self,
        stdscr: CursesWindow,
        project: ProjectRecord,
        title: str,
        subtitle: str,
        choices: tuple[MenuChoice, ...],
        *,
        sessions: Sequence[SessionRecord],
        session_selected: int,
        scroll: int = 0,
        searchable: bool = False,
    ) -> str | None:
        if not choices:
            return None
        selected = 0
        current_scroll = scroll
        query = ""
        visible_sessions = list(sessions)
        with suppress(curses.error):
            stdscr.nodelay(False)
        while True:
            if not visible_sessions:
                visible_sessions = self._project_sessions(project.path)
            session_selected = (
                max(0, min(session_selected, len(visible_sessions) - 1)) if visible_sessions else 0
            )
            active_choices = (
                self._filter_menu_choices(choices, query) if searchable else choices
            )
            selected = max(0, min(selected, len(active_choices) - 1)) if active_choices else 0
            panel = BottomPanel(
                title,
                self._menu_search_subtitle(subtitle, query, searchable),
                active_choices,
                selected,
                highlight_attr="match" if searchable else "accent",
                selected_highlight_attr="match" if searchable else "selected",
            )
            current_scroll = self._draw_project(
                stdscr,
                project,
                visible_sessions,
                session_selected,
                current_scroll,
                self._modal_prompt_text,
                self._modal_prompt_cursor,
                "",
                "light",
                0,
                bottom_panel=panel,
                file_references=self._modal_prompt_file_references,
                pasted_spans=self._modal_prompt_pasted_spans,
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
                selected = min(len(active_choices) - 1, selected + 1)
            elif key == curses.KEY_PPAGE:
                panel_viewport = self._bottom_panel_viewport(stdscr, panel)
                page_size = max(1, len(panel_viewport.visible_indices))
                selected = max(0, selected - page_size)
            elif key == curses.KEY_NPAGE:
                panel_viewport = self._bottom_panel_viewport(stdscr, panel)
                page_size = max(1, len(panel_viewport.visible_indices))
                selected = min(len(active_choices) - 1, selected + page_size)
            elif key == curses.KEY_MOUSE:
                choice = self._bottom_panel_mouse_choice(stdscr, panel)
                if choice is not None and choice < len(active_choices):
                    return active_choices[choice].value
            elif self._is_enter(key):
                if active_choices:
                    return active_choices[selected].value
            elif searchable and self._is_backspace(key):
                query = query[:-1]
                selected = 0
            elif searchable:
                query, consumed = self._apply_menu_search_key(key, query)
                if consumed:
                    selected = 0

    def _run_overlay_menu(
        self,
        stdscr: CursesWindow,
        title: str,
        subtitle: str = "",
        choices: tuple[MenuChoice, ...] = (),
        footer: str = "Esc Back · ↑↓ Navigate · Enter Select",
        initial_value: str | None = None,
    ) -> str | None:
        """Run an overlay menu loop."""
        if not choices:
            return None
        selectable_indices = tuple(
            index for index, choice in enumerate(choices) if choice.selectable
        )
        if not selectable_indices:
            return None
        saved = (getattr(self, "_onboarding_selections", {})
                 if getattr(self, "_onboarding_step", None) is not None else {})
        menu_key = title
        if self._config_breadcrumbs and title in {"Config", "Manage Settings"}:
            saved = self._config_menu_selections
            menu_key = " › ".join(self._config_breadcrumbs)
        initial_value = saved.get(menu_key, initial_value)
        selected = next(
            (index for index in selectable_indices if choices[index].value == initial_value),
            selectable_indices[0],
        )
        while True:
            self._draw_overlay(
                stdscr,
                title=title,
                subtitle=subtitle,
                choices=choices,
                selected=selected,
                footer=footer,
            )
            key = stdscr.get_wch()
            if self._is_ctrl_c(key):
                self._onboarding_cancelled = True
                return None
            if self._is_escape(key):
                return None
            if key == curses.KEY_UP:
                selected = next(
                    (
                        index
                        for index in reversed(selectable_indices)
                        if index < selected
                    ),
                    selected,
                )
            elif key == curses.KEY_DOWN:
                selected = next(
                    (index for index in selectable_indices if index > selected),
                    selected,
                )
            elif self._is_enter(key) and choices[selected].selectable:
                saved[menu_key] = choices[selected].value
                return choices[selected].value

    def _run_overlay_text(
        self,
        stdscr: CursesWindow,
        title: str,
        subtitle: str = "",
        mask: bool = False,
        optional: bool = True,
        default: str = "",
        footer: str = "Esc Cancel · Enter Save",
    ) -> str | None:
        """Run an overlay single-line text input loop."""
        value = default
        cursor = len(value)
        while True:
            display_value = "*" * len(value) if mask else value
            self._draw_overlay(
                stdscr,
                title=title,
                subtitle=subtitle,
                input_value=display_value,
                input_cursor=cursor,
                footer=footer,
                show_input_cursor=True,
            )
            key = self._read_prompt_key(stdscr)
            if isinstance(key, PromptPasteEvent):
                value, cursor = self._insert_single_line_paste(value, cursor, key.text)
                continue
            if self._is_ctrl_c(key):
                self._onboarding_cancelled = True
                return None
            if self._is_escape(key):
                return None if optional else ""
            if self._is_enter(key) and (value or optional):
                return value
            if self._is_option_left(key):
                cursor = self._previous_prompt_word(value, cursor)
            elif self._is_option_right(key):
                cursor = self._next_prompt_word(value, cursor)
            elif self._is_option_delete(key):
                value, cursor = self._delete_previous_prompt_word(value, cursor)
            elif key == curses.KEY_LEFT:
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
                cursor += 1
        height, width = stdscr.getmaxyx()
        attr = self._attr("background")
        for y in range(height):
            self._add(stdscr, y, 0, " " * max(1, width), width, attr)

    def _prompt_text(
        self,
        stdscr: CursesWindow,
        title: str,
        label: str,
        mask: bool = False,
        optional: bool = True,
        default: str = "",
    ) -> str | None:
        return self._run_overlay_text(
            stdscr,
            title,
            label,
            mask=mask,
            optional=optional,
            default=default,
        )

    def _prompt_popover_text(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        title: str,
        label: str,
        mask: bool = False,
        optional: bool = True,
        default: str = "",
    ) -> str | None:
        value = default
        cursor = len(value)
        while True:
            self._draw_session(
                stdscr,
                session,
                self._read_message_lines(session.path),
                self._modal_prompt_text,
                self._modal_prompt_cursor,
                0,
                prompt_hint_suffix=" · Esc Cancel · Enter Save",
                file_references=self._modal_prompt_file_references,
                image_attachments=self._modal_prompt_image_attachments,
                pasted_spans=self._modal_prompt_pasted_spans,
            )
            display_value = "*" * len(value) if mask else value
            self._draw_text_popover(
                stdscr,
                title,
                label,
                display_value,
                cursor,
                "",
            )
            key = self._read_prompt_key(stdscr)
            if isinstance(key, PromptPasteEvent):
                value, cursor = self._insert_single_line_paste(value, cursor, key.text)
                continue
            if self._is_escape(key) or self._is_ctrl_c(key):
                return None if optional else ""
            if self._is_enter(key) and (value or optional):
                return value
            if self._is_option_left(key):
                cursor = self._previous_prompt_word(value, cursor)
            elif self._is_option_right(key):
                cursor = self._next_prompt_word(value, cursor)
            elif self._is_option_delete(key):
                value, cursor = self._delete_previous_prompt_word(value, cursor)
            elif key == curses.KEY_LEFT:
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
                cursor += 1

    def _prompt_project_popover_text(
        self,
        stdscr: CursesWindow,
        project: ProjectRecord,
        title: str,
        label: str,
        mask: bool = False,
        optional: bool = True,
        default: str = "",
        session_selected: int = 0,
        scroll: int = 0,
    ) -> str | None:
        value = default
        cursor = len(value)
        current_scroll = scroll
        while True:
            sessions = self._project_sessions(project.path)
            current_scroll = self._draw_project(
                stdscr,
                project,
                sessions,
                session_selected,
                current_scroll,
                self._modal_prompt_text,
                self._modal_prompt_cursor,
                "",
                "light",
                0,
                prompt_hint_suffix=" · Esc Cancel · Enter Save",
                file_references=self._modal_prompt_file_references,
                pasted_spans=self._modal_prompt_pasted_spans,
            )
            display_value = "*" * len(value) if mask else value
            self._draw_text_popover(
                stdscr,
                title,
                label,
                display_value,
                cursor,
                "",
            )
            key = self._read_prompt_key(stdscr)
            if isinstance(key, PromptPasteEvent):
                value, cursor = self._insert_single_line_paste(value, cursor, key.text)
                continue
            if self._is_escape(key) or self._is_ctrl_c(key):
                return None if optional else ""
            if self._is_enter(key) and (value or optional):
                return value
            if self._is_option_left(key):
                cursor = self._previous_prompt_word(value, cursor)
            elif self._is_option_right(key):
                cursor = self._next_prompt_word(value, cursor)
            elif self._is_option_delete(key):
                value, cursor = self._delete_previous_prompt_word(value, cursor)
            elif key == curses.KEY_LEFT:
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

    def _draw_text_popover(
        self,
        stdscr: CursesWindow,
        title: str,
        label: str,
        value: str,
        cursor: int,
        footer: str,
        *,
        replaces_prompt: bool = False,
        scroll: int = 0,
    ) -> tuple[int, int, int]:
        layout = self._prompt_layout(stdscr, "")
        height, width = stdscr.getmaxyx()
        panel_width = max(1, width - 4)
        left = 2
        end_y = height - 2 if replaces_prompt else layout.top_line - 1
        label_lines = (
            self._question_markdown_lines(label, max(1, panel_width - 4))
            if replaces_prompt else [TerminalLine(line) for line in self._panel_text_lines(
                label, max(1, panel_width - 4), preserve_line_breaks=True,
            )]
        ) or [TerminalLine("")]
        fixed_rows = 5 if footer else 4
        visible_rows = max(1, min(len(label_lines), end_y - 6 - fixed_rows + 1))
        max_scroll = max(0, len(label_lines) - visible_rows)
        scroll = max(0, min(scroll, max_scroll))
        start_y = max(6, end_y - visible_rows - fixed_rows + 1)
        for y in range(start_y, end_y + 1):
            self._clear_row(stdscr, y)
        self._add(stdscr, start_y, left, "─" * panel_width, panel_width, self._attr("accent"))
        heading = (
            f"{title} · {scroll + 1}–{scroll + visible_rows}/{len(label_lines)} · Scroll ↕"
            if max_scroll else title
        )
        self._add(stdscr, start_y + 1, left + 2, heading, panel_width - 4, self._attr("accent"))
        for offset, line in enumerate(label_lines[scroll : scroll + visible_rows]):
            if replaces_prompt:
                self._draw_question_markdown_line(
                    stdscr, start_y + 2 + offset, left + 2, line, panel_width - 4,
                )
                continue
            self._add(
                stdscr, start_y + 2 + offset, left + 2, line.text,
                panel_width - 4, self._attr("light"),
            )
        input_y = start_y + 3 + visible_rows
        input_offset = max(0, cursor - max(1, panel_width - 4) + 1)
        self._add(stdscr, input_y, left + 2, value[input_offset:], panel_width - 4, curses.A_NORMAL)
        self._paint_visual_cursor(
            stdscr, input_y, left + 2, value[input_offset:], cursor - input_offset,
        )
        if footer:
            footer_y = input_y + 1
            if max_scroll:
                footer = "Scroll/PgUp/PgDn · " + footer
            self._add(
                stdscr,
                footer_y,
                left,
                " " * panel_width,
                panel_width,
                self._attr("selected"),
            )
            self._add(stdscr, footer_y, left + 2, footer, panel_width - 4, self._attr("selected"))
        stdscr.refresh()
        return scroll, max_scroll, visible_rows

    def _message(self, stdscr: CursesWindow, title: str, message: str) -> None:
        body_lines = tuple(textwrap.wrap(message, width=60) or [message])
        while True:
            self._draw_overlay(
                stdscr,
                title=title,
                body_lines=body_lines,
                footer="Esc Back · Enter Continue",
            )
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key) or self._is_enter(key):
                return

    def _request_command_approval(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        request: CommandApprovalRequest,
        scroll: int = 0,
        anchor_line: int | None = None,
    ) -> ApprovalChoice:
        self._approval_memory_reason = ""
        automatic_choice = approval_choice_for_mode(self.agent_mode, request.evaluation)
        if automatic_choice is not None:
            return automatic_choice
        if request.evaluation is not None:
            return self._request_evaluated_command_approval(
                stdscr,
                session,
                request,
                scroll=scroll,
                anchor_line=anchor_line,
            )
        allowance_label = request.allowance_label or "matching commands"
        allowance_subject = request.allowance_subject or "this command"
        title = request.reason
        selected = self._bottom_menu(
            stdscr,
            session,
            title,
            request.command,
            (
                MenuChoice("Approve", ApprovalChoice.ALLOW.value, "Run this command once"),
                MenuChoice("Reject", ApprovalChoice.REJECT.value, "Do not run this command"),
                MenuChoice(
                    "Approve always",
                    ApprovalChoice.ALWAYS_ALLOW.value,
                    f"Trust {allowance_label} globally for {allowance_subject}",
                ),
                MenuChoice(
                    "Reject always, because ...",
                    ApprovalChoice.ALWAYS_REJECT.value,
                    f"Block {allowance_label} globally and save your reason as a memory",
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
            reason = self._request_approval_memory_reason(stdscr, request)
            if reason is None:
                return ApprovalChoice.REJECT
            self._approval_memory_reason = reason
            return ApprovalChoice.ALWAYS_REJECT
        return ApprovalChoice.REJECT

    def _request_evaluated_command_approval(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        request: CommandApprovalRequest,
        scroll: int = 0,
        anchor_line: int | None = None,
    ) -> ApprovalChoice:
        selected = 0
        current_scroll = scroll
        command_scroll = 0
        show_command = False
        choices = self._command_approval_choices(request)
        with suppress(curses.error):
            stdscr.nodelay(False)
        try:
            while True:
                automatic_choice = approval_choice_for_mode(self.agent_mode, request.evaluation)
                if automatic_choice is not None:
                    return automatic_choice
                panel = self._command_approval_panel(
                    request,
                    choices,
                    selected,
                    show_command=show_command,
                    command_scroll=command_scroll,
                )
                messages = self._read_message_lines(session.path)
                viewport = self._draw_session(
                    stdscr,
                    session,
                    messages,
                    "",
                    0,
                    current_scroll,
                    bottom_panel=panel,
                    anchor_line=anchor_line,
                )
                if viewport is not None:
                    current_scroll = viewport.scroll
                key = stdscr.get_wch()
                if self._is_escape(key) or self._is_ctrl_c(key):
                    return ApprovalChoice.REJECT
                if self._is_shift_tab(key):
                    self._cycle_agent_mode(session)
                    continue
                if key == curses.KEY_UP:
                    selected = max(0, selected - 1)
                elif key == curses.KEY_DOWN:
                    selected = min(len(choices) - 1, selected + 1)
                elif key == curses.KEY_PPAGE and show_command:
                    command_scroll = max(0, command_scroll - 5)
                elif key == curses.KEY_NPAGE and show_command:
                    command_scroll = min(
                        self._command_approval_command_max_scroll(stdscr, request),
                        command_scroll + 5,
                    )
                elif key == curses.KEY_PPAGE:
                    panel_viewport = self._bottom_panel_viewport(stdscr, panel)
                    page_size = max(1, len(panel_viewport.visible_indices))
                    selected = max(0, selected - page_size)
                elif key == curses.KEY_NPAGE:
                    panel_viewport = self._bottom_panel_viewport(stdscr, panel)
                    page_size = max(1, len(panel_viewport.visible_indices))
                    selected = min(len(choices) - 1, selected + page_size)
                elif key == curses.KEY_MOUSE:
                    mouse = self._approval_panel_mouse_event()
                    if mouse is None:
                        continue
                    _x, y, button_state = mouse
                    if self._is_left_click(button_state) and self._bottom_panel_subtitle_hit(
                        stdscr,
                        panel,
                        y,
                    ):
                        show_command = not show_command
                        command_scroll = 0
                        continue
                    if self._is_left_click(button_state):
                        choice = self._bottom_panel_mouse_choice_at(stdscr, panel, y)
                        if choice is not None:
                            return self._approval_choice_for_value(
                                stdscr,
                                request,
                                choices[choice].value,
                            )
                elif self._is_enter(key):
                    return self._approval_choice_for_value(
                        stdscr,
                        request,
                        choices[selected].value,
                    )
        finally:
            with suppress(curses.error):
                stdscr.nodelay(True)

    def _command_approval_choices(
        self,
        request: CommandApprovalRequest,
    ) -> tuple[MenuChoice, ...]:
        allowance_label = request.allowance_label or "matching commands"
        allowance_subject = request.allowance_subject or "this command"
        return (
            MenuChoice("Approve", ApprovalChoice.ALLOW.value, "Run this command once"),
            MenuChoice("Reject", ApprovalChoice.REJECT.value, "Do not run this command"),
            MenuChoice(
                "Approve always",
                ApprovalChoice.ALWAYS_ALLOW.value,
                f"Trust {allowance_label} globally for {allowance_subject}",
            ),
            MenuChoice(
                "Reject always, because ...",
                ApprovalChoice.ALWAYS_REJECT.value,
                f"Block {allowance_label} globally and save your reason as a memory",
            ),
        )

    def _command_approval_panel(
        self,
        request: CommandApprovalRequest,
        choices: tuple[MenuChoice, ...],
        selected: int,
        *,
        show_command: bool,
        command_scroll: int,
    ) -> BottomPanel:
        evaluation = request.evaluation
        risk = evaluation.risk if evaluation is not None else ""
        subtitle = request.command if show_command else (
            evaluation.description if evaluation is not None else request.command
        )
        return BottomPanel(
            request.reason,
            subtitle,
            choices,
            selected,
            title_attr="bold",
            title_suffix="(Click for description)" if show_command else "(Click for command)",
            title_prefix=self._command_risk_label(risk),
            title_prefix_attr=self._command_risk_attr(risk),
            subtitle_max_lines=5 if show_command else 4,
            subtitle_scroll=command_scroll if show_command else 0,
        )

    def _approval_choice_for_value(
        self,
        stdscr: CursesWindow,
        request: CommandApprovalRequest,
        value: str,
    ) -> ApprovalChoice:
        if value == ApprovalChoice.ALLOW.value:
            return ApprovalChoice.ALLOW
        if value == ApprovalChoice.ALWAYS_ALLOW.value:
            return ApprovalChoice.ALWAYS_ALLOW
        if value == ApprovalChoice.ALWAYS_REJECT.value:
            reason = self._request_approval_memory_reason(stdscr, request)
            if reason is None:
                return ApprovalChoice.REJECT
            self._approval_memory_reason = reason
            return ApprovalChoice.ALWAYS_REJECT
        return ApprovalChoice.REJECT

    def _request_approval_memory_reason(
        self,
        stdscr: CursesWindow,
        request: CommandApprovalRequest,
    ) -> str | None:
        del request
        prompt = getattr(self, "_prompt_multiline_text", None)
        if callable(prompt):
            reason = prompt(
                stdscr,
                "Reject Always",
                (
                    "Why should matching commands be rejected? "
                    "Ctrl+S saves this reason as a memory."
                ),
                optional=False,
            )
        else:
            reason = self._run_overlay_text(
                stdscr,
                "Reject Always",
                "Why should matching commands be rejected?",
                optional=False,
            )
        if reason is None or not reason.strip():
            return None
        return reason.strip()

    def _command_risk_label(self, risk: str) -> str:
        normalized = risk.strip().lower()
        if normalized == "low":
            return "Low Risk"
        if normalized == "medium":
            return "Medium Risk"
        if normalized == "high":
            return "High Risk"
        return ""

    def _command_risk_attr(self, risk: str) -> str:
        normalized = risk.strip().lower()
        if normalized == "low":
            return "ok"
        if normalized == "medium":
            return "warning"
        if normalized == "high":
            return "danger"
        return "accent"

    def _command_approval_command_max_scroll(
        self,
        stdscr: CursesWindow,
        request: CommandApprovalRequest,
    ) -> int:
        _, width = stdscr.getmaxyx()
        lines = self._panel_text_lines(request.command, max(1, width - 8))
        return max(0, len(lines) - 5)

    def _approval_panel_mouse_event(self) -> tuple[int, int, int] | None:
        with suppress(curses.error):
            _mouse_id, x, y, _z, button_state = curses.getmouse()
            return x, y, button_state
        return None

    def _question_mouse_event(self, key: str | int) -> tuple[int, int, int] | None:
        if key == curses.KEY_MOUSE:
            with suppress(curses.error):
                _, x, y, _, button_state = curses.getmouse()
                return x, y, button_state
            return None
        return self._raw_mouse_event(key)

    @staticmethod
    def _question_mouse_scroll(button_state: int) -> int:
        if button_state & getattr(curses, "BUTTON4_PRESSED", 0):
            return -3
        if button_state & getattr(curses, "BUTTON5_PRESSED", 0):
            return 3
        return 0

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
            MenuChoice(option.label, option.value, option.description) for option in request.options
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
            replaces_prompt=True,
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
        question_scroll = 0
        with suppress(curses.error):
            stdscr.nodelay(False)
        try:
            while True:
                viewport = self._draw_session(
                    stdscr,
                    session,
                    self._read_message_lines(session.path),
                    "",
                    0,
                    current_scroll,
                    anchor_line=anchor_line,
                    show_prompt_bar=False,
                    hide_plan=True,
                )
                if viewport is not None:
                    current_scroll = viewport.scroll
                question_scroll, max_question_scroll, question_page = self._draw_text_popover(
                    stdscr,
                    "Question",
                    request.question,
                    value,
                    cursor,
                    "Esc Cancel · Enter Submit",
                    replaces_prompt=True,
                    scroll=question_scroll,
                )
                key = self._read_prompt_key(stdscr)
                if isinstance(key, PromptPasteEvent):
                    value, cursor = self._insert_single_line_paste(value, cursor, key.text)
                    continue
                mouse = self._question_mouse_event(key)
                if mouse is not None:
                    delta = self._question_mouse_scroll(mouse[2])
                    question_scroll = max(0, min(question_scroll + delta, max_question_scroll))
                    continue
                if self._is_escape(key) or self._is_ctrl_c(key):
                    return QuestionResponse(
                        answered=False,
                        kind=request.kind,
                        cancelled=True,
                    )
                if key in {curses.KEY_UP, curses.KEY_DOWN, curses.KEY_PPAGE, curses.KEY_NPAGE}:
                    delta = {
                        curses.KEY_UP: -1,
                        curses.KEY_DOWN: 1,
                        curses.KEY_PPAGE: -question_page,
                        curses.KEY_NPAGE: question_page,
                    }[key]
                    question_scroll = max(0, min(question_scroll + delta, max_question_scroll))
                    continue
                if self._is_enter(key):
                    return QuestionResponse(
                        answered=True,
                        answer=value.strip(),
                        selected_label="",
                        kind=request.kind,
                    )
                if self._is_option_left(key):
                    cursor = self._previous_prompt_word(value, cursor)
                elif self._is_option_right(key):
                    cursor = self._next_prompt_word(value, cursor)
                elif self._is_option_delete(key):
                    value, cursor = self._delete_previous_prompt_word(value, cursor)
                elif key == curses.KEY_LEFT:
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
