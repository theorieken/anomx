"""Transcript message rendering, wrapping, and expansion helpers."""

from __future__ import annotations

import curses
import hashlib
import re
import textwrap
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any

from anomx.agent.base.backends import strip_thinking_tags
from anomx.agent.helpers.terminal import (
    BOLD_START,
    CODE_END,
    CODE_START,
    EMPHASIS_MARKERS,
    ITALIC_START,
    TerminalLine,
    markdown_to_terminal_rendered_lines,
    terminal_cell_spans,
)
from anomx.agent.helpers.terminal_colors import (
    blend_color,
    read_terminal_theme,
    terminal_color_index,
)
from anomx.agent.store import (
    SessionRecord,
)
from anomx.agent.ui.constants import (
    ACTIVITY_WAVE_SHADE_COUNT,
    TABLE_BORDER_CHARS,
)
from anomx.agent.ui.models import (
    CursesWindow,
    MessageLine,
)
from anomx.agent.ui.wave import activity_wave_strengths


class MessagesComponentMixin:
    """Transcript message rendering, wrapping, and expansion helpers."""

    INLINE_CODE_MARKER_RE = re.compile(f"{re.escape(CODE_START)}|{re.escape(CODE_END)}")

    def _question_markdown_lines(self, text: str, width: int) -> list[TerminalLine]:
        return markdown_to_terminal_rendered_lines(
            text, width, wrap_long_words=True, style_emphasis=True,
        )

    def _draw_question_markdown_line(
        self, stdscr: CursesWindow, y: int, x: int, line: TerminalLine, width: int,
    ) -> None:
        if line.style in {"table_header", "table_row"}:
            text = line.text.replace(CODE_START, "").replace(CODE_END, "")
            self._draw_table_line(stdscr, y, x, text, width, line.style)
        elif line.style == "code":
            self._add(stdscr, y, x, line.text, width, self._line_attr("code"))
        else:
            markers = {CODE_START: CODE_END, **EMPHASIS_MARKERS}
            active: set[str] = set()
            column = 0
            pattern = "([" + "".join(markers) + "".join(markers.values()) + "])"
            for part in re.split(pattern, line.text):
                if part in markers:
                    active.add(part)
                elif part in markers.values():
                    active.discard(next(start for start, end in markers.items() if end == part))
                elif part:
                    attr = self._line_attr("code" if CODE_START in active else line.style)
                    if BOLD_START in active:
                        attr |= curses.A_BOLD
                    if ITALIC_START in active:
                        attr |= getattr(curses, "A_ITALIC", curses.A_UNDERLINE)
                    self._add(stdscr, y, x + column, part, width - column, attr)
                    column += sum(cells for _, _, cells in terminal_cell_spans(part, width))

    def _line_attr(self, role: str) -> int:
        if role in {"user", "pinned_user"}:
            return self._attr("user")
        if role == "user_box":
            return self._attr("bold")
        if role in {"meta_accent", "context"}:
            return self._attr("accent")
        if role in {"meta", "thought", "tool", "work_summary", "work_active", "approved", "notice"}:
            return self._attr("light")
        if role == "code":
            return self._attr("accent")
        if role == "bold":
            return self._attr("bold")
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
        expanded_users = self._expanded_pinned_users_key()
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
                and rendered_cache[4] == expanded_users
                and rendered_cache[5] == width
                and rendered_cache[6] == working_key
            ):
                return rendered_cache[7]
            rendered_messages = self._messages_with_working_status(messages, working_text)
            rendered = self._render_messages(rendered_messages, width)
            self._rendered_message_cache[session.path] = (
                cache_key[0],
                cache_key[1],
                expanded_turns,
                expanded_lines,
                expanded_users,
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
        activity_index = self._latest_activity_message_index(messages)
        if activity_index is not None:
            activity = messages[activity_index]
            model_phase = working_text in {"Thinking", "Writing"}
            if self._is_expandable_work_role(activity.role) and working_text not in {"Executing", activity.text}:
                # Extended mode has no section heading to carry the live phase.
                return [*messages, MessageLine("working", working_text), MessageLine("meta", "")]
            active_messages = list(messages)
            text = activity.text
            if working_text != "Executing":
                text = working_text + (" · collapse" if text.endswith(" · collapse") else "")
            active_messages[activity_index] = replace(activity, text=text, activity_wave=True)
            if (
                not model_phase
                and activity.role == "work_active"
                and activity.text.endswith(" · collapse")
                and activity_index > 0
                and messages[activity_index - 1].role == "tool"
                and (working_text == "Executing" or messages[activity_index - 1].text == working_text)
            ):
                active_messages[activity_index - 1] = replace(
                    messages[activity_index - 1], activity_wave=True,
                )
            return active_messages
        return [*messages, MessageLine("working", working_text), MessageLine("meta", "")]

    def _latest_activity_message_index(self, messages: list[MessageLine]) -> int | None:
        """Choose one visible tool label, stopping at the current request boundary."""
        # Streaming text and intermediate messages do not finish the latest section.
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.role in {"user", "work_summary", "context"}:
                break
            if message.role == "work_active":
                return index
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.role in {"agent", "agent_intermediate", "user"}:
                return None
            if message.role == "context":
                return index if message.activity_wave else None
            if message.role == "work_summary":
                return None
            if message.role == "work_active" and message.meta:
                return index
            if self._is_expandable_work_role(message.role):
                return index
        return None

    def _messages_with_transient_state(
        self,
        messages: list[MessageLine],
        active_turn_elapsed: float | None,
        streaming_text: str,
    ) -> list[MessageLine]:
        del active_turn_elapsed
        if not streaming_text:
            return messages
        return [*messages, MessageLine("agent", streaming_text)]

    def _draw_working_line(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        width: int,
        frame: int,
    ) -> None:
        attr = self._attr("light")
        self._add(stdscr, y, x, text, width, attr)
        self._draw_activity_wave(stdscr, y, x, text, width, frame, attr)

    def _draw_activity_wave(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        width: int,
        frame: int,
        attr: int,
        *,
        primary: bool = False,
    ) -> None:
        """Move a soft cosine-shaped brightness crest through the text."""
        label = text.removesuffix(" · collapse")
        spans = terminal_cell_spans(label, max(0, width))
        palette = self._activity_wave_palettes.get(bool(attr & curses.A_REVERSE), ())
        strengths = activity_wave_strengths(spans, frame)
        for (column, glyph, _cells), strength in zip(spans, strengths, strict=True):
            if palette:
                if primary and strength <= 0:
                    continue
                shade_strength = 1 - strength if primary else strength
                shade = palette[round(shade_strength * (len(palette) - 1))]
                wave_attr = (attr & ~(curses.A_COLOR | curses.A_DIM)) | shade
            elif strength >= 0.5:
                wave_attr = attr ^ curses.A_DIM
            else:
                continue
            self._add(stdscr, y, x + column, glyph, len(glyph), wave_attr)

    def _configure_activity_wave_colors(self, default_fg: int, default_bg: int) -> None:
        """Allocate shades without changing the terminal's color definitions."""
        first_pair = 12  # Pairs 1–11 belong to the existing CLI palette.
        color_count = getattr(curses, "COLORS", 0)
        if (
            color_count < 256
            or getattr(curses, "COLOR_PAIRS", 0) < first_pair + 2 * ACTIVITY_WAVE_SHADE_COUNT
        ):
            return
        theme = read_terminal_theme()
        if theme is None:
            return
        foreground, background = theme
        for reverse in (False, True):
            shades = []
            with suppress(curses.error):
                for index in range(ACTIVITY_WAVE_SHADE_COUNT):
                    progress = index / (ACTIVITY_WAVE_SHADE_COUNT - 1)
                    color = (
                        blend_color(foreground, background, 1 - 0.3 * progress)
                        if reverse else blend_color(background, foreground, 0.55 + 0.3 * progress)
                    )
                    color_index = terminal_color_index(color, color_count)
                    pair = first_pair + int(reverse) * ACTIVITY_WAVE_SHADE_COUNT + index
                    curses.init_pair(
                        pair, default_fg if reverse else color_index,
                        color_index if reverse else default_bg,
                    )
                    shades.append(curses.color_pair(pair))
                self._activity_wave_palettes[reverse] = tuple(shades)

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

    def _draw_user_box_line(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        width: int,
    ) -> None:
        border_attr = self._attr("bold")
        content_attr = curses.A_NORMAL
        self._add(stdscr, y, x, text.ljust(max(0, width)), width, content_attr)
        for offset, character in enumerate(text[:width]):
            if character in {"╭", "╮", "╰", "╯", "│", "─"}:
                self._add(stdscr, y, x + offset, character, 1, border_attr)

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

    def _draw_line_with_inline_code(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        width: int,
        default_attr: int,
    ) -> None:
        if CODE_START not in text:
            self._add(stdscr, y, x, text, width, default_attr)
            return
        parts = self.INLINE_CODE_MARKER_RE.split(text)
        cursor = x
        for i, part in enumerate(parts):
            if not part:
                continue
            remaining = width - (cursor - x)
            if remaining <= 0:
                break
            attr = self._attr("accent") if i % 2 == 1 else default_attr
            self._add(stdscr, y, cursor, part, remaining, attr)
            cursor += len(part)

    def _read_message_lines(self, session_path: Path) -> list[MessageLine]:
        cache_key = self._session_cache_key(session_path)
        expanded_turns = self._expanded_work_turns_key()
        streaming_summary = None
        if cache_key is not None:
            cached = self._message_line_cache.get(session_path)
            if (
                cached is not None
                and cached[0] == cache_key[0]
                and cached[1] == cache_key[1]
                and cached[2] == expanded_turns
                and cached[4] == streaming_summary
            ):
                return cached[3]
        # Streaming can change the folded transcript without changing the log.
        self._rendered_message_cache.pop(session_path, None)

        lines: list[MessageLine] = []
        turn_segments: dict[str, list[list[MessageLine]]] = {}
        turn_segment_by_key: dict[str, list[MessageLine]] = {}
        turn_summaries: dict[str, str] = {}
        current_turn_id = ""
        current_segment_key = ""
        seen_context_ids: set[str] = set()

        def append_turn_line(turn_id: str, line: MessageLine) -> None:
            nonlocal current_segment_key, current_turn_id
            # Messages are transcript boundaries, even when they belong to the
            # same runtime turn. Only adjacent tool/activity lines share a row.
            if not turn_id or not self._is_expandable_work_role(line.role):
                lines.append(line)
                current_turn_id = ""
                current_segment_key = ""
                return
            if current_turn_id != turn_id or not current_segment_key:
                segments = turn_segments.setdefault(turn_id, [])
                current_segment_key = turn_id if not segments else f"{turn_id}:{len(segments)}"
                current_turn_id = turn_id
                segments.append([])
                turn_segment_by_key[current_segment_key] = segments[-1]
                lines.append(MessageLine("__turn_placeholder__", turn_id, current_segment_key))
            turn_segment_by_key[current_segment_key].append(line)

        def append_turn_summary(turn_id: str, message: str) -> None:
            if not turn_id:
                lines.append(MessageLine("work_summary", message))
                return
            turn_summaries[turn_id] = message

        events = self._session_events(session_path)
        displayed_context_ids = {
            str(payload.get("id"))
            for event in events
            if isinstance(payload := event.get("payload"), dict)
            and payload.get("type", event.get("type")) == "context_activity_display"
        }
        context_activity_states = {
            str(payload.get("id")): payload
            for event in events
            if isinstance(payload := event.get("payload"), dict)
            and payload.get("type", event.get("type")) == "context_activity"
        }
        for event_index, event in enumerate(events):
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            event_type = str(
                payload.get("type") if event.get("type") == "event_msg" else event.get("type")
            )
            message = str(payload.get("message", "")).strip()
            if event_type in {"user_message", "skill_invocation"} and message:
                turn_id = str(payload.get("turn_id", ""))
                if event_type == "user_message" and payload.get("intermediate") and turn_id:
                    append_turn_line(
                        turn_id,
                        MessageLine(
                            "user",
                            message,
                            turn_id,
                            expansion_key=self._session_user_message_key(event_index),
                        ),
                    )
                    continue
                current_turn_id = ""
                current_segment_key = ""
                lines.append(
                    MessageLine(
                        "user",
                        message,
                        expansion_key=self._session_user_message_key(event_index),
                    )
                )
            elif event_type == "agent_message" and message:
                turn_id = str(payload.get("turn_id", ""))
                role = "agent_intermediate" if payload.get("intermediate") else "agent"
                visible_message = strip_thinking_tags(message)
                if visible_message:
                    append_turn_line(turn_id, MessageLine(role, visible_message, turn_id))
            elif event_type in {"context_activity", "context_activity_display"}:
                activity_id = str(payload.get("id") or event_index)
                # UI events preserve callback order even when the isolated runtime
                # wrote its durable activity before the UI drained earlier tools.
                if event_type == "context_activity" and activity_id in displayed_context_ids:
                    continue
                if activity_id in seen_context_ids:
                    continue
                seen_context_ids.add(activity_id)
                payload = {**payload, **context_activity_states.get(activity_id, {})}
                if payload.get("kind") == "check" or payload.get("model_requests") == 0 or (
                    payload.get("status") == "completed" and payload.get("changed") is False
                ):
                    if payload.get("status") == "running":
                        append_turn_line(
                            "", MessageLine(
                                "context", "Checking context …", str(payload.get("turn_id", "")),
                                activity_wave=True,
                            ),
                        )
                    continue
                label = "Context Optimization"
                status = str(payload.get("status") or "completed")
                if status == "running":
                    label += " …"
                elif status == "failed":
                    label += " · could not reduce context"
                elif payload.get("changed"):
                    before = int(payload.get("tokens_before") or 0)
                    after = int(payload.get("tokens_after") or 0)
                    label += f" · {before:,} → {after:,} tokens"
                else:
                    label += " · context retained"
                line = MessageLine(
                    "context", label, str(payload.get("turn_id", "")),
                    activity_wave=status == "running",
                )
                append_turn_line("", line)
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
                if role == "tool" and message in {"Thinking", "Writing", "Executing"} and not payload.get("command"):
                    # Older clients recorded these transient phases as tools.
                    continue
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
        displayed_work_turns: set[str] = set()
        latest_activity_index = -1
        for index in range(len(lines) - 1, -1, -1):
            if lines[index].role in {"user", "agent", "context"}:
                break
            if lines[index].role == "__turn_placeholder__":
                latest_activity_index = index
                break
        for line_index, line in enumerate(lines):
            work_turn_id = (
                line.text if line.role == "__turn_placeholder__"
                else line.meta if line.role in {"agent_intermediate", "user", "context"} else ""
            )
            if self.work_visualization == "default" and work_turn_id in turn_summaries:
                work_key = f"work:{work_turn_id}"
                work_expanded = work_key in self._expanded_work_turns
                if work_turn_id not in displayed_work_turns:
                    displayed_work_turns.add(work_turn_id)
                    rendered_lines.append(MessageLine(
                        "work_summary",
                        turn_summaries[work_turn_id] + (" · collapse" if work_expanded else ""),
                        work_key,
                    ))
                if not work_expanded:
                    continue
            if line.role != "__turn_placeholder__":
                rendered_lines.append(line)
                continue
            turn_id = line.text
            segment_key = line.meta
            summary = turn_summaries.get(turn_id)
            if self.work_visualization == "default":
                segment_lines = turn_segment_by_key.get(segment_key, [])
                latest_statement = next(
                    (entry.text for entry in reversed(segment_lines) if entry.text.strip()),
                    "Thinking",
                )
                active = not summary and line_index == latest_activity_index
                group_summary = self._tool_section_summary(segment_lines)
                label = latest_statement if active else group_summary
                expanded = segment_key in self._expanded_work_turns
                if expanded:
                    rendered_lines.extend(segment_lines)
                rendered_lines.append(MessageLine(
                    "work_active" if active else "work_summary",
                    self._single_line_work_text(label) + (" · collapse" if expanded else ""),
                    segment_key,
                    summary_text=group_summary + (" · collapse" if expanded else ""),
                ))
            else:
                rendered_lines.extend(turn_segment_by_key.get(segment_key, []))
        if cache_key is not None:
            self._message_line_cache[session_path] = (
                cache_key[0],
                cache_key[1],
                expanded_turns,
                rendered_lines,
                streaming_summary,
            )
        return rendered_lines

    def _tool_section_summary(self, lines: list[MessageLine]) -> str:
        """Describe a finished tool block without reusing its last activity."""
        counts: dict[str, int] = {}
        for line in lines:
            if line.role == "thought":
                label = "Thought"
            elif match := re.match(r"^Tool:\s*(\S+)", line.detail_body):
                label = match[1].replace("_", " ").capitalize()
            else:
                label = self._single_line_work_text(line.text)
            if label:
                counts[label] = counts.get(label, 0) + 1
        return ", ".join(
            f"{count}× {label}" if count > 1 else label for label, count in counts.items()
        ) or "Activity"

    def _session_events(self, session_path: Path) -> list[dict[str, Any]]:
        cache_key = self._session_cache_key(session_path)
        if cache_key is None:
            return self.home.read_session_events(session_path)
        cached = self._session_event_cache.get(session_path)
        if cached is not None and cached[0] == cache_key[0] and cached[1] == cache_key[1]:
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
        return (self.work_visualization, *sorted(self._expanded_work_turns))

    def _expanded_work_lines_key(self) -> tuple[str, ...]:
        return tuple(sorted(self._expanded_work_lines))

    def _expanded_pinned_users_key(self) -> tuple[str, ...]:
        return tuple(sorted(self._expanded_pinned_users))

    def _message_display_parts(
        self,
        role: str,
        message: str,
        command: str = "",
    ) -> tuple[str, str, str]:
        if role == "thought":
            return message, command, "Thought"
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
            command, separator, reason = message.removeprefix("Blocked command: ").partition(" · ")
            detail_title = f"Reason: {reason.strip()}" if separator else ""
            detail_command = command.strip()
            return f"Blocked: {detail_command}", detail_command, detail_title

        return message, detail_command, ""

    def _session_work_line_key(self, role: str, turn_id: str, event_index: int) -> str:
        if not self._is_expandable_work_role(role):
            return ""
        namespace = turn_id or "session"
        return f"{namespace}:{event_index}"

    def _session_user_message_key(self, event_index: int) -> str:
        return f"user:{event_index}"

    def _render_messages(self, messages: list[MessageLine], width: int) -> list[MessageLine]:
        rendered: list[MessageLine] = []
        previous_kind: str | None = None
        for message in messages:
            kind = self._message_kind(message.role)
            if rendered and previous_kind is not None and kind != previous_kind:
                rendered.append(MessageLine("meta", ""))
            message_start = len(rendered)
            if message.role == "user":
                rendered.extend(self._render_user_message(message, width))
            elif message.role in {"work_active", "work_summary"}:
                rendered.append(MessageLine(
                    message.role,
                    self._ellipsized_statement_text(message.text, width),
                    message.meta,
                ))
            elif self._is_expandable_work_role(message.role):
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
                            message.expansion_key,
                            message.detail_title,
                            message.detail_body,
                        )
                    )
            if message.activity_wave and len(rendered) > message_start:
                # Expanded tool details keep the wave on the first title row.
                wave_index = message_start + (rendered[message_start].role == "work_box")
                if wave_index < len(rendered):
                    rendered[wave_index] = replace(rendered[wave_index], activity_wave=True)
            previous_kind = kind
        return rendered

    def _render_user_message(self, message: MessageLine, width: int) -> list[MessageLine]:
        safe_width = max(20, width)
        expansion_key = message.expansion_key or self._fallback_user_line_key(message)
        if expansion_key in self._expanded_pinned_users:
            return self._expanded_user_box_lines(message, safe_width, expansion_key)

        display_text = self._single_line_work_text(message.text)
        toggle_text = " Expand"
        available = max(1, safe_width - len(toggle_text))
        collapsed_text = self._ellipsized_statement_text(display_text, available)
        return [
            MessageLine(
                "user",
                f"{collapsed_text.ljust(available)}{toggle_text}",
                message.meta,
                expansion_key,
                message.detail_title,
                message.detail_body,
            )
        ]

    def _expanded_user_box_lines(
        self,
        message: MessageLine,
        width: int,
        expansion_key: str,
    ) -> list[MessageLine]:
        safe_width = max(20, width)
        inner_width = max(1, safe_width - 4)
        collapse_text = " Collapse"
        top_width = max(1, safe_width - 2)
        top_border = "─" * max(1, top_width - len(collapse_text))
        lines = [
            MessageLine(
                "user_box",
                f"╭{top_border}{collapse_text}╮",
                message.meta,
                expansion_key,
            )
        ]
        for content_line in self._work_box_content_lines(message.text, inner_width):
            content = content_line[:inner_width].ljust(inner_width)
            lines.append(MessageLine("user_box", f"│ {content} │", message.meta, expansion_key))
        border = "─" * max(1, safe_width - 2)
        lines.append(MessageLine("user_box", f"╰{border}╯", message.meta, expansion_key))
        return lines

    def _terminal_line_role(self, fallback_role: str, style: str) -> str:
        if style in {"table_border", "table_header", "table_row"}:
            return style
        if style == "bold":
            return "bold"
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
        return role in {"thought", "tool", "approved", "forbidden"}

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

    def _fallback_user_line_key(self, message: MessageLine) -> str:
        digest = hashlib.sha1(message.text.encode("utf-8", errors="replace")).hexdigest()
        return f"user:{message.meta}:{digest}"

    def _message_kind(self, role: str) -> str:
        if role == "context":
            return "context"
        if role == "user":
            return "user"
        if role == "agent":
            return "agent"
        if role == "agent_intermediate":
            return "agent"
        return "working"
