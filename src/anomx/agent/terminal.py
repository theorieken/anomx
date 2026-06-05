"""Terminal rendering helpers for Anomx agent transcripts."""

from __future__ import annotations

import re
import textwrap

MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")


def markdown_to_terminal_lines(markdown: str, width: int) -> list[str]:
    """Convert a Markdown fragment into wrapped terminal transcript lines."""

    lines: list[str] = []
    in_code_block = False
    safe_width = max(20, width)

    for raw_line in markdown.splitlines() or [""]:
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            lines.extend(_wrap_preserving_indent(f"  {line}", safe_width))
            continue

        if not stripped:
            lines.append("")
            continue

        normalized = _normalize_markdown_line(stripped)
        lines.extend(_wrap_preserving_indent(normalized, safe_width))

    while lines and lines[-1] == "":
        lines.pop()
    return lines or [""]


def _normalize_markdown_line(line: str) -> str:
    heading = line.lstrip("#").strip() if line.startswith("#") else line
    linked = MARKDOWN_LINK_RE.sub(r"\1 (\2)", heading)
    code = INLINE_CODE_RE.sub(r"\1", linked)
    without_emphasis = code.replace("**", "").replace("__", "").replace("*", "")
    return without_emphasis.replace("–", "-")


def _wrap_preserving_indent(line: str, width: int) -> list[str]:
    indent_width = len(line) - len(line.lstrip())
    indent = " " * indent_width
    content = line.lstrip()
    bullet_prefix = _bullet_prefix(content)
    subsequent_indent = indent + (" " * len(bullet_prefix) if bullet_prefix else "")
    return textwrap.wrap(
        indent + content,
        width=width,
        subsequent_indent=subsequent_indent,
        replace_whitespace=False,
        drop_whitespace=True,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [indent]


def _bullet_prefix(content: str) -> str:
    for prefix in ("- ", "* ", "+ "):
        if content.startswith(prefix):
            return prefix
    if len(content) >= 3 and content[0].isdigit() and content[1:3] == ". ":
        return content[:3]
    return ""
