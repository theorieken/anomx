"""Read-file tool."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property
from anomx.agent.helpers.tool_results import MAX_INLINE_READ_CHARACTERS

DEFAULT_READ_LINES = 2_000
MAX_READ_LINES = 10_000
MAX_READ_CHARACTERS = 64_000


class ReadFileTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="read",
            description=(
                "Read a file inside the trusted workspace, up to 10,000 lines and "
                "64,000 characters per call (smaller for limited model contexts). "
                "Use next_read for continuation; stop when eof is true. "
                "Do not reread unchanged ranges. Use grep to locate specific fields."
            ),
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "path": {"type": "string", "description": "File path to read."},
                    "start_line": {
                        "type": "integer",
                        "description": "One-based start line.",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": (
                            "Maximum lines to select, up to 10,000; normally use 2,000. "
                            "Content is also character-bounded; follow next_read if present."
                        ),
                    },
                    "start_character": {
                        "type": "integer",
                        "description": (
                            "Zero-based character offset within the selected line range, "
                            "for long lines. Copy next_read exactly to continue a clipped range."
                        ),
                    },
                },
                ["statement", "path", "start_line", "max_lines"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(self.name, arguments)
        path_or_error = context.workspace_path(arguments.get("path"))
        if isinstance(path_or_error, str):
            return context.json_result({"error": path_or_error})
        path = path_or_error
        if not path.is_file():
            return context.json_result({"error": "Path is not a file."})

        start_line = context.positive_int(arguments.get("start_line"), 1)
        max_lines = min(
            context.positive_int(arguments.get("max_lines"), DEFAULT_READ_LINES), MAX_READ_LINES
        )
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as error:
            return context.json_result({"error": str(error)})

        start_index = max(0, start_line - 1)
        selected = lines[start_index : start_index + max_lines]
        text = "\n".join(selected)
        offset = max(0, context.positive_int(arguments.get("start_character"), 0))
        if offset > len(text):
            return context.json_result({
                "error": "start_character exceeds the selected range. Use the previous next_read.",
                "path": str(path),
                "selected_characters": len(text),
            })
        character_limit = min(
            MAX_READ_CHARACTERS, max(8_000, context.runtime.maximum_context_tokens() // 4)
        )
        content = text[offset : offset + character_limit]
        # Escaping control characters can expand a short string sixfold in JSON.
        # Find the largest fitting prefix rather than repeatedly halving the page.
        encoded_limit = min(character_limit * 3 // 2, MAX_INLINE_READ_CHARACTERS - 16_000)
        low, high = 0, len(content)
        while low < high:
            middle = (low + high + 1) // 2
            if len(json.dumps(content[:middle], ensure_ascii=False)) <= encoded_limit:
                low = middle
            else:
                high = middle - 1
        content = content[:low]
        truncated = offset + len(content) < len(text)
        next_character = offset + len(content) if truncated else None
        next_line = (
            start_line + len(selected)
            if not truncated and start_index + len(selected) < len(lines)
            else None
        )
        next_read = None
        if truncated or next_line is not None:
            next_read = {
                "path": str(path),
                "start_line": start_line if truncated else next_line,
                "max_lines": max_lines,
                "start_character": next_character if truncated else 0,
            }
        payload = {
            "path": str(path),
            "start_line": start_line,
            "line_count": len(selected),
            "total_lines": len(lines),
            "content": content,
            "start_character": offset,
            "truncated": truncated,
            "next_character": next_character,
            "next_line": next_line,
            "next_read": next_read,
            "eof": next_read is None,
        }
        # Detect alternating ranges as well as consecutive retries. Hash the
        # selected text and line count so changed content or file length resets it.
        signature = (
            str(path), start_line, max_lines, offset,
            hashlib.sha256(f"{len(lines)}\n{text}".encode()).hexdigest(),
        )
        history = context.runtime.file_read_history
        if signature in history:
            payload["repeated_read"] = True
            payload["read_hint"] = (
                "This unchanged range was already returned. Do not repeat this read. "
                + (
                    "Continue with next_read, including start_character, or use grep "
                    "to locate the fields you need."
                    if next_read is not None else
                    "End of file reached; there are no further pages. Use the information "
                    "already read, or grep earlier lines for a specific missing field."
                )
            )
        elif next_read is None:
            payload["read_hint"] = "End of file reached; there are no further pages."
        else:
            payload["read_hint"] = (
                "For more content, call read with next_read's arguments and a statement. "
                "Do not guess offsets or restart this range."
            )
        history.append(signature)
        return context.json_result(payload)
