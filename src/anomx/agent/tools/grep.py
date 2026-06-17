"""Grep tool."""

from __future__ import annotations

import re
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property


class GrepTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="grep",
            description="Search file text inside the trusted workspace.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "pattern": {
                        "type": "string",
                        "description": "Regex or literal search text.",
                    },
                    "path": {"type": "string", "description": "File or directory path."},
                    "include": {"type": "string", "description": "File glob filter."},
                    "limit": {"type": "integer", "description": "Maximum matches."},
                },
                ["statement", "pattern", "path", "include", "limit"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(self.name, arguments)
        pattern = str(arguments.get("pattern", "")).strip()
        if not pattern:
            return context.json_result({"error": "grep requires a pattern."})

        path_or_error = context.workspace_path(arguments.get("path") or ".")
        if isinstance(path_or_error, str):
            return context.json_result({"error": path_or_error})
        root = path_or_error
        include = str(arguments.get("include", "*")).strip() or "*"
        limit = min(context.positive_int(arguments.get("limit"), 100), 1_000)
        with suppress(re.error):
            regex = re.compile(pattern)
            return context.json_result(
                {
                    "matches": self._regex_matches(context, root, regex, include, limit),
                    "pattern": pattern,
                }
            )

        needle = pattern.lower()
        return context.json_result(
            {
                "matches": self._literal_matches(context, root, needle, include, limit),
                "pattern": pattern,
            }
        )

    def _regex_matches(
        self,
        context: ToolExecutionContext,
        root: Path,
        regex: re.Pattern[str],
        include: str,
        limit: int,
    ) -> list[dict[str, object]]:
        matches: list[dict[str, object]] = []
        for path in self._iter_files(context, root, include):
            for line_number, line in self._iter_file_lines(path):
                if regex.search(line):
                    matches.append(
                        {"path": str(path), "line": line_number, "text": line.rstrip()}
                    )
                    if len(matches) >= limit:
                        return matches
        return matches

    def _literal_matches(
        self,
        context: ToolExecutionContext,
        root: Path,
        needle: str,
        include: str,
        limit: int,
    ) -> list[dict[str, object]]:
        matches: list[dict[str, object]] = []
        for path in self._iter_files(context, root, include):
            for line_number, line in self._iter_file_lines(path):
                if needle in line.lower():
                    matches.append(
                        {"path": str(path), "line": line_number, "text": line.rstrip()}
                    )
                    if len(matches) >= limit:
                        return matches
        return matches

    def _iter_files(
        self,
        context: ToolExecutionContext,
        root: Path,
        include: str,
    ) -> Iterable[Path]:
        if root.is_file():
            yield root
            return
        for path in root.rglob(include):
            if path.is_file() and context.path_inside_workspace(path.resolve()):
                yield path

    def _iter_file_lines(self, path: Path) -> Iterable[tuple[int, str]]:
        try:
            with path.open(encoding="utf-8", errors="replace") as handle:
                yield from enumerate(handle, start=1)
        except OSError:
            return
