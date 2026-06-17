"""Read-file tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property


class ReadFileTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="read",
            description="Read a file inside the trusted workspace.",
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
                        "description": "Maximum lines to return.",
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
        max_lines = min(context.positive_int(arguments.get("max_lines"), 200), 1_000)
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as error:
            return context.json_result({"error": str(error)})

        start_index = max(0, start_line - 1)
        selected = lines[start_index : start_index + max_lines]
        return context.json_result(
            {
                "path": str(path),
                "start_line": start_line,
                "line_count": len(selected),
                "total_lines": len(lines),
                "content": "\n".join(selected),
            }
        )
