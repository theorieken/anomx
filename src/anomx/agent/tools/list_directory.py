"""List-directory tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property


class ListDirectoryTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="list",
            description="List a directory inside the trusted workspace.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "path": {"type": "string", "description": "Directory path to list."},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum entries to return.",
                    },
                },
                ["statement", "path", "limit"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(self.name, arguments)
        path_or_error = context.workspace_path(arguments.get("path") or ".")
        if isinstance(path_or_error, str):
            return context.json_result({"error": path_or_error})
        path = path_or_error
        if not path.is_dir():
            return context.json_result({"error": "Path is not a directory."})
        try:
            entries = sorted(
                path.iterdir(),
                key=lambda item: (not item.is_dir(), item.name.lower()),
            )
        except OSError as error:
            return context.json_result({"error": str(error)})

        limit = min(context.positive_int(arguments.get("limit"), 200), 1_000)
        return context.json_result(
            {
                "path": str(path),
                "entries": [
                    {
                        "name": entry.name,
                        "path": str(entry),
                        "kind": "directory" if entry.is_dir() else "file",
                    }
                    for entry in entries[:limit]
                ],
                "truncated": len(entries) > limit,
            }
        )
