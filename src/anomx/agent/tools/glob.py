"""Glob tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property


class GlobTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="glob",
            description="Find files by glob pattern inside the trusted workspace.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "pattern": {"type": "string", "description": "Glob pattern."},
                    "path": {"type": "string", "description": "Root path for the glob."},
                    "limit": {"type": "integer", "description": "Maximum matches."},
                },
                ["statement", "pattern", "path", "limit"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(self.name, arguments)
        pattern = str(arguments.get("pattern", "")).strip()
        if not pattern:
            return context.json_result({"error": "glob requires a pattern."})

        root_or_error = context.workspace_path(arguments.get("path") or ".")
        if isinstance(root_or_error, str):
            return context.json_result({"error": root_or_error})
        root = root_or_error
        if not root.is_dir():
            return context.json_result({"error": "Glob root is not a directory."})

        limit = min(context.positive_int(arguments.get("limit"), 200), 1_000)
        matches: list[str] = []
        try:
            for match in root.glob(pattern):
                resolved = match.resolve()
                if context.path_inside_workspace(resolved):
                    matches.append(str(resolved))
                if len(matches) >= limit:
                    break
        except (OSError, ValueError) as error:
            return context.json_result({"error": str(error)})
        return context.json_result({"matches": matches, "truncated": len(matches) >= limit})
