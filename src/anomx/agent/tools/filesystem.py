"""Read-only filesystem exploration tools."""

from __future__ import annotations

from anomx.agent.base.tools import BaseTool, object_schema, statement_property


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
