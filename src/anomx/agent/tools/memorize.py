"""Memorize tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property
from anomx.agent.memories import (
    MemoryKind,
    create_memory_record,
    fallback_memory_summary,
    fallback_memory_title,
    write_memory,
)


class MemorizeTool(BaseTool):
    """Store a durable memory for future agent turns."""

    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="memorize",
            description=(
                "Store a durable memory in the local Anomx brain for future turns. "
                "Use this for stable user preferences, durable constraints, or important "
                "project facts that should be remembered later."
            ),
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "content": {
                        "type": "string",
                        "description": "The exact memory content to preserve.",
                    },
                    "context": {
                        "type": "object",
                        "description": "Small JSON context explaining why this was memorized.",
                        "additionalProperties": True,
                    },
                    "title": {
                        "type": ["string", "null"],
                        "description": "Optional compact title; leave null to infer one.",
                    },
                    "summary": {
                        "type": ["string", "null"],
                        "description": "Optional one-sentence summary; leave null to infer one.",
                    },
                },
                ["statement", "content", "context", "title", "summary"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(self.name, arguments)
        content = str(arguments.get("content") or "").strip()
        if not content:
            return context.json_result({"created": False, "error": "memorize requires content."})

        metadata = context.runtime.suggest_memory_metadata(
            kind=MemoryKind.AGENT,
            context=arguments.get("context") if isinstance(arguments.get("context"), dict) else {},
            content=content,
        )
        title = str(arguments.get("title") or "").strip()
        summary = str(arguments.get("summary") or "").strip()
        if metadata is not None:
            title = title or metadata.title
            summary = summary or metadata.summary

        record = create_memory_record(
            title=title or fallback_memory_title(content),
            summary=summary or fallback_memory_summary(content),
            kind=MemoryKind.AGENT,
            context=arguments.get("context") if isinstance(arguments.get("context"), dict) else {},
            content=content,
        )
        saved = write_memory(context.runtime.home.brain_dir, record)
        return context.json_result(
            {
                "created": True,
                "path": str(saved.path or ""),
                "title": saved.title,
                "summary": saved.summary,
                "kind": saved.kind.value,
            }
        )
