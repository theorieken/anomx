"""Persistent memory records for the Anomx CLI agent."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4


class MemoryKind(StrEnum):
    """Source of a stored memory."""

    APPROVAL = "approval"
    AGENT = "agent"
    MANUAL = "manual"


@dataclass(frozen=True)
class MemoryMetadata:
    """Short model-generated metadata for a memory."""

    title: str
    summary: str


@dataclass(frozen=True)
class MemoryRecord:
    """A persisted agent memory."""

    path: Path | None
    created_at: str
    uses: int
    title: str
    summary: str
    kind: MemoryKind
    context: dict[str, Any]
    content: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "uses": self.uses,
            "title": self.title,
            "summary": self.summary,
            "kind": self.kind.value,
            "context": self.context,
            "content": self.content,
        }


def create_memory_record(
    *,
    title: str,
    summary: str,
    kind: MemoryKind | str,
    context: Mapping[str, Any] | None,
    content: str,
    created_at: str | None = None,
    uses: int = 0,
    path: Path | None = None,
) -> MemoryRecord:
    """Build a normalized memory record."""

    normalized_kind = normalize_memory_kind(kind)
    clean_content = content.strip()
    return MemoryRecord(
        path=path,
        created_at=created_at or utc_now_iso(),
        uses=max(0, int(uses)),
        title=sanitize_memory_title(title) or fallback_memory_title(clean_content),
        summary=sanitize_memory_summary(summary) or fallback_memory_summary(clean_content),
        kind=normalized_kind,
        context=dict(context or {}),
        content=clean_content,
    )


def normalize_memory_kind(value: MemoryKind | str) -> MemoryKind:
    """Return a valid memory kind."""

    try:
        return MemoryKind(str(value).strip().lower())
    except ValueError:
        return MemoryKind.AGENT


def write_memory(memory_dir: Path, record: MemoryRecord) -> MemoryRecord:
    """Write a memory record to a new ``.anomx`` JSON file."""

    memory_dir.mkdir(parents=True, exist_ok=True)
    created = parse_datetime(record.created_at) or datetime.now(UTC)
    filename = f"{created:%Y%m%d}_{uuid4().hex}.anomx"
    path = memory_dir / filename
    stored = MemoryRecord(
        path=path,
        created_at=record.created_at,
        uses=record.uses,
        title=record.title,
        summary=record.summary,
        kind=record.kind,
        context=record.context,
        content=record.content,
    )
    path.write_text(
        json.dumps(stored.to_payload(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return stored


def load_memories(memory_dir: Path) -> list[MemoryRecord]:
    """Load all readable memory files from a directory."""

    if not memory_dir.is_dir():
        return []
    records: list[MemoryRecord] = []
    for path in sorted(memory_dir.glob("*.anomx")):
        record = load_memory(path)
        if record is not None:
            records.append(record)
    return sorted(records, key=lambda item: item.created_at, reverse=True)


def load_memory(path: Path) -> MemoryRecord | None:
    """Load one memory file."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    context = payload.get("context")
    return create_memory_record(
        title=str(payload.get("title") or ""),
        summary=str(payload.get("summary") or ""),
        kind=str(payload.get("kind") or MemoryKind.AGENT.value),
        context=context if isinstance(context, dict) else {},
        content=str(payload.get("content") or ""),
        created_at=str(payload.get("created_at") or ""),
        uses=_safe_int(payload.get("uses")),
        path=path,
    )


def increment_memory_uses(records: list[MemoryRecord]) -> None:
    """Increment the use counter for records already selected for prompt context."""

    for record in records:
        if record.path is None:
            continue
        updated = MemoryRecord(
            path=record.path,
            created_at=record.created_at,
            uses=record.uses + 1,
            title=record.title,
            summary=record.summary,
            kind=record.kind,
            context=record.context,
            content=record.content,
        )
        try:
            record.path.write_text(
                json.dumps(
                    updated.to_payload(),
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError:
            continue


def sanitize_memory_metadata(text: str) -> MemoryMetadata | None:
    """Extract memory metadata JSON from backend output."""

    payload = _extract_json_object(text)
    if payload is None:
        return None
    title = sanitize_memory_title(str(payload.get("title") or ""))
    summary = sanitize_memory_summary(str(payload.get("summary") or ""))
    if not title or not summary:
        return None
    return MemoryMetadata(title=title, summary=summary)


def sanitize_memory_title(title: str) -> str:
    """Return a compact memory title."""

    cleaned = " ".join(title.strip().strip("\"'`").split())
    cleaned = cleaned.rstrip(".:;,-")
    words = cleaned.split()
    if len(words) > 8:
        cleaned = " ".join(words[:8])
    return cleaned[:80]


def sanitize_memory_summary(summary: str) -> str:
    """Return a compact memory summary."""

    cleaned = " ".join(summary.strip().split())
    return cleaned[:180]


def fallback_memory_title(content: str) -> str:
    """Return a deterministic title when backend metadata is unavailable."""

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*", content)
    if not words:
        return "Untitled Memory"
    return " ".join(words[:6])[:80]


def fallback_memory_summary(content: str) -> str:
    """Return a deterministic summary when backend metadata is unavailable."""

    cleaned = " ".join(content.strip().split())
    if not cleaned:
        return "Empty memory."
    return cleaned[:177] + "..." if len(cleaned) > 180 else cleaned


def utc_now_iso() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def parse_datetime(value: str) -> datetime | None:
    """Parse a stored memory timestamp."""

    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None
