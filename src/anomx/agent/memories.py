"""Persistent memory records for the Anomx CLI agent."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


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


def memory_markdown(record: MemoryRecord) -> str:
    """Render an ordinary note that can be read and edited with any text editor."""
    parts = [f"# {record.title}", record.summary, record.content]
    if record.context:
        parts.append(
            "## Context\n\n```json\n"
            + json.dumps(record.context, indent=2, ensure_ascii=False)
            + "\n```"
        )
    return "\n\n".join(part for part in parts if part) + "\n"


@contextmanager
def _memory_lock(memory_dir: Path) -> Iterator[None]:
    import fcntl

    memory_dir.parent.mkdir(parents=True, exist_ok=True)
    with (memory_dir.parent / ".brain-write.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def write_memory(memory_dir: Path, record: MemoryRecord) -> MemoryRecord:
    """Write a plain Markdown note with a descriptive, collision-safe filename."""
    memory_dir = memory_dir.resolve()
    with _memory_lock(memory_dir):
        return _write_memory(memory_dir, record)


def _write_memory(memory_dir: Path, record: MemoryRecord) -> MemoryRecord:
    from anomx.agent.brain import atomic_brain_write, available_brain_path

    memory_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^\w-]+", "-", record.title.lower()).strip("-") or "note"
    raw = memory_markdown(record).encode("utf-8")
    existing = {
        path.name: path.read_bytes()
        for path in memory_dir.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    name = available_brain_path(f"{stem}.md", raw, existing)
    path = memory_dir / name
    atomic_brain_write(path, raw)
    return load_memory(path) or record


def migrate_legacy_memories(memory_dir: Path) -> bool:
    """Convert legacy JSON notes, retaining originals outside the visible notes folder."""
    from anomx.agent.brain import atomic_brain_write, available_brain_path

    memory_dir = memory_dir.resolve()
    changed = False
    with _memory_lock(memory_dir):
        for path in sorted(memory_dir.glob("*.anomx")):
            if path.is_symlink():
                continue
            record = load_memory(path)
            if record is None:
                continue
            raw = path.read_bytes()
            _write_memory(memory_dir, record)
            archive = memory_dir.parent / ".brain-legacy"
            existing = {
                item.name: item.read_bytes()
                for item in archive.glob("*")
                if item.is_file() and not item.is_symlink()
            }
            name = available_brain_path(path.name, raw, existing)
            atomic_brain_write(archive / name, raw)
            path.unlink()
            changed = True
    return changed


def load_memories(memory_dir: Path) -> list[MemoryRecord]:
    """List ordinary Markdown and text notes, including notes written by hand."""
    if not memory_dir.is_dir():
        return []
    memory_dir = memory_dir.resolve()
    migrate_legacy_memories(memory_dir)
    records = []
    for path in sorted(memory_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"} or path.is_symlink() or not path.is_file():
            continue
        if any(
            (memory_dir / parent).is_symlink() for parent in path.relative_to(memory_dir).parents
        ):
            continue
        record = load_memory(path)
        if record is not None:
            records.append(record)
    return sorted(records, key=lambda item: item.created_at, reverse=True)


def load_memory(path: Path) -> MemoryRecord | None:
    """Read a plain note or a legacy record during migration."""
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".anomx":
            payload = json.loads(text)
            if not isinstance(payload, dict) or "content" not in payload:
                return None
            return create_memory_record(
                title=str(payload.get("title") or ""),
                summary=str(payload.get("summary") or ""),
                kind=str(payload.get("kind") or "agent"),
                context=payload.get("context") if isinstance(payload.get("context"), dict) else {},
                content=str(payload.get("content") or ""),
                created_at=str(payload.get("created_at") or ""),
                uses=_safe_int(payload.get("uses")),
                path=path,
            )
        paragraphs = text.strip().split("\n\n")
        title = paragraphs.pop(0).lstrip("# ") if text.startswith("# ") else path.stem
        summary = paragraphs.pop(0) if len(paragraphs) > 1 else ""
        return create_memory_record(
            title=title,
            summary=summary,
            kind=MemoryKind.AGENT,
            context={},
            content="\n\n".join(paragraphs),
            path=path,
            created_at=datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


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
