"""Revisioned, conflict-preserving synchronization for the agent's own files."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

MAX_BRAIN_FILES = 512
# Deleted revisions remain available for offline clients without consuming live file slots.
MAX_BRAIN_RECORDS = 8192
MAX_BRAIN_FILE_BYTES = 1_048_576
MAX_BRAIN_BYTES = 8_388_608


def brain_path(value: str) -> str:
    """Accept portable relative file paths, never traversal or special files."""
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or len(value) > 512
    ):
        raise ValueError("Brain files require a relative path without traversal.")
    return str(path)


def brain_content(record: dict[str, Any]) -> bytes | None:
    if record.get("deleted") is True:
        return None
    content = record.get("content")
    if not isinstance(content, str) or len(content) > MAX_BRAIN_FILE_BYTES * 2:
        raise ValueError("Brain file content exceeds its 1 MiB limit.")
    raw = base64.b64decode(content, validate=True)
    if len(raw) > MAX_BRAIN_FILE_BYTES:
        raise ValueError("Brain file content exceeds its 1 MiB limit.")
    return raw


def brain_digest(raw: bytes | None) -> str:
    return hashlib.sha256(raw).hexdigest() if raw is not None else ""


def available_brain_path(path: str, raw: bytes, files: dict[str, bytes]) -> str:
    """Keep both versions using ordinary numbered names; retries reuse identical copies."""
    candidate = PurePosixPath(path)
    for number in range(MAX_BRAIN_FILES + 1):
        name = str(
            candidate
            if number == 0
            else candidate.with_name(f"{candidate.stem}_{number:02d}{candidate.suffix}")
        )
        if name in files:
            if files[name] == raw:
                return name
            continue
        # Also handle a file/directory collision (e.g. notes versus notes/topic.md).
        parent = next(
            (parent for parent in PurePosixPath(name).parents if str(parent) in files), None
        )
        if parent is not None:
            renamed = available_brain_path(str(parent), b"", {**files, str(parent): b"occupied"})
            return available_brain_path(
                renamed + "/" + str(PurePosixPath(name).relative_to(parent)), raw, files
            )
        if any(other.startswith(name + "/") for other in files):
            continue
        return name
    raise ValueError("Too many files with the same name.")


def merge_brain_files(current: dict[str, Any], changes: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply compare-and-swap writes to a snapshot; callers provide persistence/locking."""
    if not isinstance(changes, list) or len(changes) > MAX_BRAIN_RECORDS:
        raise ValueError("A brain sync accepts at most 8192 revisions.")
    files = dict(current)
    conflicts = []
    renamed = {}
    seen = set()
    for change in sorted(
        changes, key=lambda item: not (isinstance(item, dict) and item.get("deleted"))
    ):
        if not isinstance(change, dict) or not isinstance(change.get("path"), str):
            raise ValueError("Each brain change requires a path.")
        path = brain_path(change["path"])
        if path in seen:
            raise ValueError("Duplicate brain file path.")
        seen.add(path)
        raw = brain_content(change)
        previous = files.get(path, {})
        revision = previous.get("revision", 0)
        base = change.get("base_revision", 0)
        if not isinstance(base, int) or isinstance(base, bool) or base < 0:
            raise ValueError("Invalid brain base revision.")
        if previous and previous.get("sha256", "") == brain_digest(raw):
            continue  # Retry after an unknown network outcome is idempotent.
        active = {
            name: brain_content(record)
            for name, record in files.items()
            if not record.get("deleted")
        }
        blocked = any(str(parent) in active for parent in PurePosixPath(path).parents) or any(
            name.startswith(path + "/") for name in active
        )
        if base != revision or (raw is not None and blocked):
            conflicts.append(path)
            if raw is None:
                continue  # A stale deletion must never remove newer remote work.
            destination = available_brain_path(path, raw, active)
            renamed[path] = destination
            path = destination
            previous = files.get(path, {})
            if previous.get("sha256") == brain_digest(raw):
                continue
            revision = previous.get("revision", 0)
        files[path] = {
            "revision": revision + 1,
            "sha256": brain_digest(raw),
            "deleted": raw is None,
            "content": base64.b64encode(raw).decode() if raw is not None else "",
        }
    active = {path: record for path, record in files.items() if not record.get("deleted")}
    paths = set(active)
    if any(
        str(parent) in paths
        for path in paths
        for parent in PurePosixPath(path).parents
        if str(parent) != "."
    ):
        raise ValueError("A brain file cannot also be a directory.")
    if (
        len(active) > MAX_BRAIN_FILES
        or len(files) > MAX_BRAIN_RECORDS
        or sum(len(brain_content(record) or b"") for record in active.values()) > MAX_BRAIN_BYTES
    ):
        raise ValueError("The brain is limited to 512 files, 8192 revisions, and 8 MiB of content.")
    return {"files": files, "conflicts": conflicts, "renamed": renamed}


def read_brain_files(root: Path) -> dict[str, bytes]:
    files = {}
    if root.is_symlink():
        raise ValueError("The brain root must not be a symlink.")
    root.mkdir(parents=True, exist_ok=True)
    for directory, subdirs, names in os.walk(root, followlinks=False):
        if any((Path(directory) / name).is_symlink() for name in subdirs + names):
            raise ValueError("Brain synchronization does not follow symlinks.")
        for name in names:
            path = Path(directory) / name
            if not path.is_file():
                raise ValueError("Brain synchronization only supports regular files.")
            if path.stat().st_size > MAX_BRAIN_FILE_BYTES:
                raise ValueError("A brain file exceeds the 1 MiB limit.")
            files[brain_path(path.relative_to(root).as_posix())] = path.read_bytes()
            if len(files) > MAX_BRAIN_FILES or sum(map(len, files.values())) > MAX_BRAIN_BYTES:
                raise ValueError("The brain exceeds its 512-file or 8 MiB limit.")
    return files


def atomic_brain_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".brain-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def synchronize_brain(
    root: Path,
    state_path: Path,
    exchange: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    scope: str,
) -> dict[str, Any]:
    """Sync files, preserving concurrent edits as conflict copies before replacing them."""
    import fcntl

    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get("scope") not in {None, scope}:
            raise ValueError(
                "This brain belongs to another platform account; select a separate agent home."
            )
        from anomx.agent.memories import migrate_legacy_memories

        migrate_legacy_memories(root)
        tracked = state.get("files", {})
        before = read_brain_files(root)
        changes = []
        for path in sorted(set(before) | set(tracked)):
            raw = before.get(path)
            previous = tracked.get(path, {})
            if previous.get("sha256", "") == brain_digest(raw):
                continue
            changes.append(
                {
                    "path": path,
                    "base_revision": previous.get("revision", 0),
                    "deleted": raw is None,
                    "content": base64.b64encode(raw).decode() if raw is not None else "",
                }
            )
        response = exchange({"changes": changes})
        remote = response.get("files")
        if not isinstance(remote, dict):
            raise ValueError("The platform returned an invalid brain snapshot.")
        # Validate every remote path and payload before modifying local files.
        normalized = merge_brain_files(
            {}, [{"path": path, **record, "base_revision": 0} for path, record in remote.items()]
        )["files"]
        current = read_brain_files(root)
        conflicts = set(response.get("conflicts", []))
        # Relocate acknowledged name collisions before applying the snapshot. This
        # also makes a file-versus-directory collision safe on the local filesystem.
        for source, destination in response.get("renamed", {}).items():
            source, destination = brain_path(source), brain_path(destination)
            raw = current.get(source)
            incoming = normalized.get(destination)
            if (
                source == destination
                or raw is None
                or raw != before.get(source)
                or incoming is None
                or brain_content(incoming) != raw
            ):
                continue
            target = root / destination
            if target.resolve() != root.resolve() / destination:
                raise ValueError("Unsafe brain destination.")
            atomic_brain_write(target, raw)
            (root / source).unlink()
            current[destination] = raw
            current.pop(source, None)
            parent = (root / source).parent
            while parent != root and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent
        ordered = sorted(
            normalized.items(),
            key=lambda item: (
                not item[1].get("deleted"),
                -item[0].count("/"),
                item[0],
            ),
        )
        for path, record in ordered:
            incoming = brain_content(record)
            local = current.get(path)
            if local == incoming:
                continue
            target = root / path
            if target.resolve() != root.resolve() / path:
                raise ValueError("Unsafe brain destination.")
            if local is not None and (
                local != before.get(path)
                or (path in conflicts and path not in response.get("renamed", {}))
            ):
                # Preserve edits made while the request was in flight, and support older servers.
                occupied = {
                    **current,
                    **{
                        name: brain_content(value)
                        for name, value in normalized.items()
                        if not value.get("deleted")
                    },
                }
                occupied[path] = incoming or b""
                copy = available_brain_path(path, local, occupied)
                atomic_brain_write(root / copy, local)
                current[copy] = local
                conflicts.add(path)
            if incoming is None:
                target.unlink(missing_ok=True)
                parent = target.parent
                while parent != root and parent.is_dir() and not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
            else:
                if target.is_dir():
                    target.rmdir()  # Only remove empty directories, never local notes.
                atomic_brain_write(target, incoming)
        atomic_brain_write(
            state_path,
            json.dumps(
                {
                    "scope": scope,
                    "files": {
                        path: {"revision": record["revision"], "sha256": record["sha256"]}
                        for path, record in remote.items()
                    },
                }
            ).encode(),
        )
        migrate_legacy_memories(root)
        return {"files": len(remote), "conflicts": sorted(conflicts)}
