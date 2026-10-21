"""Validation and local storage for portable run outputs."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024


def artifact_name(value: str) -> str:
    """Return a portable filename without accepting directory components."""
    if not isinstance(value, str) or not value.strip() or "/" in value or "\\" in value:
        raise ValueError("An artifact name must be a nonempty filename, without directories.")
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    if not name or len(name) > 255:
        raise ValueError("An artifact filename must contain 1 to 255 portable characters.")
    return name


def artifact_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Copy finite JSON metadata, rejecting lossy conversion of object keys."""
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("Artifact metadata must be a JSON object.")
    copied = dict(value) if value is not None else {}
    try:
        encoded = json.dumps(copied, allow_nan=False, ensure_ascii=True)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Artifact metadata must contain finite JSON values.") from exc
    if len(encoded.encode()) > MAX_METADATA_BYTES:
        raise ValueError("Artifact metadata must be at most 64 KiB.")
    pending: list[Any] = [copied]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("Artifact metadata object keys must be strings.")
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
    return json.loads(encoded)


def read_artifact(path: Path) -> bytes:
    """Read a bounded regular file; never stream devices or special files."""
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("An artifact must be an existing regular file.")
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("Artifacts must be at most 64 MiB.")
    with path.open("rb") as source:
        content = source.read(MAX_ARTIFACT_BYTES + 1)
    if len(content) > MAX_ARTIFACT_BYTES:
        raise ValueError("Artifacts must be at most 64 MiB.")
    return content


def store_local_artifact(
    directory: Path, name: str, content: bytes, metadata: dict[str, Any]
) -> dict[str, Any]:
    """Keep a content-addressed copy independently of the original output file."""
    digest = hashlib.sha256(content).hexdigest()
    target = directory / digest / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.resolve().is_relative_to(directory.resolve()):
        raise ValueError("The artifact destination must stay inside its storage directory.")
    try:
        with target.open("xb") as output:
            output.write(content)
    except FileExistsError:
        if hashlib.sha256(read_artifact(target)).hexdigest() != digest:
            raise ValueError("An existing local artifact has an invalid checksum.") from None
    return {
        "path": str(target.resolve()),
        "name": name,
        "sha256": digest,
        "size_bytes": len(content),
        "metadata": metadata,
    }
