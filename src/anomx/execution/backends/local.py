"""Interchangeable persistence for run lifecycle and step events."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import RLock
from typing import Any


class JsonlBackend:
    """Append portable JSON events to a local file (offline by default)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = RLock()

    def write(self, event: Mapping[str, Any]) -> None:
        line = json.dumps(dict(event), allow_nan=False, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()


class CallbackBackend:
    """Adapt an existing worker transport or application-specific event sink."""

    def __init__(self, callback: Callable[[dict[str, Any]], Any]) -> None:
        self.callback = callback

    def write(self, event: Mapping[str, Any]) -> None:
        self.callback(dict(event))
