"""Runtime process state value objects."""

from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AsyncProcessState:
    """Mutable in-process state for a long-running async command."""

    process_id: str
    command: str
    statement: str
    status: str
    started_at: str
    process: subprocess.Popen[str]
    finished_at: str = ""
    output: str = ""
    exit_code: int | None = None
    source: str = "process"
    owner_id: str = ""
    owner_name: str = ""
    session_path: Path | None = None
    output_chunks: list[str] = field(default_factory=list, repr=False)
    last_output_event_at: float = 0.0
    last_output_event_text: str = ""
    thread: threading.Thread | None = None
