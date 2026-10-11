"""Context-local runs with monotonic steps and retryable event delivery."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import Any, TypeVar, cast
from uuid import UUID, uuid4

from .backends import JsonlBackend, RunBackend

_JSON = TypeVar("_JSON")

_active: ContextVar[Run | None] = ContextVar("anomx_run", default=None)


class Run:
    """One execution in a project, with a step-indexed history and summary.

    Prefer :func:`init` for user code. ``managed=True`` binds a run whose
    lifecycle belongs to a worker: entering/exiting does not start or finish it.
    ``step=None`` events let a shared backend allocate steps atomically, useful
    when several processes log to the same hosted execution.

    A failed write remains pending with its original event ID. Call ``flush``
    before retrying application work; logging another event first also retries
    the pending write. Never silently discard telemetry on transport failure.
    """

    def __init__(
        self,
        *,
        project: str,
        name: str = "",
        config: Mapping[str, Any] | None = None,
        id: str | None = None,
        backend: RunBackend | None = None,
        directory: str | Path = ".anomx/runs",
        managed: bool = False,
    ) -> None:
        if not isinstance(project, str) or not project.strip():
            raise ValueError("A project name or reference is required.")
        self.id = str(UUID(id)) if id else str(uuid4())
        self.project = project.strip()
        self.name = name
        self.config = _json_copy(dict(config or {}))
        self.backend = (
            backend if backend is not None else JsonlBackend(Path(directory) / f"{self.id}.jsonl")
        )
        self.managed = managed
        self.summary: dict[str, Any] = {}
        self.step = 0
        self._pending: dict[str, Any] = {}
        self._pending_step: int | None = None
        self._outbox: dict[str, Any] | None = None
        self._started = managed
        self._finished = False
        self._lock = RLock()
        self._token: Token[Run | None] | None = None

    def _event(self, kind: str, **payload: Any) -> dict[str, Any]:  # noqa: ANN401 - JSON event payload
        return {
            "event_id": str(uuid4()),
            "run_id": self.id,
            "project": self.project,
            "kind": kind,
            "timestamp": datetime.now(UTC).isoformat(),
            **payload,
        }

    def _send(self, event: dict[str, Any]) -> None:
        self._outbox = event
        self.flush()

    def flush(self) -> None:
        """Retry the outstanding event, retaining its ID if delivery fails."""
        with self._lock:
            event = self._outbox
            if event is None:
                return
            self.backend.write(event)
            if event["kind"] == "history":
                self.summary.update(event["values"])
                self.step = (event["step"] if event["step"] is not None else self.step) + 1
                self._pending.clear()
                self._pending_step = None
            elif event["kind"] == "start":
                self._started = True
            elif event["kind"] == "finish":
                self._finished = True
            self._outbox = None

    def start(self) -> Run:
        """Create this run once, retrying an unacknowledged start if necessary."""
        with self._lock:
            self.flush()
            if not self._started:
                self._send(self._event("start", name=self.name, config=self.config))
        return self

    def log(
        self,
        name: str | Mapping[str, Any],
        value: Any = None,  # noqa: ANN401 - arbitrary JSON values
        *,
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        """Log JSON values at a monotonic step; numeric values become metrics.

        Each committed call advances one step. ``commit=False`` combines values
        at the current step; pass an explicit step consistently while building
        it. Non-finite numbers, empty keys, and decreasing steps are rejected.
        """
        values = dict(name) if isinstance(name, Mapping) else {name: value}
        if not values or any(
            not isinstance(key, str) or not key.strip() or len(key) > 128 for key in values
        ):
            raise ValueError("Log names must contain 1 to 128 characters.")
        values = _json_copy(values)
        with self._lock:
            self.start()
            if self._finished:
                raise RuntimeError("Cannot log to a finished run.")
            if step is not None and (
                isinstance(step, bool) or not isinstance(step, int) or step < self.step
            ):
                raise ValueError("Step must be an integer at least as large as the current step.")
            if step is not None and step > self.step and self._pending:
                raise ValueError("Commit the pending values before advancing the step.")
            if step is not None:
                self.step = step
                self._pending_step = step
            self._pending.update(values)
            if commit:
                self._send(
                    self._event(
                        "history",
                        step=self._pending_step if self.managed else self.step,
                        values=dict(self._pending),
                    )
                )

    def finish(self, *, status: str = "succeeded", error: str = "") -> None:
        """Flush pending values and finish a standalone run exactly once."""
        if status not in {"succeeded", "failed", "cancelled"}:
            raise ValueError("Run status must be succeeded, failed, or cancelled.")
        with self._lock:
            self.start()
            if self._finished:
                return
            if self._pending:
                self.log(dict(self._pending))
            if not self.managed:
                self._send(self._event("finish", status=status, error=error))
        if _active.get() is self and self._token is not None:
            _active.reset(self._token)
            self._token = None

    def __enter__(self) -> Run:
        self.start()
        if _active.get() is not self:
            self._token = _active.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.finish(status="failed" if exc_type else "succeeded", error=str(exc) if exc else "")
        finally:
            if _active.get() is self and self._token is not None:
                _active.reset(self._token)
                self._token = None


def _json_copy(value: _JSON) -> _JSON:
    return cast(_JSON, json.loads(json.dumps(value, allow_nan=False)))


def current_run() -> Run | None:
    """Return the run bound to this context, or ``None``."""
    return _active.get()


@contextmanager
def use_run(run: Run) -> Iterator[Run]:
    """Temporarily bind an existing run without changing its lifecycle."""
    token = _active.set(run)
    try:
        yield run
    finally:
        _active.reset(token)


def init(
    *,
    project: str | None = None,
    name: str = "",
    config: Mapping[str, Any] | None = None,
    id: str | None = None,
    backend: RunBackend | None = None,
    directory: str | Path = ".anomx/runs",
) -> Run:
    """Start and bind a standalone run, or reuse the active hosted execution.

    A standalone call requires ``project``. Inside a job, omit it to use the
    job's project; overriding the project of a managed execution is an error.
    """
    active = current_run()
    if active is not None and active.managed:
        if project is not None and project != active.project:
            raise ValueError("A hosted execution already belongs to a project.")
        return active
    if project is None:
        raise ValueError("A project is required for a standalone run.")
    run = Run(
        project=project, name=name, config=config, id=id, backend=backend, directory=directory
    ).start()
    run._token = _active.set(run)
    return run


def log(
    name: str | Mapping[str, Any],
    value: Any = None,  # noqa: ANN401 - arbitrary JSON values
    *,
    step: int | None = None,
    commit: bool = True,
) -> None:
    """Write values to the active run (see :meth:`Run.log`)."""
    run = current_run()
    if run is None:
        raise RuntimeError("Call anomx.init(project=...) before logging.")
    run.log(name, value, step=step, commit=commit)


def finish(*, status: str = "succeeded", error: str = "") -> None:
    """Finish the current standalone execution; a worker owns hosted lifecycles."""
    run = current_run()
    if run is None:
        raise RuntimeError("No active run.")
    run.finish(status=status, error=error)
