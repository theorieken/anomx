"""Shared state helpers for the Anomx CLI agent transcript."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

PLAN_EVENT_TYPE = "plan_update"
WORKER_EVENT_TYPE = "worker_event"
PROCESS_EVENT_TYPE = "process_event"
WORKER_STATE_WORKING = "working"
WORKER_STATE_READY = "ready"
WORKER_STATE_INTERRUPTED = "interrupted"
WORKER_STATE_REMOVED = "removed"
RUNNING_WORKER_STATUSES = frozenset({WORKER_STATE_WORKING})
RUNNING_PROCESS_STATUSES = frozenset({"running"})


@dataclass(frozen=True)
class PlanStep:
    """A single operator-owned plan step."""

    position: int
    title: str
    description: str
    is_done: bool = False


@dataclass(frozen=True)
class WorkerAgentSnapshot:
    """Latest known state for a worker agent."""

    worker_id: str
    name: str
    status: str
    statement: str
    prompt: str = ""
    response: str = ""
    started_at: str = ""
    finished_at: str = ""
    context_tokens: int = 0
    context_percent: int = 0


@dataclass(frozen=True)
class AsyncProcessSnapshot:
    """Latest known state for a long-running async process."""

    process_id: str
    command: str
    status: str
    statement: str = ""
    output: str = ""
    started_at: str = ""
    finished_at: str = ""
    exit_code: int | None = None
    source: str = "process"
    owner_id: str = ""
    owner_name: str = ""
    pid: int | None = None


def event_payload_type(event: Mapping[str, Any]) -> str:
    """Return the semantic event type for a transcript event."""

    payload = event.get("payload")
    if isinstance(payload, dict) and event.get("type") == "event_msg":
        return str(payload.get("type", ""))
    return str(event.get("type", ""))


def event_payload(event: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a transcript payload mapping when present."""

    payload = event.get("payload")
    return payload if isinstance(payload, dict) else {}


def build_plan_steps(raw_steps: object) -> tuple[PlanStep, ...]:
    """Build normalized plan steps from tool input without trusting positions."""

    if not isinstance(raw_steps, list):
        return ()

    steps: list[PlanStep] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            continue
        title = str(raw_step.get("title", "")).strip()
        description = str(raw_step.get("description", "")).strip()
        if not title:
            continue
        steps.append(
            PlanStep(
                position=len(steps) + 1,
                title=title,
                description=description,
                is_done=bool(raw_step.get("is_done", False)),
            )
        )
    return tuple(steps)


def merge_plan_steps(
    current_steps: Iterable[PlanStep],
    raw_updates: object,
) -> tuple[PlanStep, ...]:
    """Merge update-plan tool input into the current plan."""

    if not isinstance(raw_updates, list):
        return tuple(sorted(current_steps, key=lambda step: step.position))

    current = {step.position: step for step in current_steps}
    for fallback_position, raw_step in enumerate(raw_updates, start=1):
        if not isinstance(raw_step, dict):
            continue
        position = _integer(raw_step.get("position"), fallback_position)
        base = current.get(position)
        title = _optional_text(raw_step.get("title"))
        description = _optional_text(raw_step.get("description"))
        is_done = raw_step.get("is_done")
        current[position] = PlanStep(
            position=position,
            title=title if title is not None else (base.title if base else ""),
            description=(
                description if description is not None else (base.description if base else "")
            ),
            is_done=bool(is_done if is_done is not None else (base.is_done if base else False)),
        )

    return tuple(
        step
        for step in sorted(current.values(), key=lambda item: item.position)
        if step.title
    )


def serialize_plan_steps(steps: Iterable[PlanStep]) -> list[dict[str, object]]:
    """Serialize plan steps for transcript storage and tool results."""

    return [
        {
            "position": step.position,
            "title": step.title,
            "description": step.description,
            "is_done": step.is_done,
        }
        for step in steps
    ]


def latest_plan_steps(events: Iterable[Mapping[str, Any]]) -> tuple[PlanStep, ...]:
    """Return the most recent plan in a session transcript."""

    steps: tuple[PlanStep, ...] = ()
    for event in events:
        if event_payload_type(event) != PLAN_EVENT_TYPE:
            continue
        steps = build_plan_steps_with_positions(event_payload(event).get("steps"))
    return steps


def build_plan_steps_with_positions(raw_steps: object) -> tuple[PlanStep, ...]:
    """Build normalized plan steps from transcript payloads with positions."""

    if not isinstance(raw_steps, list):
        return ()

    steps: list[PlanStep] = []
    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            continue
        title = str(raw_step.get("title", "")).strip()
        if not title:
            continue
        steps.append(
            PlanStep(
                position=_integer(raw_step.get("position"), index),
                title=title,
                description=str(raw_step.get("description", "")).strip(),
                is_done=bool(raw_step.get("is_done", False)),
            )
        )
    return tuple(sorted(steps, key=lambda step: step.position))


def worker_snapshots(
    events: Iterable[Mapping[str, Any]],
) -> tuple[WorkerAgentSnapshot, ...]:
    """Return latest worker snapshots derived from transcript events."""

    snapshots: dict[str, WorkerAgentSnapshot] = {}
    for event in events:
        if event_payload_type(event) != WORKER_EVENT_TYPE:
            continue
        payload = event_payload(event)
        worker_id = str(payload.get("worker_id", "")).strip()
        if not worker_id:
            continue
        previous = snapshots.get(worker_id)
        status = _normalize_worker_status(
            _text_with_default(
                payload.get("status"),
                previous.status if previous else WORKER_STATE_WORKING,
            )
        )
        snapshots[worker_id] = WorkerAgentSnapshot(
            worker_id=worker_id,
            name=_text_with_default(payload.get("name"), previous.name if previous else "Worker"),
            status=status,
            statement=_text_event_value(
                payload,
                "statement",
                previous.statement if previous else "Thinking",
            ),
            prompt=_text_with_default(payload.get("prompt"), previous.prompt if previous else ""),
            response=_text_with_default(
                payload.get("response"),
                previous.response if previous else "",
            ),
            started_at=_text_with_default(
                payload.get("started_at"),
                previous.started_at if previous else "",
            ),
            finished_at=_text_with_default(
                payload.get("finished_at"),
                previous.finished_at if previous else "",
            ),
            context_tokens=_optional_integer(
                payload.get("context_tokens"),
                previous.context_tokens if previous else 0,
            )
            or 0,
            context_percent=_optional_integer(
                payload.get("context_percent"),
                previous.context_percent if previous else 0,
            )
            or 0,
        )
    return tuple(
        snapshot
        for snapshot in snapshots.values()
        if snapshot.status != WORKER_STATE_REMOVED
    )


def running_worker_snapshots(
    events: Iterable[Mapping[str, Any]],
) -> tuple[WorkerAgentSnapshot, ...]:
    """Return worker snapshots that are currently running."""

    return tuple(
        worker
        for worker in worker_snapshots(events)
        if worker.status in RUNNING_WORKER_STATUSES
    )


def process_snapshots(
    events: Iterable[Mapping[str, Any]],
) -> tuple[AsyncProcessSnapshot, ...]:
    """Return latest async process snapshots derived from transcript events."""

    snapshots: dict[str, AsyncProcessSnapshot] = {}
    for event in events:
        if event_payload_type(event) != PROCESS_EVENT_TYPE:
            continue
        payload = event_payload(event)
        process_id = str(payload.get("process_id", "")).strip()
        if not process_id:
            continue
        previous = snapshots.get(process_id)
        snapshots[process_id] = AsyncProcessSnapshot(
            process_id=process_id,
            command=_text_with_default(
                payload.get("command"),
                previous.command if previous else "",
            ),
            status=_text_with_default(
                payload.get("status"),
                previous.status if previous else "running",
            ),
            statement=_text_with_default(
                payload.get("statement"),
                previous.statement if previous else "",
            ),
            output=_text_with_default(payload.get("output"), previous.output if previous else ""),
            started_at=_text_with_default(
                payload.get("started_at"),
                previous.started_at if previous else "",
            ),
            finished_at=_text_with_default(
                payload.get("finished_at"),
                previous.finished_at if previous else "",
            ),
            exit_code=_optional_integer(
                payload.get("exit_code"),
                previous.exit_code if previous else None,
            ),
            source=_text_with_default(
                payload.get("source"),
                previous.source if previous else "process",
            ),
            owner_id=_text_with_default(
                payload.get("owner_id"),
                previous.owner_id if previous else "",
            ),
            owner_name=_text_with_default(
                payload.get("owner_name"),
                previous.owner_name if previous else "",
            ),
            pid=_optional_integer(
                payload.get("pid"),
                previous.pid if previous else None,
            ),
        )
    return tuple(snapshots.values())


def running_process_snapshots(
    events: Iterable[Mapping[str, Any]],
) -> tuple[AsyncProcessSnapshot, ...]:
    """Return async process snapshots that are currently running."""

    return tuple(
        process
        for process in process_snapshots(events)
        if process.status in RUNNING_PROCESS_STATUSES
    )


def _integer(value: object, fallback: int) -> int:
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return fallback
    else:
        return fallback
    return parsed if parsed > 0 else fallback


def _optional_integer(value: object, fallback: int | None) -> int | None:
    if value is None:
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return fallback
    else:
        return fallback


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _text_with_default(value: object, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or fallback


def _text_event_value(
    payload: Mapping[str, Any],
    key: str,
    fallback: str,
) -> str:
    if key not in payload:
        return fallback
    value = payload.get(key)
    return str(value).strip() if value is not None else ""


def _normalize_worker_status(status: str) -> str:
    normalized = status.strip().lower()
    return {
        "running": WORKER_STATE_WORKING,
        "finished": WORKER_STATE_READY,
        "done": WORKER_STATE_READY,
        "failed": WORKER_STATE_INTERRUPTED,
        "stopped": WORKER_STATE_INTERRUPTED,
    }.get(normalized, normalized or WORKER_STATE_WORKING)
