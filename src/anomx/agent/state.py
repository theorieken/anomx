"""Shared state helpers for the Anomx CLI agent transcript."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

PLAN_EVENT_TYPE = "plan_update"
WORKER_EVENT_TYPE = "worker_event"
RUNNING_WORKER_STATUSES = frozenset({"running"})


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
        snapshots[worker_id] = WorkerAgentSnapshot(
            worker_id=worker_id,
            name=_text_with_default(payload.get("name"), previous.name if previous else "Worker"),
            status=_text_with_default(
                payload.get("status"),
                previous.status if previous else "running",
            ),
            statement=_text_with_default(
                payload.get("statement"),
                previous.statement if previous else "thinking",
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
        )
    return tuple(snapshots.values())


def running_worker_snapshots(
    events: Iterable[Mapping[str, Any]],
) -> tuple[WorkerAgentSnapshot, ...]:
    """Return worker snapshots that are currently running."""

    return tuple(
        worker
        for worker in worker_snapshots(events)
        if worker.status in RUNNING_WORKER_STATUSES
    )


def _integer(value: object, fallback: int) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _text_with_default(value: object, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or fallback
