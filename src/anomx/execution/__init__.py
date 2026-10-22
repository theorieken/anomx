"""Portable execution, work contexts, and run tracking."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "LocalExecutor": "anomx.execution.local",
    "LocalResult": "anomx.execution.local",
    "JobDefinition": "anomx.execution.jobs",
    "JobDefinitionError": "anomx.execution.jobs",
    "JobNode": "anomx.execution.jobs",
    "JobNodeType": "anomx.execution.jobs",
    "JobRunError": "anomx.execution.nodes",
    "JobRunResult": "anomx.execution.runner",
    "JobRunner": "anomx.execution.runner",
    "NodeExecution": "anomx.execution.nodes",
    "NodeExecutor": "anomx.execution.nodes",
    "frame_to_records": "anomx.execution.context",
    "normalize_context": "anomx.execution.context",
    "read_context_flag": "anomx.execution.context",
    "read_context_frame": "anomx.execution.context",
    "records_to_frame": "anomx.execution.context",
    "run_python_body": "anomx.execution.nodes",
    "write_context_flag": "anomx.execution.context",
    "write_context_frame": "anomx.execution.context",
    "CallbackBackend": "anomx.execution.backends",
    "JsonlBackend": "anomx.execution.backends",
    "Run": "anomx.execution.runs",
    "RunBackend": "anomx.execution.base.tracking",
    "current_run": "anomx.execution.runs",
    "finish": "anomx.execution.runs",
    "init": "anomx.execution.runs",
    "log": "anomx.execution.runs",
    "use_run": "anomx.execution.runs",
    "BaseAlgorithm": "anomx.execution.base.algorithm",
    "JobResult": "anomx.execution.results",
    "JobSpec": "anomx.execution.results",
    "JobSummary": "anomx.execution.results",
    "PipelineAlgorithm": "anomx.execution.pipeline",
    "PipelineOrchestrator": "anomx.execution.pipeline",
    "FRAME_KEYS": "anomx.execution.context",
    "WorkContext": "anomx.execution.work",
    "WorkGraph": "anomx.execution.work",
    "WorkStep": "anomx.execution.work",
    "get_work_context": "anomx.execution.work",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:  # noqa: ANN401 - lazy public API
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
