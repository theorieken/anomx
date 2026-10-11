"""Project-scoped execution tracking, without a dependency on the platform.

Use ``with init(project="forecast", name="baseline")`` to track an execution.
Inside that context, ``anomx.log("loss", 0.1)`` writes a step. Pass a mapping to
log several values together, or ``commit=False`` to accumulate the current step.
The default backend writes JSONL locally; :class:`PlatformBackend` sends the
same events to an authenticated Anomx installation. Workers bind their existing
run, so logging in job code never creates a second execution.
"""

from .backends import CallbackBackend, JsonlBackend, PlatformBackend, RunBackend
from .core import Run, current_run, finish, init, log, use_run

__all__ = [
    "CallbackBackend",
    "JsonlBackend",
    "PlatformBackend",
    "Run",
    "RunBackend",
    "current_run",
    "finish",
    "init",
    "log",
    "use_run",
]
