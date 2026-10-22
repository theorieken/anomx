"""Small, notebook-friendly progress and status helpers for local Anomx work."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from tqdm.auto import tqdm  # type: ignore[import-untyped]


def progress_bar(label: str, *, enabled: bool = True, **kwargs: Any) -> tqdm:  # noqa: ANN401
    """Create a tqdm progress bar with the shared Anomx label and quiet switch."""
    return tqdm(desc=f"anomx · {label}", disable=not enabled, dynamic_ncols=True, **kwargs)


@contextmanager
def operation(label: str, *, enabled: bool = True) -> Iterator[None]:
    """Print a concise elapsed-time receipt; exceptions retain their traceback."""
    started = perf_counter()
    if enabled:
        print(f"anomx · {label} …", file=sys.stderr, flush=True)
    try:
        yield
    except BaseException:
        if enabled:
            print(f"anomx · {label} failed ({perf_counter() - started:.2f}s)", file=sys.stderr)
        raise
    else:
        if enabled:
            print(f"anomx · {label} done ({perf_counter() - started:.2f}s)", file=sys.stderr)


def status(message: str) -> None:
    """Write one Anomx status line without disrupting an active progress bar."""
    tqdm.write(f"anomx · {message}", file=sys.stderr)
