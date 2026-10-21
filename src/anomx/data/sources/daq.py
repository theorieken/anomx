"""Optional, isolated readers for scalar fields in native FLASH/XFEL DAQ files."""

from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
from collections.abc import Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from anomx.data.sources.mounted import MountedSource

__all__ = ["DaqReadError", "DaqRecord", "iter_daq_records"]

_MAX_MESSAGE = 8 * 1024 * 1024
_MAX_STDERR = 16 * 1024


class DaqReadError(RuntimeError):
    """The optional native reader failed, timed out or returned an unsupported record."""


@dataclass(frozen=True)
class DaqRecord:
    """One native event, without inferred time units, calibration or physical labels."""

    timestamp: float
    event_id: int
    channel: str
    values: Mapping[str, float | None]
    invalid_fields: Mapping[str, str]
    field_metadata: Mapping[str, Mapping[str, Any]]
    source_path: str
    dataset_version: str
    record_index: int


# Deliberately stdlib-only: this runs in the caller-selected native-library Python,
# which need not have Anomx (or the ML interpreter's Python ABI) installed.
_CHILD = r"""
import json, math, os, sys, time

protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
MAX_MESSAGE = 8 * 1024 * 1024

def emit(message):
    raw = json.dumps(message, allow_nan=False, separators=(",", ":"))
    if len(raw.encode("utf8")) + 1 > MAX_MESSAGE:
        raise ValueError("Native DAQ batch exceeds 8 MiB; reduce batch_size.")
    protocol.write(raw + "\n")
    protocol.flush()

def identity(path):
    s = os.stat(path)
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]

connected = False
try:
    request = json.load(sys.stdin)
    try:
        import pydaq
    except ImportError as error:
        raise RuntimeError(
            "Optional pydaq is unavailable in the selected interpreter: " + str(error)
        )
    if identity(request["path"]) != request["identity"]:
        raise ValueError("Mounted source identity changed before native open.")
    connected = True
    pydaq.connect(
        files=[request["path"]], chans=[request["channel"]],
        exp=request["experiment"], linac=request["linac"], local=True,
    )
    count, batch = 0, []
    while request["limit"] is None or count < request["limit"]:
        event = pydaq.getdata()
        if event is None:
            break
        if not event:
            time.sleep(0.001)
            continue
        if not isinstance(event, list) or len(event) != 1 or not isinstance(event[0], list):
            raise ValueError("Expected one grouped scalar A_TS_GSPECTRUM channel.")
        selected = {}
        for field in event[0]:
            if not isinstance(field, dict):
                raise ValueError("Invalid native DAQ field.")
            meta = field.get("miscellaneous") or {}
            index = meta.get("index")
            if index in request["fields"].values():
                if index in selected:
                    raise ValueError("Duplicate requested subchannel index.")
                selected[index] = field
        values, invalid, descriptions, event_identity = {}, {}, {}, None
        for alias, index in request["fields"].items():
            field = selected.get(index)
            if field is None:
                raise ValueError("Requested subchannel is absent: " + str(index))
            meta = field["miscellaneous"]
            if field.get("type") != "A_TS_GSPECTRUM":
                raise ValueError("Only scalar A_TS_GSPECTRUM fields are supported.")
            if meta.get("daqname") != request["channel"]:
                raise ValueError("Native channel identity differs from the request.")
            data = field.get("data")
            if getattr(data, "shape", None) != (1, 2):
                raise ValueError("Expected scalar coordinate/value shape (1, 2).")
            timestamp, event_id = field.get("timestamp"), field.get("macropulse")
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp)
            ):
                raise ValueError("Native event requires a finite timestamp.")
            if isinstance(event_id, bool) or not isinstance(event_id, int):
                raise ValueError("Native event requires an integer macropulse.")
            current = (float(timestamp), event_id)
            if event_identity is not None and current != event_identity:
                raise ValueError("Selected fields have inconsistent timestamps or event IDs.")
            event_identity = current
            value = float(data[0, 1])
            values[alias] = value if math.isfinite(value) else None
            if not math.isfinite(value):
                invalid[alias] = (
                    "nan" if math.isnan(value)
                    else ("positive_infinity" if value > 0 else "negative_infinity")
                )
            comment = meta.get("comment")
            if comment is not None and (not isinstance(comment, str) or len(comment) > 4096):
                raise ValueError("Invalid or oversized native field description.")
            descriptions[alias] = {"index": index, "name": comment, "type": field["type"]}
        batch.append({
            "timestamp": event_identity[0], "event_id": event_identity[1],
            "values": values, "invalid_fields": invalid,
            "field_metadata": descriptions, "record_index": count,
        })
        count += 1
        if len(batch) >= request["batch_size"]:
            emit({"kind": "batch", "records": batch})
            batch = []
    if batch:
        emit({"kind": "batch", "records": batch})
    pydaq.disconnect()
    connected = False
    if identity(request["path"]) != request["identity"]:
        raise ValueError("Mounted source identity changed during native read.")
    emit({"kind": "done", "records": count})
except Exception as error:
    try:
        emit({"kind": "error", "message": type(error).__name__ + ": " + str(error)[:2000]})
    finally:
        sys.exit(1)
finally:
    if connected:
        try:
            pydaq.disconnect()
        except Exception:
            pass
"""


def iter_daq_records(
    source: MountedSource,
    *,
    channel: str,
    fields: Mapping[str, int],
    experiment: str,
    linac: str,
    python_executable: str | None = None,
    environment: Mapping[str, str] | None = None,
    batch_size: int = 128,
    timeout_seconds: float = 60.0,
    limit: int | None = None,
) -> Iterator[DaqRecord]:
    """Read explicit scalar subchannels using one isolated native process per source.

    ``fields`` maps output aliases to native subchannel indices. Actual native
    names/types are retained, never guessed. Nonfinite values become ``None``
    with an explicit ``invalid_fields`` reason. Timestamps and macropulse IDs
    are preserved numerically; their scientific interpretation belongs to the
    caller. Missing requested fields or unsupported shapes fail closed.

    ``python_executable`` can select a separate Python with a compatible pydaq
    ABI. ``environment`` extends its environment (for example library paths).
    No pydaq dependency is imported in the parent. Data is transferred in
    batches through a bounded pipe/queue, not a new process per record.
    ``timeout_seconds`` bounds waiting for startup, each batch or completion. Source
    size/time/identity are checked by ``MountedSource.open`` around the read.
    Close the iterator explicitly when stopping early, e.g. with
    ``contextlib.closing``. No content hash is implied by metadata validation.
    """
    for name, value in (("channel", channel), ("experiment", experiment), ("linac", linac)):
        if not isinstance(value, str) or not value or len(value) > 1024 or "\0" in value:
            raise ValueError(f"{name} must be a nonempty bounded string.")
    if not isinstance(fields, Mapping) or not 1 <= len(fields) <= 64:
        raise ValueError("Select between 1 and 64 explicit subchannel fields.")
    selected = dict(fields)
    for alias, index in selected.items():
        if not isinstance(alias, str) or not alias or len(alias) > 128:
            raise ValueError("Field aliases must be nonempty strings of at most 128 characters.")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= 65535:
            raise ValueError("Subchannel indices must be integers between 0 and 65535.")
    if len(set(selected.values())) != len(selected):
        raise ValueError("Subchannel indices must be unique.")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= 256
    ):
        raise ValueError("batch_size must be between 1 and 256.")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be finite and positive.")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
        raise ValueError("limit must be a nonnegative integer or None.")
    child_env = os.environ.copy()
    if environment is not None:
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in environment.items()):
            raise ValueError("environment must map string keys to string values.")
        child_env.update(environment)
    if limit == 0:
        return
    with source.open() as stream:
        stat = os.fstat(stream.fileno())
        request = {
            "path": str(source.path),
            "identity": [
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            ],
            "channel": channel,
            "fields": selected,
            "experiment": experiment,
            "linac": linac,
            "batch_size": batch_size,
            "limit": limit,
        }
        yield from _read_process(
            source, request, python_executable or sys.executable, child_env, timeout_seconds
        )


def _read_process(
    source: MountedSource,
    request: dict[str, Any],
    executable: str,
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> Iterator[DaqRecord]:
    payload = json.dumps(request, allow_nan=False).encode("utf8")
    try:
        process = subprocess.Popen(
            [executable, "-u", "-c", _CHILD],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(environment),
        )
    except OSError as error:
        raise DaqReadError("Could not start the selected native DAQ interpreter.") from error
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    messages: queue.Queue[bytes | Exception | None] = queue.Queue(maxsize=2)
    stopped = threading.Event()
    diagnostics = bytearray()

    def put(value: bytes | Exception | None) -> None:
        while not stopped.is_set():
            try:
                messages.put(value, timeout=0.1)
                return
            except queue.Full:
                pass

    def read_stdout() -> None:
        try:
            while not stopped.is_set():
                line = process.stdout.readline(_MAX_MESSAGE + 1)
                if not line:
                    put(None)
                    return
                if len(line) > _MAX_MESSAGE or not line.endswith(b"\n"):
                    put(DaqReadError("Native DAQ protocol message exceeds its size bound."))
                    return
                put(line)
        except (OSError, ValueError) as error:
            if not stopped.is_set():
                put(error)

    def read_stderr() -> None:
        try:
            while not stopped.is_set():
                chunk = process.stderr.read(4096)
                if not chunk:
                    return
                diagnostics.extend(chunk)
                del diagnostics[:-_MAX_STDERR]
        except (OSError, ValueError):
            pass

    def write_request() -> None:
        try:
            process.stdin.write(payload)
            process.stdin.close()
        except (OSError, ValueError) as error:
            if not stopped.is_set():
                put(DaqReadError(f"Native DAQ request transport failed: {error}"))
        finally:
            with suppress(OSError, ValueError):
                process.stdin.close()

    # Sending a legal request can fill the pipe before an interpreter finishes
    # startup. Keep that write under the same progress watchdog as output reads.
    threads = [
        threading.Thread(target=read_stdout, daemon=True),
        threading.Thread(target=read_stderr, daemon=True),
        threading.Thread(target=write_request, daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        count = 0
        while True:
            try:
                raw = messages.get(timeout=timeout_seconds)
            except queue.Empty as error:
                raise DaqReadError("Native DAQ reader timed out waiting for progress.") from error
            if isinstance(raw, Exception):
                raise DaqReadError(str(raw)) from raw
            if raw is None:
                detail = diagnostics.decode("utf8", errors="replace").strip()
                raise DaqReadError("Native DAQ reader exited without completion. " + detail)
            try:
                message = json.loads(raw)
                kind = message["kind"]
            except (ValueError, KeyError, TypeError) as error:
                raise DaqReadError("Invalid native DAQ protocol response.") from error
            if kind == "error":
                raise DaqReadError(str(message.get("message", "Native DAQ read failed.")))
            if kind == "done":
                if message.get("records") != count:
                    raise DaqReadError(
                        "Native DAQ completion count differs from delivered records."
                    )
                try:
                    code = process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired as error:
                    raise DaqReadError(
                        "Native DAQ reader did not exit after completion."
                    ) from error
                if code != 0:
                    raise DaqReadError(f"Native DAQ reader exited with status {code}.")
                return
            if kind != "batch" or not isinstance(message.get("records"), list):
                raise DaqReadError("Invalid native DAQ batch.")
            for row in message["records"]:
                if row.get("record_index") != count:
                    raise DaqReadError("Native DAQ record ordering differs from the stream.")
                count += 1
                yield DaqRecord(
                    timestamp=row["timestamp"],
                    event_id=row["event_id"],
                    channel=request["channel"],
                    values=row["values"],
                    invalid_fields=row["invalid_fields"],
                    field_metadata=row["field_metadata"],
                    source_path=source.source_path,
                    dataset_version=source.dataset_version,
                    record_index=row["record_index"],
                )
    except (BrokenPipeError, OSError) as error:
        raise DaqReadError("Native DAQ reader transport failed.") from error
    finally:
        stopped.set()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        for thread in threads:
            thread.join(timeout=1)
        for pipe in (process.stdin, process.stdout, process.stderr):
            pipe.close()
