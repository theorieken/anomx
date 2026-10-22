"""Durable page journals for resumable, credential-scoped dataset downloads.

This module stores raw JSON responses independently of DataFrame materialization.
Each committed page is durable before control returns to the caller, so stopping
iteration or restarting Python does not discard already downloaded observations.
The caller owns authentication and chooses a directory scoped to its server,
credentials, and dataset reference. Requested prefix lengths and batch sizes are
deliberately absent from journal identity: a short read can later extend into a
complete download.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO

_PAGE_NAME = re.compile(r"page-(\d{12})\.json\Z")
_FORMAT = 1


class DownloadConflict(RuntimeError):
    """Another writer owns a download, or an append uses an obsolete cursor."""


class DatasetDownload:
    """Keep a validated, append-only prefix of a remote dataset on disk.

    Use this object as a context manager. Its exclusive nonblocking lock lasts
    until context exit; a second writer gets :class:`DownloadConflict` rather
    than silently overwriting pages. Operating-system locks are released when a
    process exits, including an interrupted interpreter.

    Parameters
    ----------
    directory
        Dedicated directory already scoped to the API origin, authorization
        identity, and dataset reference. For example, derive it from the
        unbounded dataset cache path with ``cache.path.with_suffix('.parts')``.
        Never include an API token directly in the directory name.
    definition
        Freshly authorized dataset definition. ``version``, ``cache_token``,
        ``kind``, ``time_column``, and ``time_unit`` bind the journal to its
        source and interpretation. The authorization token is not stored.
    max_age
        Explicit stale-data interval in seconds for completed downloads when
        the host has no content token, measured from the oldest included page.
        ``None`` explicitly permits version-pinned snapshot reuse without an
        expiry; it does not imply freshness against mutable source records.
        Without a token, completed journals are otherwise re-read unless the
        interval permits reuse. Matching partial downloads
        resume by pinned source version even after the interval expires; hosts
        without content validators cannot guarantee snapshot consistency for
        mutable records during such a resume.
    refresh
        Discard the previous journal under the lock before starting again.

    Notes
    -----
    Every page has its own checksum and atomic file replacement. The manifest
    contains only constant-size progress metadata, never a growing page list.
    Recovery rebuilds progress from the contiguous valid page prefix, including
    a page committed just before a crash prevented the manifest update. A bad
    page discards only that page and its suffix. No pickle is deserialized.
    Files are created with mode 0600 and directories with mode 0700 on POSIX.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        definition: Mapping[str, Any],
        max_age: float | None = 0,
        refresh: bool = False,
    ) -> None:
        version = definition.get("version")
        if not isinstance(version, str) or not version:
            raise ValueError("A download requires a nonempty source version.")
        if max_age is not None and (
            isinstance(max_age, bool) or not math.isfinite(max_age) or max_age < 0
        ):
            raise ValueError("max_age must be finite nonnegative seconds or None.")
        self.directory = Path(directory).expanduser()
        self.version = version
        self.max_age = max_age
        self.refresh = refresh
        self._has_token = bool(definition.get("cache_token"))
        identity = {
            key: definition.get(key)
            for key in ("version", "cache_token", "kind", "time_column", "time_unit")
        }
        self._identity = hashlib.sha256(_encode(identity)).hexdigest()
        self._lock: BinaryIO | None = None
        self._page_count = 0
        self._row_count = 0
        self._next_cursor: str | None = None
        self._complete = False
        self._first_created: float | None = None
        self._last_created: float | None = None
        self._columns: list[str] | None = None
        self._seen_cursors: set[str] = set()
        self.reused = False

    @property
    def page_count(self) -> int:
        """Number of contiguous, committed pages, including empty pages."""
        return self._page_count

    @property
    def row_count(self) -> int:
        """Number of observations durably stored in committed pages."""
        return self._row_count

    @property
    def next_cursor(self) -> str | None:
        """Cursor for the next request; inspect ``complete`` to distinguish EOF."""
        return self._next_cursor

    @property
    def complete(self) -> bool:
        """Whether the last committed response explicitly reached the source end."""
        return self._complete

    def __enter__(self) -> DatasetDownload:
        if self._lock is not None:
            raise DownloadConflict("This download context is already open.")
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.directory / ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        lock = os.fdopen(descriptor, "r+b")
        try:
            _acquire(lock)
            self._lock = lock
            if self.refresh:
                self.invalidate()
            else:
                self._recover()
            self.reused = self._page_count > 0
            return self
        except BaseException:
            self._lock = None
            lock.close()
            raise

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release the writer lock; committed data remains available for reuse."""
        if self._lock is not None:
            lock, self._lock = self._lock, None
            try:
                _release(lock)
            finally:
                lock.close()

    @classmethod
    def clear(cls, directory: str | Path) -> None:
        """Discard saved observations without resolving or contacting their source.

        Local deletion does not require a current source definition or working
        credentials. The existing writer lock and durable invalidation marker
        still protect against concurrent downloads and interrupted deletion.
        Keep the lock file in place: unlinking it could let another process
        acquire a different lock while an existing waiter retains the old one.
        """
        directory = Path(directory).expanduser()
        if not directory.exists():
            return
        with cls(directory, definition={"version": "local-cache-clear"}, refresh=True):
            pass

    def iter_pages(self) -> Iterator[dict[str, Any]]:
        """Replay committed raw responses in order while this context is open."""
        self._require_open()
        for index in range(self._page_count):
            self._require_open()
            envelope = self._read_page(self._page_path(index))
            yield envelope["response"]

    def append(
        self,
        response: Mapping[str, Any],
        *,
        request_cursor: str | None,
    ) -> None:
        """Durably commit a response before yielding its records to application code.

        ``request_cursor`` must be the current ``next_cursor``. This protects
        against accidentally committing a response fetched before another page
        advanced the journal. Response source versions, columns, and cursor
        progression are validated before any file is written. An I/O failure
        propagates; reopening recovers any page that did commit successfully.
        """
        self._require_open()
        if self._complete:
            raise DownloadConflict("The dataset download is already complete.")
        if request_cursor != self._next_cursor:
            raise DownloadConflict("The response belongs to an obsolete download cursor.")
        page = dict(response)
        self._validate_response(page)
        created = time.time()
        envelope = {
            "format": _FORMAT,
            "identity": self._identity,
            "index": self._page_count,
            "request_cursor": request_cursor,
            "created": created,
            "response": page,
        }
        envelope["sha256"] = hashlib.sha256(_encode(envelope)).hexdigest()
        _atomic_json(self._page_path(self._page_count), envelope)
        self._advance(page, created)
        self._write_manifest()

    def invalidate(self) -> None:
        """Remove this journal's pages and reset progress while retaining its lock.

        Call after a host reports a changed strong content token during a
        transfer. A new context with the new authorized definition can then
        download a coherent source selection.
        """
        self._require_open()
        # Persist invalidation intent before removing files. If the process
        # stops halfway through deletion, reopening must finish the reset
        # instead of resurrecting the surviving pages of the old snapshot.
        self._write_manifest(invalidating=True)
        self._discard_suffix(0)
        self._reset_state()
        self.reused = False
        self._write_manifest()

    def _require_open(self) -> None:
        if self._lock is None:
            raise RuntimeError("Use DatasetDownload inside a context manager.")

    def _reset_state(self) -> None:
        self._page_count = self._row_count = 0
        self._next_cursor = None
        self._complete = False
        self._first_created = None
        self._last_created = None
        self._columns = None
        self._seen_cursors.clear()

    def _recover(self) -> None:
        self._reset_state()
        # An expired completed download must not become a fresh partial one
        # merely because a later page was damaged. The checksummed manifest
        # preserves that completion fact independently of individual pages.
        try:
            manifest = _verified_json(self.directory / "manifest.json")
        except (FileNotFoundError, ValueError, TypeError, KeyError, OverflowError):
            manifest = {}
        if manifest.get("invalidating") is True:
            self.invalidate()
            return
        if (
            manifest.get("identity") == self._identity
            and manifest.get("complete") is True
            and not self._fresh_completed(manifest.get("first_created"))
        ):
            self.invalidate()
            return
        paths = sorted(path for path in self.directory.iterdir() if _PAGE_NAME.fullmatch(path.name))
        for path in paths:
            if self._complete or path != self._page_path(self._page_count):
                break
            try:
                envelope = self._read_page(path)
                if (
                    envelope["identity"] != self._identity
                    or envelope["index"] != self._page_count
                    or envelope["request_cursor"] != self._next_cursor
                ):
                    break
                self._validate_response(envelope["response"])
                created = envelope["created"]
                if isinstance(created, bool) or not isinstance(created, (int, float)):
                    break
                if not math.isfinite(created):
                    break
            except (ValueError, TypeError, KeyError, OverflowError):
                break
            self._advance(envelope["response"], created)

        self._discard_suffix(self._page_count)
        # Partial transfers intentionally remain resumable across TTL expiry.
        # A completed response is a cache, and needs an explicit freshness policy.
        if self._complete and not self._fresh_completed(self._first_created):
            self.invalidate()
            return
        self._write_manifest()

    def _fresh_completed(self, created: object) -> bool:
        if self._has_token or self.max_age is None:
            return True
        if isinstance(created, bool) or not isinstance(created, (int, float)):
            return False
        age = time.time() - created
        return self.max_age > 0 and 0 <= age <= self.max_age

    def _validate_response(self, page: Mapping[str, Any]) -> None:
        if page.get("version") != self.version:
            raise ValueError("Dataset source version changed during download.")
        records = page.get("records", [])
        columns = page.get("columns", [])
        if not isinstance(records, list):
            raise ValueError("Dataset page records must be a JSON list.")
        if (
            not isinstance(columns, list)
            or any(not isinstance(column, str) for column in columns)
            or len(set(columns)) != len(columns)
        ):
            raise ValueError("Dataset page columns must be unique strings.")
        if self._columns is not None and columns != self._columns:
            raise ValueError("Dataset column schema changed during download.")
        cursor = page.get("next_cursor")
        if cursor is not None and (
            not isinstance(cursor, str)
            or not cursor
            or cursor == self._next_cursor
            or cursor in self._seen_cursors
        ):
            raise ValueError("Dataset page must advance to a new nonempty cursor.")

    def _advance(self, page: Mapping[str, Any], created: float) -> None:
        self._page_count += 1
        self._row_count += len(page.get("records", []))
        self._next_cursor = page.get("next_cursor")
        self._complete = self._next_cursor is None
        self._first_created = (
            created if self._first_created is None else min(self._first_created, created)
        )
        self._last_created = created
        self._columns = page.get("columns", [])
        if self._next_cursor is not None:
            self._seen_cursors.add(self._next_cursor)

    def _page_path(self, index: int) -> Path:
        return self.directory / f"page-{index:012d}.json"

    def _read_page(self, path: Path) -> dict[str, Any]:
        envelope = _verified_json(path)
        if not isinstance(envelope.get("response"), dict):
            raise ValueError("Invalid download page response.")
        return envelope

    def _discard_suffix(self, first: int) -> None:
        for path in self.directory.iterdir():
            match = _PAGE_NAME.fullmatch(path.name)
            if (match and int(match[1]) >= first) or path.name.startswith(".download-"):
                path.unlink()

    def _write_manifest(self, *, invalidating: bool = False) -> None:
        manifest = {
            "format": _FORMAT,
            "identity": self._identity,
            "version": self.version,
            "page_count": self._page_count,
            "row_count": self._row_count,
            "next_cursor": self._next_cursor,
            "complete": self._complete,
            "invalidating": invalidating,
            "first_created": self._first_created,
            "last_created": self._last_created,
        }
        manifest["sha256"] = hashlib.sha256(_encode(manifest)).hexdigest()
        _atomic_json(self.directory / "manifest.json", manifest)


def _encode(value: object) -> bytes:
    """Canonical safe JSON used for both persistence and integrity checks."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _verified_json(path: Path) -> dict[str, Any]:
    envelope = json.loads(path.read_text())
    if not isinstance(envelope, dict) or envelope.get("format") != _FORMAT:
        raise ValueError("Invalid download journal format.")
    checksum = envelope.pop("sha256", None)
    if not isinstance(checksum, str) or hashlib.sha256(_encode(envelope)).hexdigest() != checksum:
        raise ValueError("Download journal checksum does not match.")
    return envelope


def _atomic_json(path: Path, value: object) -> None:
    payload = _encode(value)
    descriptor, temporary = tempfile.mkstemp(prefix=".download-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _acquire(stream: BinaryIO) -> None:
    try:
        if sys.platform == "win32":
            import msvcrt

            stream.seek(0)
            if not stream.read(1):
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
            raise DownloadConflict("Another process is downloading this dataset.") from exc
        raise


def _release(stream: BinaryIO) -> None:
    if sys.platform == "win32":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
