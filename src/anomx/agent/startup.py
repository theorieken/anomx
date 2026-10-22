"""Non-blocking startup tasks for terminal interfaces."""

from __future__ import annotations

import json
import multiprocessing
import queue
import threading
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from anomx import __version__

VERSION_CHECK_TIMEOUT_SECONDS = 3


@dataclass(frozen=True)
class UpdateCheck:
    """A successful check with an optional update, or an unreachable index."""

    checked: bool
    available_version: str | None = None


class CliStartupProcess:
    """Run platform and update checks outside the interface process."""

    def __init__(self, home_root: Path) -> None:
        self.home_root = home_root.expanduser()
        self._context = multiprocessing.get_context("spawn")
        self._results: Queue[dict[str, Any]] | None = None
        self._process: BaseProcess | None = None
        self.version_ready = False
        self.latest_version: str | None = None
        self.platform_ready = False
        self.platform_connected = False
        self.provider_models: dict[str, tuple[str, ...] | None] = {}

    def start(self) -> None:
        """Start the worker and return immediately."""

        if self._process is not None:
            return
        results = self._context.Queue()
        process = self._context.Process(
            target=_run_startup_tasks,
            args=(str(self.home_root), results, __version__),
            name="anomx-cli-startup",
            daemon=True,
        )
        process.start()
        self._results = results
        self._process = process

    def poll(self) -> None:
        """Apply every result currently available without waiting."""

        if self._results is None:
            return
        while True:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                return
            kind = str(result.get("kind") or "")
            if kind == "version":
                self.version_ready = True
                value = str(result.get("value") or "").strip()
                self.latest_version = value or None
            elif kind == "platform":
                self.platform_ready = True
                self.platform_connected = bool(result.get("value"))
            elif kind == "models":
                value = result.get("value")
                self.provider_models[str(result["provider"])] = (
                    tuple(value) if isinstance(value, (list, tuple)) else None
                )

    def close(self) -> None:
        """Release worker resources, stopping only unfinished startup work."""

        process = self._process
        if process is not None:
            process.join(timeout=0.2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        if self._results is not None:
            self._results.close()
        self._process = None
        self._results = None


def check_for_update(current_version: str) -> UpdateCheck:
    """Check PyPI within a wall-clock deadline, including DNS resolution."""

    results: queue.Queue[UpdateCheck] = queue.Queue(maxsize=1)

    def fetch() -> None:
        try:
            with urlopen(
                "https://pypi.org/pypi/anomx/json",
                timeout=VERSION_CHECK_TIMEOUT_SECONDS,
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
            latest = str(data.get("info", {}).get("version") or "").strip()
            if not latest:
                results.put(UpdateCheck(False))
                return
            available = latest if _version_key(latest) > _version_key(current_version) else None
            results.put(UpdateCheck(True, available))
        except Exception:
            results.put(UpdateCheck(False))

    threading.Thread(target=fetch, name="anomx-update-check", daemon=True).start()
    try:
        return results.get(timeout=VERSION_CHECK_TIMEOUT_SECONDS)
    except queue.Empty:
        return UpdateCheck(False)


def check_latest_version(current_version: str) -> str | None:
    """Return a newer PyPI version, or None when current or unreachable."""

    return check_for_update(current_version).available_version


def _run_startup_tasks(home_root: str, results: Queue[dict[str, Any]], version: str) -> None:
    from anomx.agent.helpers.platform_client import heartbeat_platform_connection
    from anomx.agent.model_catalog import discover_provider_models
    from anomx.agent.store import AnomxHome

    home = AnomxHome(Path(home_root))

    def check_version() -> None:
        results.put({"kind": "version", "value": check_latest_version(version) or ""})

    def connect_platform() -> None:
        connected = False
        if home.has_platform_connection():
            try:
                connected = heartbeat_platform_connection(home)
            except Exception:
                connected = False
        results.put({"kind": "platform", "value": connected})

    def check_models(provider: str) -> None:
        api_keys = home.load_auth().get("api_keys", {})
        models = discover_provider_models(provider, api_keys.get(provider))
        results.put({"kind": "models", "provider": provider, "value": models})

    workers = (
        threading.Thread(target=check_version, daemon=True),
        threading.Thread(target=connect_platform, daemon=True),
        *(
            threading.Thread(target=check_models, args=(provider,), daemon=True)
            for provider in home.connected_backend_keys()
        ),
    )
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()


def _version_key(value: str) -> tuple[tuple[int, int | str], ...]:
    parts: list[tuple[int, int | str]] = []
    for part in value.split("."):
        if part.isdigit():
            parts.append((0, int(part)))
        else:
            parts.append((1, part))
    parts.extend((0, 0) for _ in range(max(0, 8 - len(parts))))
    return tuple(parts)


__all__ = ["CliStartupProcess", "UpdateCheck", "check_for_update", "check_latest_version"]
