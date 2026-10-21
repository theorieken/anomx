"""Sandbox mode: containerised execution via Docker or Podman."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anomx.agent.helpers.container_runtime import (
    SandboxStartupError,
    ensure_container_runtime,
    ensure_runtime_image,
    runtime_image_revision,
)
from anomx.agent.helpers.container_runtime import (
    container_runtime_status as container_runtime_status,
)
from anomx.agent.helpers.tool_manager import (
    COMMAND_TIMEOUT_SECONDS,
    command_timeout_output,
)

StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class SandboxConfig:
    """Sandbox configuration from the user's config.toml."""

    enabled: bool = False
    system: str = "docker"
    method: str = "mount"
    cpu_limit: str = "2"
    ram_limit: str = "4g"
    hd_limit: str = "10g"
    copy_threshold_bytes: int = 2_000_000_000
    strategy: str = "keep"

    @property
    def method_label(self) -> str:
        return "mounted" if self.method == "mount" else "copied"

    def sandbox_context_prompt(self) -> str:
        return (
            "## Secure Agent Runtime\n"
            "Commands run inside the persistent anomx-agent container. "
            "Shared brain and skills are /anomx/brain and /anomx/skills. "
            "Each chat has /anomx/sessions/<id>/ with session.jsonl, tmp, "
            "sub_agents and an isolated task folder. Host projects are not mounted "
            "by default. Use mount_folder only when the user's task requires host "
            "files; explain why and obtain the explicit approval requested by the tool. "
            "Do not mount the launch directory merely because it was listed. "
            "Mounted data persists across runtime replacement; packages installed "
            "only in the container do not."
        )


SANDBOX_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".next",
        ".turbo",
        "dist",
        "build",
        "target",
        ".idea",
        ".vscode",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".tox",
        ".nox",
    }
)


def sandbox_config_from_dict(d: Mapping[str, Any]) -> SandboxConfig:
    return SandboxConfig(
        enabled=bool(d.get("sandbox_enabled", False)),
        system=str(d.get("sandbox_system", "docker")),
        method=str(d.get("sandbox_method", "mount")),
        cpu_limit=str(d.get("sandbox_cpu_limit", "2")),
        ram_limit=str(d.get("sandbox_ram_limit", "4g")),
        hd_limit=str(d.get("sandbox_hd_limit", "10g")),
        strategy=str(d.get("sandbox_strategy", "keep")),
    )


def detect_container_runtime() -> str | None:
    for candidate in ("docker", "podman"):
        if shutil.which(candidate) is not None:
            return candidate
    return None


def project_size_bytes(path: Path) -> int:
    total = 0
    abs_path = path.resolve()
    ignored = frozenset(SANDBOX_IGNORED_DIRS)

    for dirpath_str, dirnames, filenames in os.walk(abs_path):
        dirpath = Path(dirpath_str)
        dirnames[:] = [d for d in dirnames if d not in ignored and not d.startswith(".")]

        for name in filenames:
            if name in ignored or name.endswith((".pyc", ".pyo", ".swp", ".swo")):
                continue
            try:
                fp = dirpath / name
                total += fp.stat(follow_symlinks=False).st_size
            except OSError:
                pass
    return total


class SandboxCopySizeException(Exception):
    def __init__(self, size_bytes: int, threshold_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.threshold_bytes = threshold_bytes
        super().__init__(f"project size {size_bytes:,d} B exceeds {threshold_bytes:,d} B threshold")


class SandboxSession:
    """One persistent container, with isolated task folders and approved host mounts.

    Construction only prepares path metadata. ``start`` is called for submitted
    turns; exiting the CLI deliberately leaves the managed container running.
    """

    CONTAINER_NAME = "anomx-agent"
    CONTAINER_NAME_PREFIX = CONTAINER_NAME  # Compatibility for explicit management.

    def __init__(
        self,
        config: SandboxConfig,
        project_path: Path,
        sandbox_hash: str = "",
        runtime: str | None = None,
        runtime_directories: Mapping[str, Path] | None = None,
        session_key: str = "",
        session_path: Path | None = None,
        home_root: Path | None = None,
    ) -> None:
        from anomx.agent.store import resolve_anomx_home

        self.config = config
        self.project_path = project_path.expanduser().resolve()  # Context only; never mounted.
        self.home_root = (home_root or resolve_anomx_home()).expanduser().resolve()
        self.sessions_root = self.home_root / "sessions"
        self.session_key = session_key or (session_path.parent.name if session_path else "default")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", self.session_key):
            raise SandboxStartupError("The runtime session identifier is invalid.")
        self.session_root = self.sessions_root / self.session_key
        if session_path and session_path.resolve() != self.session_root / "session.jsonl":
            raise SandboxStartupError("The transcript must belong to this Anomx home and session.")
        self.task_path = self.session_root / "task"
        self._project_dir = Path("/anomx/sessions") / self.session_key / "task"
        provided = runtime_directories or {}
        self.runtime_directories = {
            "brain": Path(provided.get("brain", self.home_root / "brain")).expanduser().resolve(),
            "skills": Path(provided.get("skills", self.home_root / "skills"))
            .expanduser()
            .resolve(),
            "tmp": self.session_root / "tmp",
            "sub_agents": self.session_root / "sub_agents",
        }
        self._runtime = runtime or config.system
        self._container_id: str | None = None
        self._container_name = self.CONTAINER_NAME
        self._revision: str | None = None
        self._mounts_path = self.home_root / ".runtime-mounts.json"
        self._owner = hashlib.sha256(str(self.home_root).encode()).hexdigest()

    @property
    def container_id(self) -> str | None:
        return self._container_id

    @property
    def container_name(self) -> str:
        return self._container_name

    @property
    def is_running(self) -> bool:
        return self._container_id is not None and self._is_container_running(self._container_id)

    def _run(
        self,
        args: list[str],
        *,
        timeout: float = 120,
        capture: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self._runtime, *args],
            capture_output=capture,
            text=True,
            timeout=timeout,
            errors="backslashreplace",
        )

    def _run_captured(self, args: list[str], *, timeout: float = 120) -> str:
        result = self._run(args, timeout=timeout)
        output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        if result.returncode:
            output = f"[exit {result.returncode}]" + (f"\n{output}" if output else "")
        return output

    @contextmanager
    def _storage_lock(self, *, shared: bool = False, blocking: bool = True) -> Iterator[None]:
        """Keep replacement atomic with other CLI processes and their commands."""
        self.home_root.mkdir(parents=True, exist_ok=True)
        path = self.home_root / ".runtime.lock"
        if path.is_symlink():
            raise SandboxStartupError("The runtime lock must not be a symbolic link.")
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "a+b") as lock:
            if os.name == "nt":  # pragma: no cover - platform integration
                import msvcrt

                if lock.tell() == 0:
                    lock.write(b"0")
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
                try:
                    fcntl.flock(lock.fileno(), operation | (0 if blocking else fcntl.LOCK_NB))
                except BlockingIOError:
                    raise SandboxStartupError(
                        "anomx-agent is executing a command. "
                        "Retry changing the runtime after it finishes."
                    ) from None
                try:
                    yield
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _prepare_directories(self) -> None:
        for path in (
            self.sessions_root,
            self.session_root,
            self.task_path,
            *self.runtime_directories.values(),
        ):
            if path.resolve() != path.absolute():
                raise SandboxStartupError(f"Runtime data must not use a symbolic link: {path}")
            if path == Path.home() or self.home_root.is_relative_to(path):
                raise SandboxStartupError(
                    "Runtime data must not expose the host home or credentials."
                )
            path.mkdir(parents=True, exist_ok=True)

    def _read_mounts(self) -> dict[str, list[str]]:
        if self._mounts_path.is_symlink():
            raise SandboxStartupError("The approved-folder registry must not be a symbolic link.")
        if not self._mounts_path.exists():
            return {}
        try:
            value = json.loads(self._mounts_path.read_text())
        except (OSError, ValueError) as error:
            raise SandboxStartupError("The approved-folder registry could not be read.") from error
        if not isinstance(value, dict) or any(
            not isinstance(key, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", key)
            or not isinstance(paths, list)
            or any(not isinstance(path, str) for path in paths)
            for key, paths in value.items()
        ):
            raise SandboxStartupError("The approved-folder registry is invalid.")
        return value

    def _write_mounts(self, mounts: dict[str, list[str]]) -> None:
        with tempfile.NamedTemporaryFile(mode="w", dir=self.home_root, delete=False) as file:
            json.dump(mounts, file, sort_keys=True)
            file.flush()
            os.fsync(file.fileno())
            temporary = Path(file.name)
        temporary.replace(self._mounts_path)

    def validate_mount_folder(self, path: Path) -> Path:
        """Validate an explicit folder request without granting access or creating it."""
        if not path.is_absolute() or path.expanduser() != path:
            raise SandboxStartupError("mount_folder requires a full absolute host folder path.")
        if path.resolve() != path.absolute() or not path.is_dir():
            raise SandboxStartupError("Choose an existing folder using its real absolute path.")
        if (
            path == Path.home()
            or Path.home().is_relative_to(path)
            or self.home_root.is_relative_to(path)
        ):
            raise SandboxStartupError(
                "Mount a specific task folder, not the host home, root, or Anomx home."
            )
        if path.is_relative_to(self.home_root):
            raise SandboxStartupError(
                "Anomx runtime data is already available; its configuration stays private."
            )
        if path.is_relative_to(Path("/anomx")):
            raise SandboxStartupError("The /anomx namespace is reserved for runtime storage.")
        if ":" in str(path):
            raise SandboxStartupError("Container bind paths cannot contain a colon.")
        return path

    @staticmethod
    def _folder_target(path: Path) -> Path:
        identity = hashlib.sha256(str(path).encode()).hexdigest()[:16]
        return Path("/anomx/mounts") / identity

    @property
    def approved_folders(self) -> dict[Path, Path]:
        return {
            self._folder_target(Path(path)): self.validate_mount_folder(Path(path))
            for path in self._read_mounts().get(self.session_key, [])
        }

    def _mount_plan(self, mounts: dict[str, list[str]] | None = None) -> dict[Path, Path]:
        plan = {
            Path("/anomx/brain"): self.runtime_directories["brain"],
            Path("/anomx/skills"): self.runtime_directories["skills"],
            Path("/anomx/sessions"): self.sessions_root,
        }
        for source in tuple(plan.values()):
            plan[source] = source  # Existing tools may use absolute host references.
        selected = (self._read_mounts() if mounts is None else mounts).get(self.session_key, [])
        for raw in selected:
            source = self.validate_mount_folder(Path(raw))
            plan[self._folder_target(source)] = source
            plan[source] = source
        return plan

    def _fingerprint(self, plan: Mapping[Path, Path]) -> str:
        value = (
            sorted((str(target), str(source)) for target, source in plan.items()),
            self.config.cpu_limit,
            self.config.ram_limit,
        )
        return hashlib.sha256(repr(value).encode()).hexdigest()

    def _inspect(self, container_id: str) -> dict[str, Any]:
        result = self._run(["inspect", container_id], timeout=15)
        try:
            record = json.loads(result.stdout)[0]
            if result.returncode or not isinstance(record, dict):
                raise ValueError
            return record
        except (ValueError, IndexError, TypeError):
            raise SandboxStartupError("Could not inspect anomx-agent. Retry the message.") from None

    def _assert_owned(self, record: Mapping[str, Any]) -> None:
        labels = record.get("Config", {}).get("Labels") or {}
        if labels.get("io.anomx.managed") != "true" or labels.get("io.anomx.home") != self._owner:
            raise SandboxStartupError(
                "The container named anomx-agent belongs to another home "
                "or was not created by Anomx. "
                "Stop and rename it explicitly before retrying. It was not changed."
            )

    def _matches(self, record: Mapping[str, Any], plan: Mapping[Path, Path], revision: str) -> bool:
        labels = record.get("Config", {}).get("Labels") or {}
        actual = {
            (item.get("Destination"), item.get("Source")) for item in record.get("Mounts", [])
        }
        expected = {(str(target), str(source)) for target, source in plan.items()}
        return (
            labels.get("io.anomx.runtime-revision") == revision
            and labels.get("io.anomx.mounts") == self._fingerprint(plan)
            and actual == expected
        )

    def _assert_idle(self, container_id: str) -> None:
        if not self._is_container_running(container_id):
            return
        # Docker requires the PID column to map host processes to the container.
        format_args = ["pid,args"] if self._runtime == "podman" else ["-eo", "pid,args"]
        processes = self._run(["top", container_id, *format_args], timeout=15)
        if processes.returncode:
            raise SandboxStartupError(
                "Could not verify that anomx-agent is idle. It was not replaced."
            )
        lines = processes.stdout.strip().splitlines()
        if not lines or len(lines[0].split()) < 2 or lines[0].split()[0] != "PID":
            raise SandboxStartupError(
                "Could not verify that anomx-agent is idle. It was not replaced."
            )
        idle_command = 0
        init_process = 0
        for line in lines[1:]:
            fields = line.strip().split(maxsplit=1)
            if len(fields) != 2 or not fields[0].isdigit():
                raise SandboxStartupError(
                    "Could not verify the runtime's process list; it was not replaced."
                )
            try:
                parts = shlex.split(fields[1])
            except ValueError:
                raise SandboxStartupError(
                    "Could not verify the runtime's process list; it was not replaced."
                ) from None
            if parts == ["sleep", "infinity"] and idle_command == 0:
                idle_command += 1
                continue
            if (
                parts
                and Path(parts[0]).name in {"docker-init", "catatonit", "tini", "init"}
                and parts[-2:] == ["sleep", "infinity"]
                and init_process == 0
            ):
                init_process += 1
                continue
            raise SandboxStartupError(
                "anomx-agent has active commands or background processes. Finish them or stop "
                "the container explicitly before changing mounts or updating its image."
            )
        if idle_command != 1:
            raise SandboxStartupError(
                "Could not verify the runtime's idle process; it was not replaced."
            )

    def start(self, *, status_callback: StatusCallback | None = None) -> bool:
        ensure_container_runtime(self._runtime, status_callback)
        self._revision = runtime_image_revision()
        image = ensure_runtime_image(self._runtime, self._revision, status_callback)
        with self._storage_lock(shared=True):
            self._prepare_directories()
            plan = self._mount_plan()
            existing = self._find_existing_container()
            if existing is not None:
                record = self._inspect(existing)
                self._assert_owned(record)
                if self._matches(record, plan, self._revision) and self._is_container_running(
                    existing
                ):
                    self._container_id = existing
                    return True
        with self._storage_lock(blocking=False):
            return self._start_locked(image, self._revision, self._mount_plan(), status_callback)

    def _start_locked(
        self,
        image: str,
        revision: str,
        plan: Mapping[Path, Path],
        status_callback: StatusCallback | None = None,
    ) -> bool:
        existing = self._find_existing_container()
        if existing is not None:
            record = self._inspect(existing)
            self._assert_owned(record)
            if self._matches(record, plan, revision):
                self._container_id = existing
                if not self._is_container_running(existing):
                    started = self._run(["start", existing], timeout=30)
                    if started.returncode:
                        self._container_id = None
                        raise SandboxStartupError(f"Could not start anomx-agent: {started.stderr}")
                return True
            self._assert_idle(existing)
            if status_callback:
                status_callback("Updating anomx-agent · preserving mounted data")
            removed = self._run(["rm", "-f", existing], timeout=30)
            if removed.returncode:
                raise SandboxStartupError(f"Could not update anomx-agent: {removed.stderr}")
            self._container_id = None
        if status_callback:
            status_callback("Starting anomx-agent")
        arguments = [
            "run",
            "-d",
            "--name",
            self.CONTAINER_NAME,
            "--cpus",
            self.config.cpu_limit,
            "--memory",
            self.config.ram_limit,
            "--init",
            "--label",
            "io.anomx.managed=true",
            "--label",
            f"io.anomx.home={self._owner}",
            "--label",
            f"io.anomx.runtime-revision={revision}",
            "--label",
            f"io.anomx.mounts={self._fingerprint(plan)}",
            "-w",
            str(self._project_dir),
        ]
        for target, source in plan.items():
            arguments.extend(["-v", f"{source}:{target}"])
        arguments.extend([image, "sleep", "infinity"])
        result = self._run(arguments, timeout=60)
        if result.returncode:
            raise SandboxStartupError(f"Could not start anomx-agent: {result.stderr.strip()}")
        self._container_id = result.stdout.strip()
        if status_callback:
            status_callback("Anomx runtime ready")
        return True

    def mount_folder(self, path: Path, *, cancelled: Callable[[], bool] | None = None) -> Path:
        """Mount a folder after the caller has obtained explicit user approval."""
        path = self.validate_mount_folder(path)
        ensure_container_runtime(self._runtime)
        revision = runtime_image_revision()
        image = ensure_runtime_image(self._runtime, revision)
        with self._storage_lock(blocking=False):
            if cancelled is not None and cancelled():
                raise SandboxStartupError("Folder mount cancelled; no access was granted.")
            self._prepare_directories()
            mounts = self._read_mounts()
            selected = mounts.setdefault(self.session_key, [])
            if str(path) not in selected:
                selected.append(str(path))
            # Registry is committed only after the replacement is ready.
            self._start_locked(image, revision, self._mount_plan(mounts))
            self._write_mounts(mounts)
            self._revision = revision
        return self._folder_target(path)

    def ensure_started(self, *, status_callback: StatusCallback | None = None) -> bool:
        return self.start(status_callback=status_callback)

    def exec_command(
        self,
        command: str,
        *,
        timeout: float = COMMAND_TIMEOUT_SECONDS,
        current_dir: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> str:
        if self._container_id is None or self._revision is None:
            return "[sandbox: container not running]"
        with self._storage_lock(shared=True):
            record = self._inspect(self._container_id)
            self._assert_owned(record)
            if not self._matches(record, self._mount_plan(), self._revision):
                raise SandboxStartupError(
                    "The runtime changed. Retry the turn before running commands."
                )
            arguments = [
                "exec",
                "-i",
                "-w",
                str(self.container_path(current_dir or self.task_path)),
            ]
            for key, value in (env or {}).items():
                if key.startswith("ANOMX_"):
                    arguments.extend(["-e", f"{key}={value}"])
            arguments.extend([self._container_id, "/bin/bash", "-c", command])
            try:
                return self._run_captured(arguments, timeout=timeout)
            except subprocess.TimeoutExpired as error:
                return command_timeout_output(error, timeout)

    def _find_existing_container(self) -> str | None:
        result = self._run(
            ["ps", "-a", "--filter", f"name=^{self.CONTAINER_NAME}$", "--format", "{{.ID}}"],
            timeout=15,
        )
        if result.returncode:
            raise SandboxStartupError(f"Could not inspect anomx-agent: {result.stderr}")
        return result.stdout.strip() or None

    def _is_container_running(self, container_id: str) -> bool:
        result = self._run(
            ["ps", "--filter", f"id={container_id}", "--format", "{{.ID}}"], timeout=15
        )
        return result.returncode == 0 and result.stdout.strip() == container_id[:12]

    @property
    def path_aliases(self) -> dict[Path, Path]:
        return {
            self._project_dir: self.task_path,
            Path("/anomx/brain"): self.runtime_directories["brain"],
            Path("/anomx/skills"): self.runtime_directories["skills"],
            Path("/anomx/sessions") / self.session_key / "tmp": self.runtime_directories["tmp"],
            Path("/anomx/sessions") / self.session_key / "sub_agents": self.runtime_directories[
                "sub_agents"
            ],
            **self.approved_folders,
        }

    def container_path(self, path: Path) -> Path:
        resolved = path.expanduser().resolve()
        for target, source in sorted(
            self.path_aliases.items(), key=lambda item: -len(str(item[1]))
        ):
            if resolved.is_relative_to(source):
                return target / resolved.relative_to(source)
        raise ValueError(f"Path is not mounted in the agent runtime: {path}")

    def sandbox_context_prompt(self) -> str:
        mounted = "\n".join(
            f"- {target}: approved host folder {source}"
            for target, source in self.approved_folders.items()
        )
        return (
            self.config.sandbox_context_prompt() + f"\nTask directory: {self._project_dir}. "
            f"This session's data is /anomx/sessions/{self.session_key}/."
            + (f"\nExplicitly approved host folders:\n{mounted}" if mounted else "")
        )

    def stop(self) -> None:
        if self._container_id is not None:
            with self._storage_lock():
                self._assert_owned(self._inspect(self._container_id))
                self._run(["stop", self._container_id], timeout=30)
                self._container_id = None

    def remove(self) -> None:
        if self._container_id is not None:
            with self._storage_lock():
                self._assert_owned(self._inspect(self._container_id))
                self._run(["rm", "-f", self._container_id], timeout=30)
                self._container_id = None

    @staticmethod
    def _manage_named_container(
        runtime: str, *, remove: bool, home_root: Path | None = None
    ) -> int:
        from anomx.agent.store import resolve_anomx_home

        home = (home_root or resolve_anomx_home()).expanduser().resolve()
        expected_owner = hashlib.sha256(str(home).encode()).hexdigest()
        inspected = subprocess.run(
            [runtime, "inspect", SandboxSession.CONTAINER_NAME],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if inspected.returncode:
            return 0
        try:
            record = json.loads(inspected.stdout)[0]
            labels = record.get("Config", {}).get("Labels") or {}
            managed = labels.get("io.anomx.managed")
        except (ValueError, IndexError, TypeError):
            return 0
        if managed != "true" or labels.get("io.anomx.home") != expected_owner:
            return 0
        arguments = [runtime, "rm", "-f"] if remove else [runtime, "stop"]
        result = subprocess.run(
            [*arguments, SandboxSession.CONTAINER_NAME], capture_output=True, text=True, timeout=30
        )
        return int(result.returncode == 0)

    @staticmethod
    def stop_all(runtime: str = "docker", *, home_root: Path | None = None) -> int:
        """Explicitly stop the single managed runtime; never called on CLI exit."""
        return SandboxSession._manage_named_container(runtime, remove=False, home_root=home_root)

    @staticmethod
    def remove_all(runtime: str = "docker", *, home_root: Path | None = None) -> int:
        """Explicitly remove the single managed runtime, preserving mounted data."""
        return SandboxSession._manage_named_container(runtime, remove=True, home_root=home_root)

    def build_run_command(
        self,
        image: str,
        command: list[str],
        *,
        interactive: bool = True,
        rm: bool = True,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        """Build a one-off command using the same explicitly scoped mounts."""
        self._prepare_directories()
        args = [self._runtime, "run"]
        if interactive:
            args.append("-it")
        if rm:
            args.append("--rm")
        args.extend(["--cpus", self.config.cpu_limit, "--memory", self.config.ram_limit])
        args.extend(extra_args or [])
        for target, source in self._mount_plan().items():
            args.extend(["-v", f"{source}:{target}"])
        args.extend(["-w", workdir or str(self._project_dir)])
        for key, value in (env or {}).items():
            args.extend(["-e", f"{key}={value}"])
        return [*args, image, *command]

    def launch(self, command: list[str]) -> subprocess.Popen[bytes]:
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
