"""Sandbox mode: containerised execution via Docker or Podman."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from contextlib import suppress
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
    strategy: str = "stop"

    @property
    def method_label(self) -> str:
        return "mounted" if self.method == "mount" else "copied"

    def sandbox_context_prompt(self) -> str:
        return (
            "## Sandbox\n"
            "- Commands run inside a managed Anomx container with Python, Git, Node.js "
            "and common tools. "
            "The task directory is /anomx/task (also /project).\n"
            "- Shared memory and skills are /anomx/brain and /anomx/skills.\n"
            "- Chat buffers and child agents are /anomx/tmp and /anomx/sub_agents.\n"
            f"- Project files are {self.method_label} into the container.\n"
            "- Only mounted directories persist across runtime image upgrades. "
            "You can install packages, create files, and run commands normally."
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
        strategy=str(d.get("sandbox_strategy", "stop")),
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
    """A persistent sandbox container session tied to a project hash.

    Usage:
        session = SandboxSession(config, project_path, sandbox_hash="abc123")
        session.start(status_callback=print)
        session.exec_command("ls -la")
        session.exec_command("python script.py")
        session.stop()
    """

    CONTAINER_NAME_PREFIX = "anomx-sandbox-"

    def __init__(
        self,
        config: SandboxConfig,
        project_path: Path,
        sandbox_hash: str = "",
        runtime: str | None = None,
        runtime_directories: Mapping[str, Path] | None = None,
        session_key: str = "",
    ) -> None:
        self.config = config
        self.project_path = project_path.resolve()
        self._runtime = (runtime or config.system)
        self._container_id: str | None = None
        self._project_dir = Path("/anomx/task")
        self.runtime_directories = {
            name: path.expanduser().resolve()
            for name, path in (runtime_directories or {}).items()
            if name in {"brain", "skills", "tmp", "sub_agents"}
        }
        identity = hashlib.sha256(
            repr((str(self.project_path), session_key, sorted(
                (name, str(path)) for name, path in self.runtime_directories.items()
            ))).encode()
        ).hexdigest()[:12]
        self._container_name = f"{self.CONTAINER_NAME_PREFIX}{sandbox_hash}-{identity}"
        self._prepared_project: tuple[Path, str] | None = None

    @property
    def container_id(self) -> str | None:
        return self._container_id

    @property
    def container_name(self) -> str:
        return self._container_name

    @property
    def is_running(self) -> bool:
        if self._container_id is None:
            return False
        result = subprocess.run(
            [self._runtime, "ps", "--filter", f"id={self._container_id}",
             "--format", "{{.ID}}"],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0 and result.stdout.strip() == self._container_id[:12]

    def _run(
        self,
        args: list[str],
        *,
        timeout: float = 120,
        capture: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [self._runtime] + args
        return subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=timeout,
            errors="backslashreplace",
        )

    def _run_captured(
        self,
        args: list[str],
        *,
        timeout: float = 120,
    ) -> str:
        result = self._run(args, timeout=timeout, capture=True)
        output = "\n".join(
            part for part in (result.stdout.strip(), result.stderr.strip()) if part
        )
        if result.returncode != 0:
            if not output:
                output = f"[exit {result.returncode}]"
            else:
                output = f"[exit {result.returncode}]\n{output}"
        return output

    # --- Lifecycle ---

    def start(
        self,
        *,
        status_callback: StatusCallback | None = None,
    ) -> bool:
        ensure_container_runtime(self._runtime, status_callback)
        revision = runtime_image_revision()
        image = ensure_runtime_image(self._runtime, revision, status_callback)
        existing_id = self._find_existing_container()
        if existing_id is not None:
            inspected = self._run(["inspect", existing_id], timeout=15)
            try:
                record = json.loads(inspected.stdout)[0]
                previous_revision = (record.get("Config", {}).get("Labels") or {}).get(
                    "io.anomx.runtime-revision", ""
                )
            except (ValueError, IndexError, TypeError):
                raise SandboxStartupError(
                    "Could not inspect the existing runtime. Retry startup."
                ) from None
            if previous_revision == revision:
                self._container_id = existing_id
                if not self._is_container_running(existing_id):
                    if status_callback:
                        status_callback("Starting Anomx runtime")
                    started = self._run(["start", existing_id], timeout=30)
                    if started.returncode != 0:
                        self._container_id = None
                        raise SandboxStartupError(
                            f"Could not start Anomx runtime: {started.stderr}"
                        )
                if status_callback:
                    status_callback("Anomx runtime ready")
                return True
            if status_callback:
                status_callback("Updating Anomx runtime · keeping mounted files")
            # Update only this exact chat's managed container, after its replacement image exists.
            removed = self._run(["rm", "-f", existing_id], timeout=30)
            if removed.returncode != 0:
                raise SandboxStartupError(f"Could not update Anomx runtime: {removed.stderr}")

        if status_callback:
            status_callback("Starting Anomx runtime")

        if status_callback and self.config.method == "copy":
            status_callback("Evaluating project size")

        project_dir, mount_arg = self._prepare_project()

        run_args: list[str] = [
            "run", "-d",
            "--name", self._container_name,
            "--cpus", self.config.cpu_limit,
            "--memory", self.config.ram_limit,
            "-v", mount_arg,
            "-w", str(project_dir),
            "--init",
            "--label", f"io.anomx.runtime-revision={revision}",
            "--label", "io.anomx.managed=true",
        ]
        run_args.extend(self._directory_mount_args())
        run_args.extend([image, "sleep", "infinity"])

        result = self._run(run_args, timeout=60, capture=True)
        if result.returncode != 0:
            raise SandboxStartupError(
                f"Could not start Anomx runtime: {result.stderr.strip()}"
            )

        self._container_id = result.stdout.strip()

        if status_callback:
            status_callback("Anomx runtime ready")

        return True

    def stop(self) -> None:
        if self._container_id is None:
            return
        with suppress(Exception):
            self._run(["stop", self._container_id], timeout=30)
        self._container_id = None

    def remove(self) -> None:
        if self._container_id is None:
            return
        with suppress(Exception):
            self._run(["rm", "-f", self._container_id], timeout=30)
        self._container_id = None

    def ensure_started(
        self,
        *,
        status_callback: StatusCallback | None = None,
    ) -> bool:
        return self.start(status_callback=status_callback)

    @staticmethod
    def stop_all(runtime: str = "docker") -> int:
        result = subprocess.run(
            [runtime, "ps", "-q", "--filter",
             f"name={SandboxSession.CONTAINER_NAME_PREFIX}"],
            capture_output=True, text=True, timeout=30,
        )
        ids = [cid for cid in result.stdout.strip().splitlines() if cid.strip()]
        if not ids:
            return 0
        subprocess.run(
            [runtime, "stop"] + ids,
            capture_output=True, text=True, timeout=60,
        )
        return len(ids)

    @staticmethod
    def remove_all(runtime: str = "docker") -> int:
        result = subprocess.run(
            [runtime, "ps", "-aq", "--filter",
             f"name={SandboxSession.CONTAINER_NAME_PREFIX}"],
            capture_output=True, text=True, timeout=30,
        )
        ids = [cid for cid in result.stdout.strip().splitlines() if cid.strip()]
        if not ids:
            return 0
        subprocess.run(
            [runtime, "rm", "-f"] + ids,
            capture_output=True, text=True, timeout=30,
        )
        return len(ids)

    def exec_command(
        self,
        command: str,
        *,
        timeout: float = COMMAND_TIMEOUT_SECONDS,
        current_dir: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> str:
        if self._container_id is None:
            return "[sandbox: container not running]"

        exec_args: list[str] = [
            "exec", "-i",
            "-w", str(self.container_path(current_dir or self.project_path)),
        ]
        for key, value in (env or {}).items():
            if key.startswith("ANOMX_"):
                exec_args.extend(["-e", f"{key}={value}"])
        exec_args.append(self._container_id)
        exec_args.extend(["/bin/bash", "-c", command])

        try:
            return self._run_captured(exec_args, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            return command_timeout_output(error, timeout)

    # --- Internals ---

    def _find_existing_container(self) -> str | None:
        result = self._run(
            ["ps", "-a", "--filter", f"name=^{self._container_name}$",
             "--format", "{{.ID}}"],
            timeout=15,
        )
        cid = result.stdout.strip()
        if result.returncode != 0:
            raise SandboxStartupError(f"Could not inspect Anomx runtime: {result.stderr}")
        return cid if cid else None

    def _is_container_running(self, container_id: str) -> bool:
        result = self._run(
            ["ps", "--filter", f"id={container_id}", "--format", "{{.ID}}"],
            timeout=15,
        )
        return result.stdout.strip() == container_id[:12]

    def _prepare_project(self) -> tuple[Path, str]:
        if self._prepared_project is not None:
            return self._prepared_project
        project_dir = self._project_dir
        if self.config.method == "mount":
            mount_arg = f"{self.project_path}:{project_dir}"
            self._prepared_project = project_dir, mount_arg
            return self._prepared_project

        size = project_size_bytes(self.project_path)
        if size > self.config.copy_threshold_bytes:
            raise SandboxCopySizeException(size, self.config.copy_threshold_bytes)

        temp_root = Path(tempfile.mkdtemp(prefix="anomx-sandbox-"))
        dest = temp_root / "project"
        shutil.copytree(
            str(self.project_path),
            str(dest),
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", "*.pyc",
                "node_modules", ".venv", "venv",
                ".next", ".turbo", "dist", "build", "target",
            ),
            symlinks=True,
        )
        mount_arg = f"{dest}:{project_dir}"
        self._prepared_project = project_dir, mount_arg
        return self._prepared_project

    @property
    def path_aliases(self) -> dict[Path, Path]:
        """Map container-facing paths back to authorized host directories."""

        return {
            self._project_dir: self.project_path,
            Path("/project"): self.project_path,
            **{Path("/anomx") / name: path for name, path in self.runtime_directories.items()},
        }

    def container_path(self, path: Path) -> Path:
        """Translate a host path to its mounted container location."""

        resolved = path.expanduser().resolve()
        aliases = sorted(self.path_aliases.items(), key=lambda item: -len(str(item[1])))
        for target, source in aliases:
            if resolved.is_relative_to(source):
                return target / resolved.relative_to(source)
        raise ValueError(f"Path is not mounted in the agent runtime: {path}")

    def _directory_mount_args(self) -> list[str]:
        arguments: list[str] = []
        # Keep absolute references from file tools and skills usable in commands.
        _, project_mount = self._prepare_project()
        project_source = project_mount.rsplit(":", 1)[0]
        arguments.extend(["-v", f"{project_source}:{self.project_path}"])
        arguments.extend(["-v", f"{project_source}:/project"])
        for name, source in self.runtime_directories.items():
            source.mkdir(parents=True, exist_ok=True)
            arguments.extend(["-v", f"{source}:/anomx/{name}"])
            arguments.extend(["-v", f"{source}:{source}"])
        return arguments

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
        cmd = [self._runtime, "run"]

        if interactive:
            cmd.append("-it")
        if rm:
            cmd.append("--rm")

        cmd.extend(["--cpus", self.config.cpu_limit])
        cmd.extend(["--memory", self.config.ram_limit])

        if extra_args:
            cmd.extend(extra_args)

        project_dir, mount_arg = self._prepare_project()
        cmd.extend(["-v", mount_arg])
        cmd.extend(self._directory_mount_args())

        if workdir is not None:
            cmd.extend(["-w", workdir])
        else:
            cmd.extend(["-w", str(project_dir)])

        if env:
            for k, v in env.items():
                cmd.extend(["-e", f"{k}={v}"])

        cmd.append(image)
        cmd.extend(command)
        return cmd

    def launch(self, command: list[str]) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
