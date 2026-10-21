"""Container engine readiness and the CLI's automatically versioned tool image."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from anomx import __version__

StatusCallback = Callable[[str], None]
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = PACKAGE_ROOT / "agent" / "runtime_image" / "Dockerfile"


class SandboxStartupError(RuntimeError):
    """An actionable setup failure, safe to show without a Python traceback."""


@dataclass(frozen=True)
class ContainerRuntimeStatus:
    installed: bool
    ready: bool
    guidance: str


def container_runtime_status(runtime: str) -> ContainerRuntimeStatus:
    """Check the engine, not just the presence of its command-line client."""
    if runtime not in {"docker", "podman"} or shutil.which(runtime) is None:
        return ContainerRuntimeStatus(
            False, False, f"Install {runtime.title()} to use this runtime.",
        )
    try:
        result = subprocess.run(
            [runtime, "info"], capture_output=True,
            text=True, timeout=3,
        )
        if result.returncode == 0:
            return ContainerRuntimeStatus(True, True, "Ready")
    except (OSError, subprocess.TimeoutExpired):
        pass
    guidance = (
        "Docker Desktop will start when you open a chat."
        if runtime == "docker" and sys.platform == "darwin" else
        "Start your Podman machine, then retry."
        if runtime == "podman" else "Start the Docker engine, then retry."
    )
    return ContainerRuntimeStatus(True, False, guidance)


def ensure_container_runtime(
    runtime: str, status_callback: StatusCallback | None = None,
) -> None:
    status = container_runtime_status(runtime)
    if status.ready:
        return
    if status.installed and runtime == "docker" and sys.platform == "darwin":
        if status_callback:
            status_callback("Starting Docker Desktop · waiting for the engine")
        try:
            launched = subprocess.run(
                ["open", "-a", "Docker"], capture_output=True, text=True, timeout=10,
            )
            if launched.returncode == 0:
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    if container_runtime_status(runtime).ready:
                        return
                    time.sleep(1)
        except (OSError, subprocess.TimeoutExpired):
            pass
    raise SandboxStartupError(
        f"{runtime.title()} is not ready. {status.guidance} "
        "No agent commands were run. Retry your message after starting the engine, "
        "or choose host mode in /config → Manage Sandbox."
    )


def runtime_image_revision() -> str:
    """Include source as well as the recipe so editable installs also refresh safely."""
    digest = hashlib.sha256(DOCKERFILE.read_bytes())
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".md", ".json", ".toml"}:
            digest.update(str(path.relative_to(PACKAGE_ROOT)).encode())
            digest.update(path.read_bytes())
    return f"{__version__}-{digest.hexdigest()[:12]}"


def ensure_runtime_image(
    runtime: str, revision: str, status_callback: StatusCallback | None = None,
) -> str:
    """Build once from a minimal public-package context; never send project files or keys."""
    image = f"localhost/anomx-agent-runtime:{revision}"
    try:
        inspected = subprocess.run(
            [runtime, "image", "inspect", image], capture_output=True, text=True, timeout=15,
        )
        if inspected.returncode == 0:
            return image
        if status_callback:
            status_callback(
                f"Preparing Anomx runtime {__version__} · first start may take a few minutes"
            )
        with tempfile.TemporaryDirectory(prefix="anomx-runtime-image-") as directory:
            context = Path(directory)
            shutil.copyfile(DOCKERFILE, context / "Dockerfile")
            (context / ".dockerignore").write_text("build.log\n")
            shutil.copytree(
                PACKAGE_ROOT, context / "anomx",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
            )
            # Stream build output to disk instead of holding unbounded output in memory.
            with (context / "build.log").open("w+") as log:
                result = subprocess.run(
                    [runtime, "build", "--tag", image, "--label",
                     f"io.anomx.runtime-revision={revision}", str(context)],
                    stdout=log, stderr=subprocess.STDOUT, text=True, timeout=900,
                )
                if result.returncode:
                    log.seek(max(0, log.tell() - 5000))
                    detail = log.read()[-3000:]
                    raise SandboxStartupError(
                        "Could not prepare the Anomx runtime image. Check your network and "
                        f"container engine, then retry.\n{detail}"
                    )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SandboxStartupError(f"Could not prepare the Anomx runtime image: {exc}") from exc
    return image
