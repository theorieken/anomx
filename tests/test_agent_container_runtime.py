"""Engine readiness, prepared runtime image updates and recoverable startup failures."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from anomx.agent.helpers import container_runtime as images
from anomx.agent.runtime_process import RuntimeProcessClient
from anomx.agent.store import AnomxHome


def result(args=(), code=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args, code, stdout, stderr)


def test_installed_client_with_stopped_engine_is_not_ready(monkeypatch):
    monkeypatch.setattr(images.shutil, "which", lambda _: "/bin/docker")
    calls = []
    monkeypatch.setattr(
        images.subprocess,
        "run",
        lambda cmd, **kw: calls.append(cmd) or result(code=1),
    )
    status = images.container_runtime_status("docker")
    assert status.installed and not status.ready
    assert calls == [["docker", "info"]]
    monkeypatch.setattr(images.subprocess, "run", lambda *a, **kw: result())
    assert images.container_runtime_status("podman").ready


def test_engine_start_on_mac_waits_for_readiness(monkeypatch):
    statuses = iter(
        [
            images.ContainerRuntimeStatus(True, False, "Start Docker Desktop."),
            images.ContainerRuntimeStatus(True, True, "Ready"),
        ]
    )
    monkeypatch.setattr(images, "container_runtime_status", lambda _: next(statuses))
    monkeypatch.setattr(images.sys, "platform", "darwin")
    calls = []
    monkeypatch.setattr(images.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or result())
    images.ensure_container_runtime("docker")
    assert calls == [["open", "-a", "Docker"]]


def test_missing_engine_error_survives_worker_boundary_without_traceback(tmp_path):
    home = AnomxHome(tmp_path / "home")
    home.ensure()
    runtime = RuntimeProcessClient(home, tmp_path, workspace_root=tmp_path)
    session = home.create_session(tmp_path, provider="missing", model="missing")
    try:
        runtime.start()
        assert runtime.wait_until_ready(timeout=20)
        assert runtime.init_sandbox(
            config={"sandbox_enabled": True, "sandbox_system": "absent-engine"},
            session_path=session.path,
        )
        with pytest.raises(images.SandboxStartupError, match="No agent commands were run") as error:
            runtime.ensure_sandbox_started(session.path)
        assert "Traceback" not in str(error.value)
    finally:
        runtime.close()


def test_image_is_cached_and_build_context_excludes_user_project(tmp_path, monkeypatch):
    root = tmp_path / "public-package"
    root.mkdir()
    (root / "__init__.py").write_text("__version__='test'")
    recipe = tmp_path / "Dockerfile"
    recipe.write_text("FROM python:3.12-slim\nCOPY anomx /opt/anomx\n")
    monkeypatch.setattr(images, "PACKAGE_ROOT", root)
    monkeypatch.setattr(images, "DOCKERFILE", recipe)
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1:3] == ["image", "inspect"]:
            return result(code=1)
        context = Path(cmd[-1])
        assert (context / "anomx" / "__init__.py").is_file()
        assert not (context / "auth.json").exists()
        return result()

    monkeypatch.setattr(images.subprocess, "run", run)
    first = images.runtime_image_revision()
    assert images.ensure_runtime_image("docker", first).endswith(first)
    assert calls[-1][1] == "build"
    (root / "__init__.py").write_text("__version__='updated'")
    assert images.runtime_image_revision() != first
    calls.clear()
    monkeypatch.setattr(images.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or result())
    images.ensure_runtime_image("docker", first)
    assert len(calls) == 1
