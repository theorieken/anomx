"""Persistent container lifecycle and explicit host-folder permission boundaries."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from anomx.agent.base.tools import ToolExecutionContext
from anomx.agent.helpers import sandbox as module
from anomx.agent.helpers.container_runtime import SandboxStartupError
from anomx.agent.helpers.tool_manager import ApprovalChoice, CliToolManager
from anomx.agent.runtime import RuntimeCallbacks
from anomx.agent.tools.mount_folder import MountFolderTool


class Engine:
    """Model Docker's immutable mounts and distinct IDs after replacement."""

    def __init__(self):
        self.record = None
        self.calls = []
        self.running = False
        self.number = 0
        self.processes = (
            "PID COMMAND\n100 /sbin/docker-init -- sleep infinity\n101 sleep infinity\n"
        )
        self.fail_start = False

    def run(self, args, **kwargs):
        self.calls.append(args)
        output, code = "", 0
        if args[0] == "ps":
            queried = next(item for item in args if item.startswith(("name=", "id=")))
            if self.record and (
                queried.startswith("name=")
                or (queried == "id=" + self.record["Id"] and self.running)
            ):
                output = self.record["Id"]
        elif args[0] == "inspect":
            if self.record and args[1] in {self.record["Id"], "anomx-agent"}:
                output = json.dumps([self.record])
            else:
                code = 1
        elif args[0] == "run":
            self.number += 1
            labels = dict(
                args[i + 1].split("=", 1) for i, value in enumerate(args) if value == "--label"
            )
            mounts = [args[i + 1].split(":", 1) for i, value in enumerate(args) if value == "-v"]
            self.record = {
                "Id": f"container{self.number}",
                "Config": {"Labels": labels},
                "Mounts": [{"Source": source, "Destination": target} for source, target in mounts],
            }
            self.running = True
            output = self.record["Id"]
        elif args[0] == "top":
            assert args[2:] in (["-eo", "pid,args"], ["pid,args"])
            output = self.processes
        elif args[0] == "rm":
            self.record, self.running = None, False
        elif args[0] == "start":
            code = int(self.fail_start)
            self.running = not self.fail_start
        elif args[0] == "stop":
            self.running = False
        elif args[0] == "exec":
            output = "executed"
        return subprocess.CompletedProcess(args, code, output, "engine stopped" if code else "")


@pytest.fixture
def environment(tmp_path, monkeypatch):
    launch = tmp_path / "project"
    launch.mkdir()
    (launch / "secret.txt").write_text("never mounted by default")
    home = tmp_path / "home"
    home.mkdir()
    (home / "auth.json").write_text("private")
    (home / "config.toml").write_text("private")
    engine = Engine()
    monkeypatch.setattr(module, "ensure_container_runtime", lambda *_: None)
    monkeypatch.setattr(module, "runtime_image_revision", lambda: "current")
    monkeypatch.setattr(module, "ensure_runtime_image", lambda *_: "anomx:current")
    monkeypatch.setattr(
        module.SandboxSession, "_run", lambda self, args, **kw: engine.run(args, **kw)
    )

    def session(key="one", project=launch):
        return module.SandboxSession(
            module.SandboxConfig(), project, home_root=home, session_key=key
        )

    return SimpleNamespace(home=home, launch=launch, engine=engine, session=session)


def test_default_runtime_reuses_one_container_without_mounting_launch_folder(environment):
    env = environment
    first, second = env.session(), env.session("two")
    assert not env.engine.calls  # Construction and task navigation are lazy.
    assert first.start() and second.start()
    assert first.container_name == second.container_name == "anomx-agent"
    assert first.container_id == second.container_id
    assert first.task_path.is_dir() and not first.task_path.is_symlink()
    assert first.task_path != second.task_path
    sources = {item["Source"] for item in env.engine.record["Mounts"]}
    assert sources == {str(env.home / name) for name in ("brain", "skills", "sessions")}
    assert str(env.launch) not in sources and str(env.home) not in sources
    assert all("docker.sock" not in source for source in sources)
    assert len([call for call in env.engine.calls if call[0] == "run"]) == 1


def test_another_launch_project_does_not_change_default_mounts(environment, tmp_path):
    first = environment.session()
    first.start()
    another = tmp_path / "another-project"
    another.mkdir()
    second = environment.session("two", another)
    second.start()
    assert first.container_id == second.container_id


def test_approved_mount_is_persistent_for_its_task_and_absent_from_other_tasks(environment):
    env = environment
    first = env.session()
    first.start()
    original = first.container_id
    (first.task_path / "result.txt").write_text("preserved")
    target = first.mount_folder(env.launch)
    assert first.container_id != original
    assert str(target).startswith("/anomx/mounts/")
    assert first.approved_folders[target] == env.launch
    resumed = env.session()
    resumed.start()
    assert resumed.container_id == first.container_id
    second = env.session("two")
    second.start()
    assert not second.approved_folders
    assert str(env.launch) not in {item["Source"] for item in env.engine.record["Mounts"]}
    assert (first.task_path / "result.txt").read_text() == "preserved"
    with pytest.raises(SandboxStartupError):
        first.exec_command("pwd")  # Its old handle must never execute in the replacement.


def test_image_upgrade_prepares_replacement_before_retiring_idle_container(
    environment, monkeypatch
):
    env = environment
    first = env.session()
    first.start()
    old_id = first.container_id
    monkeypatch.setattr(module, "runtime_image_revision", lambda: "new")
    events = []
    monkeypatch.setattr(
        module, "ensure_runtime_image", lambda *_: events.append("prepared") or "anomx:new"
    )
    original_run = env.engine.run

    def run(args, **kwargs):
        if args[0] == "rm":
            assert events == ["prepared"]
        return original_run(args, **kwargs)

    monkeypatch.setattr(env.engine, "run", run)
    first.start()
    assert first.container_id != old_id
    assert env.engine.record["Config"]["Labels"]["io.anomx.runtime-revision"] == "new"


@pytest.mark.parametrize("background", ["python worker.py", "sleep infinity"])
def test_background_process_blocks_mount_replacement(environment, background):
    env = environment
    session = env.session()
    session.start()
    old_id = session.container_id
    env.engine.processes += "102 " + background + "\n"
    with pytest.raises(SandboxStartupError, match="background processes"):
        session.mount_folder(env.launch)
    assert env.engine.record["Id"] == old_id
    assert not session.approved_folders


@pytest.mark.parametrize(
    "processes", ["COMMAND\nsleep infinity\n", "PID COMMAND\n? sleep infinity\n"]
)
def test_unverifiable_process_list_refuses_replacement(environment, processes):
    env = environment
    session = env.session()
    session.start()
    old_id = session.container_id
    env.engine.processes = processes
    with pytest.raises(SandboxStartupError, match="Could not verify"):
        session.mount_folder(env.launch)
    assert env.engine.record["Id"] == old_id
    assert not session.approved_folders


@pytest.mark.skipif(__import__("os").name == "nt", reason="Shared flock is POSIX specific")
def test_active_command_lock_refuses_mount_change_without_waiting(environment):
    env = environment
    session = env.session()
    session.start()
    ready, release = threading.Event(), threading.Event()

    def command():
        with session._storage_lock(shared=True):
            ready.set()
            release.wait(5)

    worker = threading.Thread(target=command)
    worker.start()
    assert ready.wait(2)
    try:
        with pytest.raises(SandboxStartupError, match="executing a command"):
            session.mount_folder(env.launch)
    finally:
        release.set()
        worker.join(5)
    assert not session.approved_folders


def test_foreign_named_container_is_never_removed(environment):
    env = environment
    session = env.session()
    session.start()
    env.engine.record["Config"]["Labels"]["io.anomx.home"] = "another-home"
    with pytest.raises(SandboxStartupError, match="another home"):
        session.start()
    assert not any(call[0] == "rm" for call in env.engine.calls)


def test_stopped_container_restarts_and_failed_restart_does_not_report_ready(environment):
    env = environment
    session = env.session()
    session.start()
    env.engine.running = False
    session.start()
    assert env.engine.running
    env.engine.running = False
    env.engine.fail_start = True
    with pytest.raises(SandboxStartupError, match="engine stopped"):
        session.start()
    assert session.container_id is None


def test_task_symlink_is_rejected_and_never_exposes_project(environment):
    env = environment
    session = env.session()
    session.session_root.mkdir(parents=True)
    session.task_path.symlink_to(env.launch)
    with pytest.raises(SandboxStartupError, match="symbolic link"):
        session.start()
    assert not any(call[0] == "run" for call in env.engine.calls)


@pytest.mark.parametrize("folder", ["/", "relative", "~", "home", "auth"])
def test_mount_validation_rejects_broad_or_non_absolute_paths(environment, folder):
    env = environment
    path = (
        env.home
        if folder == "home"
        else env.home / "auth.json"
        if folder == "auth"
        else Path(folder)
    )
    with pytest.raises(SandboxStartupError):
        env.session().validate_mount_folder(path)


def tool_context(env, callback=None):
    session = env.session()
    session._prepare_directories()

    class Runtime:
        sandbox_session = session
        platform_chat_id = ""
        cancel_event = threading.Event()
        tool_manager = CliToolManager(session.task_path)

        @property
        def trusted_roots(self):
            return (session.task_path, *session.approved_folders.values())

    runtime = Runtime()
    return ToolExecutionContext(runtime, RuntimeCallbacks(approval=callback)), session


@pytest.mark.parametrize("decision", [None, ApprovalChoice.REJECT, ApprovalChoice.ALWAYS_REJECT])
def test_mount_tool_requires_explicit_approval(environment, decision):
    env = environment
    callback = None if decision is None else lambda _: decision
    context, session = tool_context(env, callback)
    output = MountFolderTool(statement_description="Working").execute(
        {
            "path": str(env.launch),
            "reason": "Read the requested source code",
            "statement": "Open source",
        },
        context,
    )
    assert json.loads(output)["approved"] is False
    assert not env.engine.calls and not session.approved_folders


@pytest.mark.parametrize("cancel_when", ["before", "during"])
def test_mount_tool_cancellation_does_not_grant_access(environment, cancel_when):
    env = environment
    context, session = tool_context(env)

    def approve(_):
        context.runtime.cancel_event.set()
        return ApprovalChoice.ALLOW

    context = ToolExecutionContext(context.runtime, RuntimeCallbacks(approval=approve))
    if cancel_when == "before":
        context.runtime.cancel_event.set()
    output = MountFolderTool(statement_description="Working").execute(
        {"path": str(env.launch), "reason": "Read requested code", "statement": "Open source"},
        context,
    )
    assert json.loads(output)["approved"] is False
    assert not env.engine.calls and not session.approved_folders


def test_mount_tool_shows_scope_and_grants_only_requested_folder(environment):
    env = environment
    requests = []
    context, session = tool_context(
        env, lambda request: requests.append(request) or ApprovalChoice.ALLOW
    )
    output = MountFolderTool(statement_description="Working").execute(
        {
            "path": str(env.launch),
            "reason": "Fix the specified source code",
            "statement": "Open source",
        },
        context,
    )
    payload = json.loads(output)
    assert payload["approved"] and payload["scope"] == "current_task"
    assert requests[0].requires_user_approval
    assert str(env.launch) in requests[0].reason and "read/write" in requests[0].reason
    assert "remembered" in requests[0].reason and "Other tasks" in requests[0].reason
    assert context.runtime.tool_manager.path_inside_workspace(env.launch / "secret.txt")
    assert not context.runtime.tool_manager.path_inside_workspace(env.home / "auth.json")
    assert session.container_path(env.launch / "secret.txt") == Path(payload["path"]) / "secret.txt"


def test_unrelated_expired_mount_does_not_block_another_session(environment):
    env = environment
    (env.home / ".runtime-mounts.json").write_text(json.dumps({"old": ["/missing/old/project"]}))
    assert env.session().start()


def test_mount_approval_is_not_requested_again_for_the_same_task(environment):
    env = environment
    context, session = tool_context(env, lambda _: ApprovalChoice.ALLOW)
    tool = MountFolderTool(statement_description="Working")
    args = {"path": str(env.launch), "reason": "Read requested code", "statement": "Open source"}
    assert json.loads(tool.execute(args, context))["approved"]
    # Restored task approval is sufficient even without an interactive provider.
    restored = ToolExecutionContext(context.runtime, RuntimeCallbacks())
    assert json.loads(tool.execute(args, restored))["approved"]
    assert len(session.approved_folders) == 1


def test_cancellation_during_image_preparation_does_not_mount(environment, monkeypatch):
    env = environment
    context, session = tool_context(env, lambda _: ApprovalChoice.ALLOW)

    def prepare(*_):
        context.runtime.cancel_event.set()
        return "image:prepared"

    monkeypatch.setattr(module, "ensure_runtime_image", prepare)
    output = MountFolderTool(statement_description="Working").execute(
        {"path": str(env.launch), "reason": "Read requested code", "statement": "Open source"},
        context,
    )
    assert not json.loads(output)["approved"]
    assert not session.approved_folders
    assert not any(call[0] in {"run", "rm"} for call in env.engine.calls)
