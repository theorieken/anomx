"""CLI sandbox startup is lazy and grants no implicit access to launch folders."""

from __future__ import annotations

import json

import pytest

from anomx.agent.base.tools import ToolExecutionContext
from anomx.agent.helpers.container_runtime import SandboxStartupError
from anomx.agent.helpers.sandbox import SandboxSession
from anomx.agent.helpers.utils import AgentKind
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.store import AnomxHome
from anomx.agent.tools.read_file import ReadFileTool


@pytest.fixture
def sandbox_chat(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "secret.txt").write_text("private project contents", encoding="utf-8")
    home = AnomxHome(tmp_path / "home")
    config = home.load_config()
    config.update(
        sandbox_enabled=True, sandbox_system="docker", provider="missing", model="missing",
    )
    home.save_config(config)
    session = home.create_session(project, provider="missing", model="missing")
    return home, project, session


def test_container_descriptor_is_lazy_and_isolates_launch_folder(sandbox_chat, monkeypatch):
    home, project, session = sandbox_chat

    def unexpected_start(*args, **kwargs):
        pytest.fail("Opening a runtime or preparing its paths must not start Docker")

    monkeypatch.setattr(SandboxSession, "start", unexpected_start)
    runtime = AgentRuntime(home, project)
    assert runtime.sandbox_session is None
    assert runtime.init_sandbox(session_path=session.path)
    task = session.path.parent / "task"
    assert runtime.cwd == task
    assert runtime.workspace_root == runtime.tool_manager.root == task
    assert runtime.tool_manager.current_dir == task
    assert runtime.tool_manager.strict_workspace
    assert not runtime.tool_manager.path_inside_workspace(project)
    assert not runtime.tool_manager.path_inside_workspace(home.auth_path)
    context = runtime._sandbox_instruction_section()
    assert str(project) in context and "secret.txt" in context
    assert "private project contents" not in context
    assert "not mounted" in context
    output = ReadFileTool(statement_description="Reading").execute(
        {"path": str(project / "secret.txt")},
        ToolExecutionContext(runtime, RuntimeCallbacks(), session.path),
    )
    assert "error" in json.loads(output)
    assert "private project contents" not in output


def test_first_turn_starts_runtime_before_goal_and_exit_leaves_it_running(
    sandbox_chat, monkeypatch,
):
    home, project, session = sandbox_chat
    starts = []

    def start(sandbox, **kwargs):
        starts.append(sandbox.session_key)
        sandbox.task_path.mkdir(parents=True, exist_ok=True)
        assert not (project / "GOAL.md").exists()
        return True

    def forbidden(*args, **kwargs):
        pytest.fail("Runtime shutdown must leave the persistent container running")

    monkeypatch.setattr(SandboxSession, "start", start)
    monkeypatch.setattr(SandboxSession, "stop", forbidden)
    monkeypatch.setattr(SandboxSession, "remove", forbidden)
    runtime = AgentRuntime(home, project)
    runtime.configure_session_features(session.path, goal_enabled=True)
    assert starts == []
    runtime.backend_response(session.path)
    assert starts == [session.path.parent.name]
    assert (session.path.parent / "task" / "GOAL.md").is_file()
    assert not (project / "GOAL.md").exists()
    runtime.backend_response(session.path)
    assert starts == [session.path.parent.name] * 2  # Revalidate container each turn.
    runtime.shutdown(session.path)


def test_failed_container_start_prevents_provider_and_host_goal(sandbox_chat, monkeypatch):
    home, project, session = sandbox_chat

    def fail_start(*args, **kwargs):
        raise SandboxStartupError("Engine unavailable")

    monkeypatch.setattr(SandboxSession, "start", fail_start)
    monkeypatch.setattr(
        "anomx.agent.runtime.backend_for_provider",
        lambda *_: pytest.fail("Provider must wait for runtime startup"),
    )
    runtime = AgentRuntime(home, project, goal_enabled=True)
    with pytest.raises(SandboxStartupError, match="Engine unavailable"):
        runtime.backend_response(session.path)
    assert not (project / "GOAL.md").exists()


def test_subagent_keeps_own_cwd_with_shared_container(sandbox_chat, monkeypatch):
    home, project, session = sandbox_chat
    main = AgentRuntime(home, project)
    main.init_sandbox(session_path=session.path)
    worker_path = main.home.sub_agents_dir / "research"
    worker_path.mkdir(parents=True)
    child = AgentRuntime(
        main.home, worker_path, workspace_root=main.workspace_root, agent_kind=AgentKind.SUB,
    )
    child._sandbox_session = main.sandbox_session
    monkeypatch.setattr(SandboxSession, "start", lambda *_args, **_kwargs: True)
    child.ensure_sandbox_started(worker_path / "session.jsonl")
    assert child.cwd == worker_path


def test_container_task_symlink_cannot_grant_host_access(sandbox_chat):
    home, project, session = sandbox_chat
    (session.path.parent / "task").symlink_to(project, target_is_directory=True)
    runtime = AgentRuntime(home, project)
    with pytest.raises(SandboxStartupError, match="symbolic links"):
        runtime.init_sandbox(session_path=session.path)


def test_reused_main_runtime_changes_task_and_drops_previous_mounts(sandbox_chat, monkeypatch):
    home, project, first = sandbox_chat
    second = home.create_session(project, provider="missing", model="missing")
    monkeypatch.setattr(SandboxSession, "start", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        SandboxSession,
        "approved_folders",
        property(
            lambda sandbox: {project: project} if sandbox.session_key == first.session_id else {}
        ),
    )
    runtime = AgentRuntime(home, project)
    runtime.ensure_sandbox_started(first.path)
    assert runtime.tool_manager.path_inside_workspace(project)
    runtime.ensure_sandbox_started(second.path)
    assert runtime.cwd == second.path.parent / "task"
    assert runtime.home.tmp_dir == second.path.parent / "tmp"
    assert not runtime.tool_manager.path_inside_workspace(project)
    assert not runtime.tool_manager.path_inside_workspace(first.path.parent / "task")


def test_explicitly_disabling_container_restores_host_execution_roots(sandbox_chat):
    home, project, session = sandbox_chat
    runtime = AgentRuntime(home, project, workspace_root=project)
    runtime.init_sandbox(session_path=session.path)
    runtime.init_sandbox(config={"sandbox_enabled": False}, session_path=session.path)
    assert runtime.sandbox_session is None
    assert runtime.cwd == runtime.workspace_root == runtime.tool_manager.root == project
    assert runtime.tool_manager.current_dir == project
    assert runtime.tool_manager.path_aliases == {}
    assert not runtime.tool_manager.strict_workspace


def test_direct_tool_session_switch_drops_old_host_mount_without_starting_engine(
    sandbox_chat, monkeypatch,
):
    home, project, first = sandbox_chat
    second = home.create_session(project, provider="missing", model="missing")
    monkeypatch.setattr(
        SandboxSession, "start", lambda *_args, **_kwargs: pytest.fail("Unexpected engine start"),
    )
    monkeypatch.setattr(
        SandboxSession,
        "approved_folders",
        property(
            lambda sandbox: {project: project} if sandbox.session_key == first.session_id else {}
        ),
    )
    runtime = AgentRuntime(home, project)
    arguments = {"path": str(project / "secret.txt")}
    first_result = runtime._execute_tool("read", arguments, RuntimeCallbacks(), first.path)
    assert json.loads(first_result)["content"] == "private project contents"
    second_result = runtime._execute_tool("read", arguments, RuntimeCallbacks(), second.path)
    assert "error" in json.loads(second_result)
    assert runtime.cwd == second.path.parent / "task"


def test_mount_tool_is_only_available_in_cli_container(sandbox_chat):
    home, project, session = sandbox_chat
    runtime = AgentRuntime(home, project)
    assert "mount_folder" not in {tool.name for tool in runtime._available_tools()}
    runtime.init_sandbox(session_path=session.path)
    assert "mount_folder" in {tool.name for tool in runtime._available_tools()}
    runtime.platform_chat_id = "cloud-chat"
    assert "mount_folder" not in {tool.name for tool in runtime._available_tools()}


def test_platform_runtime_does_not_adopt_cli_container_preferences(sandbox_chat, monkeypatch):
    home, project, session = sandbox_chat
    runtime = AgentRuntime(home, project, platform_chat_id="cloud-chat")
    monkeypatch.setattr(
        SandboxSession, "start", lambda *_args, **_kwargs: pytest.fail("CLI container started"),
    )
    runtime.backend_response(session.path)
    assert runtime.cwd == project
    assert runtime.sandbox_session is None


def test_config_no_longer_offers_container_cleanup_on_exit(sandbox_chat):
    from anomx.agent.ui.views.config import ConfigViewMixin

    home, _project, _session = sandbox_chat
    choices = ConfigViewMixin()._sandbox_menu_choices(home.load_config())
    assert "strategy" not in [choice.value for choice in choices]
    assert any("approval" in choice.detail for choice in choices)
