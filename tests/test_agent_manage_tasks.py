import json
from uuid import uuid4

import pytest

from anomx.agent.base.tools import ToolExecutionContext
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.store import AnomxHome
from anomx.agent.tools.manage_tasks import ManageTasksTool


@pytest.fixture
def tasks(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    home.ensure()
    home.set_platform_connection(
        url="https://example.invalid",
        token="test-token",
        user_email="agent@example.invalid",
        organization_url="example",
        hostname="test",
    )
    runtime = AgentRuntime(home, tmp_path, mode=AgentMode.AUTONOMOUS, agent_user=True)
    requests = []

    def call(connection, **kwargs):
        requests.append(kwargs)
        return {"ok": True, "response": {"task": {"id": str(uuid4())}}}

    monkeypatch.setattr("anomx.agent.tools.use_anomx_api.call_anomx_api", call)
    return runtime, requests, ToolExecutionContext(runtime=runtime, callbacks=RuntimeCallbacks())


def test_tasks_are_only_exposed_to_connected_agent_users(tasks):
    runtime, calls, context = tasks
    assert runtime._tool_for_call("manage_tasks") is not None
    assert (
        "tasks"
        in runtime._tool_for_call("manage_workspace_files").parameters["properties"]["area"]["enum"]
    )
    runtime.agent_user = False
    assert runtime._tool_for_call("manage_tasks") is None
    assert (
        "tasks"
        not in runtime._tool_for_call("manage_workspace_files").parameters["properties"]["area"][
            "enum"
        ]
    )
    assert not json.loads(ManageTasksTool().execute({"action": "list"}, context))["ok"]
    assert not calls


def test_task_mutations_preserve_reminders_and_cannot_override_owner(tasks):
    _, calls, context = tasks
    reminders = [
        {"kind": "weekly", "time": "08:00", "weekdays": [0], "timezone": "Europe/Berlin"},
        {"kind": "once", "at": "2027-03-12T12:00"},
    ]
    arguments = {
        "action": "add",
        "key": "weekly-check",
        "title": "Weekly check",
        "reminders": reminders,
        "user_id": str(uuid4()),
    }
    ManageTasksTool().execute(arguments, context)
    ManageTasksTool().execute(arguments, context)
    assert calls[0]["body"]["reminders"] == reminders
    assert calls[0]["body"] == calls[1]["body"]
    assert "user_id" not in calls[0]["body"]
    assert calls[0]["path"] == "/users/agent-tasks"


def test_tasks_respect_write_policy_and_validate_identifiers(tasks):
    runtime, calls, context = tasks
    result = ManageTasksTool().execute({"action": "edit", "task_id": "../../users"}, context)
    assert not json.loads(result)["ok"]
    runtime.tool_manager.mode = AgentMode.RECOMMEND
    result = ManageTasksTool().execute({"action": "delete", "task_id": str(uuid4())}, context)
    assert json.loads(result)["approved"] is False
    assert not calls


def test_live_instructions_refresh_without_changing_the_static_prompt(tasks):
    runtime, _, _ = tasks
    runtime.additional_instructions = "Standing assignment"
    current = ["Task: inspect logs"]
    runtime.live_instructions = lambda: current[0]
    assert "Task: inspect logs" in runtime._instruction_environment_sections()
    current[0] = "Task: inspect logs (completed)"
    sections = runtime._instruction_environment_sections()
    assert current[0] in sections
    assert "Task: inspect logs" not in sections
    assert "Standing assignment" in sections
