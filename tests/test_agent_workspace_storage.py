import json
from unittest.mock import Mock

import pytest

from anomx.agent.base.backends import TokenUsage, UsageSnapshot
from anomx.agent.exceptions import AgentBackendError, BackendFailure
from anomx.agent.helpers.workspace_storage import workspace_storage_prompt
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.store import AnomxHome


def usage(size):
    return {"available": True, "complete": True, "bytes": size, "limit_bytes": 100}


def runtime_at(tmp_path, size):
    home = AnomxHome(tmp_path / "home")
    handler = Mock(return_value={"usage": usage(size)})
    runtime = AgentRuntime(home, tmp_path, workspace_storage=handler, brain_sync=lambda: None)
    session = home.create_session(cwd=tmp_path, provider="openai", model="gpt-4.1").path
    return runtime, session, handler


@pytest.mark.parametrize(
    "size, wording, available",
    [
        (49, "", False),
        (50, "Be mindful", True),
        (79, "Be mindful", True),
        (80, "very important", True),
        (94, "very important", True),
    ],
)
def test_threshold_guidance_and_tools(tmp_path, size, wording, available):
    runtime, session, handler = runtime_at(tmp_path, size)
    runtime.prepare_workspace_request(session)
    assert (
        "manage_workspace_storage" in {tool.name for tool in runtime._available_tools()}
    ) == available
    assert wording in workspace_storage_prompt(runtime._workspace_usage)
    if size < 50:
        assert workspace_storage_prompt(runtime._workspace_usage) == ""
    assert handler.call_args.args[0]["refresh"]


@pytest.mark.parametrize("size", [95, 120])
def test_forced_cleanup_is_restricted_and_resumes_with_cumulative_usage(
    tmp_path, monkeypatch, size
):
    runtime, session, handler = runtime_at(tmp_path, size)
    activities, totals = [], []
    callbacks = runtime._with_usage_tracking(
        RuntimeCallbacks(
            workspace_activity=activities.append,
            usage=lambda snapshot: totals.append(snapshot.total.total_tokens),
        )
    )
    callbacks.usage(UsageSnapshot(total=TokenUsage(total_tokens=100), context_tokens=40))
    control = Mock()
    runtime.before_model_request = control

    def cleanup(child, cleanup_path, callbacks):
        assert child._workspace_cleanup_mode
        assert {tool.name for tool in child._available_tools()} == {"manage_workspace_storage"}
        assert child._tool_for_call("run_command") is None
        assert child.cancel_event is runtime.cancel_event
        assert cleanup_path != session
        assert "asynchronous maintenance prompt" in child._instructions(cleanup_path)
        child.before_model_request()
        callbacks.usage(UsageSnapshot(total=TokenUsage(total_tokens=25), context_tokens=2000))
        callbacks.usage(UsageSnapshot(total=TokenUsage(total_tokens=35), context_tokens=2500))
        handler.return_value = {"usage": usage(70)}
        return "Removed obsolete temporary responses."

    monkeypatch.setattr(AgentRuntime, "backend_response", cleanup)
    runtime.prepare_workspace_request(session)
    assert [item["status"] for item in activities] == ["running", "completed"]
    assert activities[-1]["bytes_after"] == 70
    assert runtime.last_usage_snapshot.context_tokens == 40
    callbacks.usage(UsageSnapshot(total=TokenUsage(total_tokens=140), context_tokens=60))
    assert totals == [100, 125, 135, 175]
    assert control.call_count == 1
    assert not runtime._workspace_cleanup_active
    assert "run_command" in {tool.name for tool in runtime._available_tools()}


def test_failed_cleanup_blocks_normal_work_without_retry_loop(tmp_path, monkeypatch):
    runtime, session, _ = runtime_at(tmp_path, 101)
    activities = []
    monkeypatch.setattr(
        AgentRuntime, "backend_response", lambda *args, **kwargs: "No safe candidates remain."
    )
    with pytest.raises(AgentBackendError, match="still at least 95%") as failure:
        runtime.prepare_workspace_request(
            session, callbacks=RuntimeCallbacks(workspace_activity=activities.append)
        )
    assert failure.value.code == "workspace_limit_exceeded"
    assert [item["status"] for item in activities] == ["running", "failed"]
    assert not runtime._workspace_cleanup_active


def test_cleanup_has_request_limit_and_propagates_provider_failures(tmp_path, monkeypatch):
    runtime, session, _ = runtime_at(tmp_path, 95)

    def cleanup(child, *args, **kwargs):
        for _ in range(25):
            child.before_model_request()

    monkeypatch.setattr(AgentRuntime, "backend_response", cleanup)
    with pytest.raises(AgentBackendError, match="request limit"):
        runtime.prepare_workspace_request(session)
    monkeypatch.setattr(
        AgentRuntime, "backend_response", lambda *args, **kwargs: BackendFailure("provider failed")
    )
    with pytest.raises(AgentBackendError, match="provider failed"):
        runtime.prepare_workspace_request(session)


def test_cleanup_cancellation_is_reported_without_failure(tmp_path, monkeypatch):
    runtime, session, _ = runtime_at(tmp_path, 100)
    activities = []

    def cancel(*args, **kwargs):
        runtime.cancel_event.set()
        return ""

    monkeypatch.setattr(AgentRuntime, "backend_response", cancel)
    runtime.prepare_workspace_request(
        session, callbacks=RuntimeCallbacks(workspace_activity=activities.append)
    )
    assert [item["status"] for item in activities] == ["running", "cancelled"]


def test_active_tools_must_settle_before_cleanup(tmp_path, monkeypatch):
    runtime, session, handler = runtime_at(tmp_path, 97)
    monkeypatch.setattr(runtime, "_running_command_states", lambda: [object()])
    cleanup = Mock()
    monkeypatch.setattr(runtime, "_run_workspace_cleanup", cleanup)
    runtime.prepare_workspace_request(session)
    cleanup.assert_not_called()
    assert {tool.name for tool in runtime._available_tools()} == {
        "check_command_status",
        "kill_command",
        "wait",
    }


def test_incomplete_storage_measurement_blocks_without_deleting(tmp_path):
    runtime, session, handler = runtime_at(tmp_path, 0)
    handler.return_value["usage"]["complete"] = False
    with pytest.raises(AgentBackendError, match="fully measured"):
        runtime.prepare_workspace_request(session)
    assert handler.call_count == 1


def test_large_candidate_result_stays_inline_for_restricted_cleanup(tmp_path):
    runtime, session, handler = runtime_at(tmp_path, 95)
    runtime._workspace_cleanup_mode = True
    handler.return_value = {
        "usage": usage(95),
        "candidates": [{"id": "chosen", "path": "x" * 20000}],
    }
    result = runtime._execute_tool(
        "manage_workspace_storage", {"action": "inspect"}, RuntimeCallbacks(), session_path=session
    )
    assert json.loads(result)["candidates"][0]["id"] == "chosen"
    assert not list(runtime.home.responses_dir.glob("*"))
