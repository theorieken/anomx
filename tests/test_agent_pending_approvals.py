"""Approval review is concurrent; approved actions execute once when collected."""

import json
import threading
from types import SimpleNamespace

import pytest

from anomx.agent.base.backends import BaseBackend
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.helpers.tool_manager import ApprovalChoice, CommandRiskEvaluation
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.store import AnomxHome


def runtime_at(path):
    return AgentRuntime(AnomxHome(path / "home"), path, mode=AgentMode.STANDARD)


def write_arguments(**extra):
    return {"statement": "Write result", "command": "printf done >> result.txt", **extra}


def execute(runtime, callback, **extra):
    return json.loads(
        runtime._execute_tool(
            "run_command",
            write_arguments(**extra),
            RuntimeCallbacks(approval=callback),
        )
    )


def test_async_approval_leaves_action_pending_while_other_work_continues(tmp_path):
    runtime = runtime_at(tmp_path)
    release = threading.Event()
    requests = []

    def approve(request):
        requests.append(request)
        release.wait(2)
        return ApprovalChoice.ALLOW

    pending = execute(runtime, approve)
    approval_id = pending["approval_id"]
    assert pending["approval_pending"] and not (tmp_path / "result.txt").exists()
    assert runtime.approval_request_active(approval_id)
    independent = json.loads(
        runtime._execute_tool(
            "run_command",
            {"statement": "Read workspace", "command": "pwd"},
            RuntimeCallbacks(),
        )
    )
    assert independent["approved"]
    assert runtime.check_approval(approval_id)["status"] == "pending"
    release.set()
    completed = runtime.check_approval(approval_id, 2)
    assert completed["status"] == "completed"
    assert json.loads(completed["result"])["approved"]
    assert (tmp_path / "result.txt").read_text() == "done"
    assert runtime.check_approval(approval_id) == completed
    assert (tmp_path / "result.txt").read_text() == "done"
    assert requests[0].request_id == approval_id and not requests[0].wait_for_approval
    assert not runtime.approval_request_active(approval_id)


@pytest.mark.parametrize("decision", [ApprovalChoice.REJECT, ApprovalChoice.ALWAYS_REJECT])
def test_denied_async_approval_never_executes(tmp_path, decision):
    runtime = runtime_at(tmp_path)
    runtime.session_rejected_commands = set()
    runtime.tool_manager.session_rejected_commands = runtime.session_rejected_commands
    pending = execute(runtime, lambda _: decision)
    completed = runtime.check_approval(pending["approval_id"], 2)
    assert not json.loads(completed["result"])["approved"]
    assert not (tmp_path / "result.txt").exists()
    if decision == ApprovalChoice.ALWAYS_REJECT:
        assert runtime.session_rejected_commands
        result = runtime.tool_manager.run_command(
            write_arguments()["command"],
            "Try same action",
            lambda _: pytest.fail("Must not review"),
        )
        assert not result.approved


def test_synchronous_critical_approval_waits_and_has_cancellable_identity(tmp_path):
    runtime = runtime_at(tmp_path)
    seen = []

    def approve(request):
        seen.append(request)
        assert request.request_id and request.wait_for_approval
        assert runtime.approval_request_active(request.request_id)
        assert not (tmp_path / "result.txt").exists()
        return ApprovalChoice.ALLOW

    result = execute(runtime, approve, wait_for_approval=True)
    assert result["approved"] and "approval_pending" not in result
    assert (tmp_path / "result.txt").read_text() == "done"
    assert not runtime.approval_request_active(seen[0].request_id)


@pytest.mark.parametrize("blocking", [False, True])
def test_cancelled_approval_cannot_execute_on_late_allow(tmp_path, blocking):
    runtime = runtime_at(tmp_path)

    def approve(request):
        runtime.abort_current_turn()
        assert not runtime.approval_request_active(request.request_id)
        return ApprovalChoice.ALLOW

    result = execute(runtime, approve, wait_for_approval=blocking)
    if not blocking:
        result = runtime.check_approval(result["approval_id"], 2)
    assert not result.get("approved", False)
    assert not (tmp_path / "result.txt").exists()
    runtime._join_approval_workers()


def test_pending_action_keeps_original_directory_and_does_not_change_new_directory(tmp_path):
    runtime = runtime_at(tmp_path)
    pending = execute(runtime, lambda _: ApprovalChoice.ALLOW)
    other = tmp_path / "other"
    other.mkdir()
    runtime.tool_manager.current_dir = other
    runtime.check_approval(pending["approval_id"], 2)
    assert (tmp_path / "result.txt").read_text() == "done"
    assert not (other / "result.txt").exists()
    assert runtime.tool_manager.current_dir == other


def test_final_output_cannot_abandon_pending_approval(tmp_path, monkeypatch):
    runtime = runtime_at(tmp_path)
    runtime.platform_chat_id = "chat"
    monkeypatch.setattr(runtime, "has_platform_connection", lambda: True)
    release = threading.Event()
    pending = execute(runtime, lambda _: release.wait(2) and ApprovalChoice.ALLOW)
    blocked = json.loads(
        runtime._execute_tool(
            "produce_output",
            {"items": [{"kind": "text", "content": "Done"}]},
            RuntimeCallbacks(output_response=lambda _: pytest.fail("Must not publish")),
        )
    )
    assert "check_approval" in blocked["error"]
    release.set()
    prompt, _ = runtime._continuation_prompt_after_text("Done", RuntimeCallbacks(), None, 0)
    assert "Approval results" in prompt
    assert (tmp_path / "result.txt").read_text() == "done"
    assert not runtime._approval_updates()
    assert runtime.check_approval(pending["approval_id"])["status"] == "completed"


def test_root_recognizes_child_approval_and_removal_cancels_it(tmp_path):
    root = runtime_at(tmp_path)
    child = runtime_at(tmp_path / "child")
    child.cwd.mkdir(parents=True, exist_ok=True)
    release = threading.Event()
    pending = execute(child, lambda _: release.wait(2) and ApprovalChoice.ALLOW)
    root._subagents["child"] = SimpleNamespace(runtime=child, status="working", worker=None)
    assert root.approval_request_active(pending["approval_id"])
    child.abort_current_turn()
    assert not root.approval_request_active(pending["approval_id"])
    release.set()
    child._join_approval_workers()
    assert not (child.cwd / "result.txt").exists()


def test_evaluator_decision_is_separate_from_display_risk_and_fails_closed(tmp_path):
    backend = BaseBackend(runtime_at(tmp_path))
    allowed = backend._sanitize_command_evaluation(
        json.dumps(
            {
                "risk": "medium",
                "description": "Authorized bounded write.",
                "decision": "allow",
            }
        )
    )
    denied = backend._sanitize_command_evaluation(
        json.dumps(
            {
                "risk": "low",
                "description": "User disallowed this target.",
                "decision": "deny",
            }
        )
    )
    assert allowed.allows_action
    assert not denied.allows_action
    assert (
        backend._sanitize_command_evaluation('{"risk":"low","description":"Missing decision"}')
        is None
    )
    assert (
        backend._sanitize_command_evaluation(
            '{"risk":"low","description":"Invalid","decision":"maybe"}'
        )
        is None
    )
    assert CommandRiskEvaluation("low", "Legacy").allows_action


def test_new_mode_labels_keep_stored_values_and_accept_new_aliases():
    assert AgentMode.parse("Request Approvals") is AgentMode.STANDARD
    assert AgentMode.parse("auto-approve") is AgentMode.AUTOMATIC
    assert [AgentMode.STANDARD.label, AgentMode.AUTOMATIC.label, AgentMode.AUTONOMOUS.label] == [
        "Request Approvals",
        "Auto Approve",
        "Autonomous",
    ]
    assert [AgentMode.REQUEST_APPROVALS.value, AgentMode.AUTO_APPROVE.value] == [
        "standard",
        "automatic",
    ]


@pytest.mark.parametrize("mode", [AgentMode.AUTOMATIC, AgentMode.AUTONOMOUS])
@pytest.mark.parametrize(
    "evaluation",
    [
        None,
        CommandRiskEvaluation("high", "Denied", "deny"),
        CommandRiskEvaluation("medium", "Allowed", "allow"),
    ],
)
def test_unattended_modes_never_require_a_user_prompt(mode, evaluation):
    from anomx.agent.helpers.tool_manager import approval_choice_for_mode

    choice = approval_choice_for_mode(mode, evaluation)
    if mode == AgentMode.AUTONOMOUS or evaluation is not None and evaluation.allows_action:
        assert choice == ApprovalChoice.ALLOW
    else:
        assert choice == ApprovalChoice.REJECT


def test_approval_callback_failure_denies_without_execution(tmp_path):
    runtime = runtime_at(tmp_path)

    def unavailable(_):
        raise RuntimeError("Evaluator unavailable")

    pending = execute(runtime, unavailable)
    result = runtime.check_approval(pending["approval_id"], 2)
    assert not json.loads(result["result"])["approved"]
    assert not (tmp_path / "result.txt").exists()


def test_auto_approve_denial_is_attributed_to_evaluator(tmp_path):
    runtime = runtime_at(tmp_path)
    runtime.set_mode(AgentMode.AUTOMATIC)
    result = execute(runtime, lambda _: ApprovalChoice.REJECT)
    assert not result["approved"]
    assert "evaluator denied" in result["output"]
    assert "user does not allow" not in result["output"]


def test_cancelled_deferred_action_does_not_launch_after_switching_to_autonomous(
    tmp_path, monkeypatch
):
    from anomx.agent.tools.cli_command import CliCommandTool

    runtime = runtime_at(tmp_path)
    pending = execute(runtime, lambda _: ApprovalChoice.ALLOW)
    runtime.set_mode(AgentMode.AUTONOMOUS)
    original_execute = CliCommandTool.execute

    def abort_before_execute(tool, arguments, context):
        runtime.abort_current_turn()
        return original_execute(tool, arguments, context)

    monkeypatch.setattr(CliCommandTool, "execute", abort_before_execute)
    result = runtime.check_approval(pending["approval_id"], 2)
    assert not json.loads(result["result"])["approved"]
    assert not (tmp_path / "result.txt").exists()


def test_main_final_output_waits_for_child_approvals(tmp_path, monkeypatch):
    root = runtime_at(tmp_path)
    root.platform_chat_id = "chat"
    monkeypatch.setattr(root, "has_platform_connection", lambda: True)
    child = runtime_at(tmp_path / "child")
    child.cwd.mkdir(parents=True, exist_ok=True)
    release = threading.Event()
    pending = execute(child, lambda _: release.wait(2) and ApprovalChoice.ALLOW)
    root._subagents["child"] = SimpleNamespace(runtime=child, status="working", worker=None)
    blocked = json.loads(
        root._execute_tool(
            "produce_output",
            {"items": [{"kind": "text", "content": "Done"}]},
            RuntimeCallbacks(output_response=lambda _: pytest.fail("Must not publish")),
        )
    )
    assert "subagents with pending approvals" in blocked["error"]
    root.cancel_pending_approvals()
    release.set()
    assert child.check_approval(pending["approval_id"], 2)["status"] == "cancelled"
    child._join_approval_workers()
    assert not (child.cwd / "result.txt").exists()
