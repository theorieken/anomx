"""Deferred approval reviews; only the foreground runtime executes approved actions."""

from __future__ import annotations

import copy
import json
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from anomx.agent.base.tools import BaseTool, ToolExecutionContext
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.helpers.tool_manager import ApprovalChoice, CommandApprovalRequest

if TYPE_CHECKING:
    from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks


class ApprovalDeferred(Exception):
    """Unwind a tool before its side effect while an approval decision is pending."""

    def __init__(self, approval_id: str) -> None:
        self.approval_id = approval_id


@dataclass
class PendingApproval:
    """An exact deferred action and the separately computed approval decision."""

    request: CommandApprovalRequest
    tool: BaseTool
    arguments: dict[str, Any]
    callbacks: RuntimeCallbacks
    session_path: Path | None
    current_dir: Path
    decision: ApprovalChoice | None = None
    result: str | None = None
    cancelled: bool = False
    consumed: bool = False
    ready: threading.Event = field(default_factory=threading.Event)
    worker: threading.Thread | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "approval_id": self.request.request_id,
            "status": "cancelled"
            if self.cancelled
            else ("completed" if self.consumed else "ready" if self.ready.is_set() else "pending"),
            "command": self.request.command,
            "statement": self.request.statement,
        }


class PendingApprovalsMixin:
    """Shared CLI/cloud lifecycle for asynchronous approvals in Request Approvals mode."""

    @property
    def _approval_runtime(self) -> AgentRuntime:
        """The concrete runtime supplies execution, cancellation and child-agent state."""
        return cast("AgentRuntime", self)

    def _init_pending_approvals(self) -> None:
        self._pending_approvals: dict[str, PendingApproval] = {}
        self._approval_lock = threading.RLock()

    def approval_request_active(self, request_id: str) -> bool:
        """Tell adapters whether a review is still active, including child-agent reviews."""
        if self._approval_runtime._turn_aborted():
            return False
        with self._approval_lock:
            pending = self._pending_approvals.get(request_id)
            if pending is not None:
                return not pending.cancelled and not pending.consumed
        return any(
            state.runtime is not None and state.runtime.approval_request_active(request_id)
            for state in self._approval_runtime._subagent_states()
        )

    def _approval_updates(self) -> list[dict[str, Any]]:
        with self._approval_lock:
            return [
                state.payload()
                for state in self._pending_approvals.values()
                if not state.consumed and not state.cancelled
            ]

    def _has_pending_approvals(self) -> bool:
        if self._approval_updates():
            return True
        return any(
            state.runtime is not None and state.runtime._has_pending_approvals()
            for state in self._approval_runtime._subagent_states()
        )

    def cancel_pending_approvals(self) -> None:
        """Invalidate deferred actions before cancelling or replacing a turn."""
        with self._approval_lock:
            for state in self._pending_approvals.values():
                if state.result is None:
                    state.cancelled = True
                    state.ready.set()
        for child in self._approval_runtime._subagent_states():
            if child.runtime is not None:
                child.runtime.cancel_pending_approvals()

    def _join_approval_workers(self) -> None:
        self.cancel_pending_approvals()
        with self._approval_lock:
            workers = [state.worker for state in self._pending_approvals.values() if state.worker]
        deadline = time.monotonic() + 5.0
        for worker in workers:
            if worker is not threading.current_thread():
                worker.join(timeout=max(0.0, deadline - time.monotonic()))

    def _execute_with_approval(
        self,
        tool: BaseTool,
        arguments: dict[str, Any],
        callbacks: RuntimeCallbacks,
        session_path: Path | None,
    ) -> str:
        callback = callbacks.approval
        supports_approval = tool.name in {"run_command", "start_process", "mount_folder"}
        wait = arguments.get("wait_for_approval") is True
        if callback is not None and supports_approval:

            def review(request: CommandApprovalRequest) -> ApprovalChoice:
                request_id = uuid4().hex
                state = PendingApproval(
                    request=replace(request, request_id=request_id, wait_for_approval=wait),
                    tool=tool,
                    arguments=copy.deepcopy(arguments),
                    callbacks=callbacks,
                    session_path=session_path,
                    current_dir=self._approval_runtime.tool_manager.current_dir,
                )
                with self._approval_lock:
                    self._pending_approvals[request_id] = state
                if self._approval_runtime.tool_manager.mode != AgentMode.STANDARD or wait:
                    try:
                        decision = callback(state.request)
                        return (
                            ApprovalChoice.REJECT
                            if self._approval_runtime._turn_aborted() or state.cancelled
                            else decision
                        )
                    finally:
                        with self._approval_lock:
                            state.consumed = True
                            state.ready.set()

                def decide() -> None:
                    try:
                        decision = callback(state.request)
                    except Exception:
                        decision = ApprovalChoice.REJECT
                    with self._approval_lock:
                        if not state.cancelled:
                            state.decision = decision
                        state.ready.set()

                state.worker = threading.Thread(
                    target=decide,
                    name=f"anomx-approval-{request_id[:8]}",
                    daemon=True,
                )
                state.worker.start()
                raise ApprovalDeferred(request_id)

            callbacks = replace(callbacks, approval=review)
        try:
            return tool.execute(arguments, ToolExecutionContext(self, callbacks, session_path))
        except ApprovalDeferred as pending:
            return json.dumps(
                {
                    "approved": False,
                    "approval_pending": True,
                    "approval_id": pending.approval_id,
                    "status": "pending",
                    "output": "Approval review is pending; the action has not executed. "
                    "Continue independent work. Call check_approval with this approval_id "
                    "to collect the decision and execute the action if approved. "
                    "Do not resubmit the original action.",
                }
            )

    def check_approval(self, approval_id: str, wait_seconds: float = 0) -> dict[str, Any]:
        """Collect a decision and execute its exact action at most once, in the foreground."""
        with self._approval_lock:
            state = self._pending_approvals.get(approval_id)
        if state is None:
            return {"error": "Unknown approval_id.", "approval_id": approval_id}
        deadline = time.monotonic() + min(60.0, max(0.0, wait_seconds))
        while not state.ready.is_set() and not self._approval_runtime._turn_aborted():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            state.ready.wait(min(0.1, remaining))
        with self._approval_lock:
            if self._approval_runtime._turn_aborted():
                state.cancelled = True
            if state.cancelled:
                return {**state.payload(), "approved": False, "output": "Action cancelled."}
            if state.consumed:
                return {**state.payload(), "result": state.result}
            if not state.ready.is_set():
                return {**state.payload(), "approval_pending": True, "approved": False}
            # Consume before execution: repeated checks must never replay a write.
            state.consumed = True
        decision = state.decision
        if decision not in {ApprovalChoice.ALLOW, ApprovalChoice.ALWAYS_ALLOW}:
            rejected = self._approval_runtime.session_rejected_commands
            if decision == ApprovalChoice.ALWAYS_REJECT and rejected is not None:
                rejected.add(state.request.allowance_key or state.request.canonical_command)
            state.result = json.dumps(
                {
                    "approved": False,
                    "output": "Approval was denied; the action did not execute.",
                }
            )
        else:

            def approved_once(request: CommandApprovalRequest) -> ApprovalChoice:
                if (
                    self._approval_runtime._turn_aborted()
                    or state.cancelled
                    or request.command != state.request.command
                    or request.canonical_command != state.request.canonical_command
                ):
                    return ApprovalChoice.REJECT
                return decision

            manager = self._approval_runtime.tool_manager
            current_dir = manager.current_dir
            sandbox = self._approval_runtime.local_sandbox_session
            sandbox_dir = sandbox.current_dir if sandbox is not None else None
            manager.current_dir = state.current_dir
            if sandbox is not None:
                sandbox.current_dir = state.current_dir
            try:
                if self._approval_runtime._turn_aborted() or state.cancelled:
                    state.cancelled = True
                    state.result = json.dumps({"approved": False, "output": "Action cancelled."})
                    return {**state.payload(), "result": state.result}
                state.result = state.tool.execute(
                    state.arguments,
                    ToolExecutionContext(
                        self,
                        replace(state.callbacks, approval=approved_once),
                        state.session_path,
                    ),
                )
            except Exception as error:
                state.result = json.dumps({"approved": False, "error": str(error)})
            finally:
                manager.current_dir = current_dir
                if sandbox is not None and sandbox_dir is not None:
                    sandbox.current_dir = sandbox_dir
        if state.session_path is not None:
            self._approval_runtime.home.append_session_event(
                state.session_path,
                "approval_execution",
                {
                    **state.payload(),
                    "result": state.result,
                },
            )
        return {**state.payload(), "result": state.result}

    def _approval_continuation(self, callbacks: RuntimeCallbacks) -> str | None:
        updates = self._approval_updates()
        if not updates:
            return None
        if callbacks.status is not None:
            callbacks.status("Waiting:60.0")
        # A text final cannot abandon approvals. Wait once, then collect ready decisions.
        first = self.check_approval(str(updates[0]["approval_id"]), wait_seconds=60)
        results = [first]
        for item in updates[1:]:
            results.append(self.check_approval(str(item["approval_id"])))
        return (
            "Approval results (pending actions have not executed):\n"
            + json.dumps(results, ensure_ascii=False)
            + "\nContinue independent work or use check_approval. Integrate completed results "
            "before finishing; do not resubmit the original actions."
        )
