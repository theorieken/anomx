"""Wait tool."""

from __future__ import annotations

import time
from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema

SUBAGENT_WAIT_SECONDS = 60.0 * 5


class WaitTool(BaseTool):
    def __init__(self, *, target_description: str) -> None:
        super().__init__(
            name="wait",
            description=f"Wait up to 60 seconds for {target_description}.",
            parameters=object_schema({}, []),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        del arguments
        seconds = SUBAGENT_WAIT_SECONDS
        started_at = time.monotonic()
        if not context.runtime._has_running_wait_targets():
            return context.json_result(
                {
                    "waited_seconds": 0.0,
                    "commands": [
                        context.runtime._command_state_payload(command)
                        for command in context.runtime._command_states()
                    ],
                    "subagents": [
                        context.runtime._subagent_runtime_payload(subagent)
                        for subagent in context.runtime._subagent_states()
                    ],
                }
            )
        if context.callbacks.status is not None:
            context.callbacks.status(f"Waiting:{seconds}")
        deadline = started_at + seconds
        while time.monotonic() < deadline:
            if context.runtime._turn_aborted() or not context.runtime._has_running_wait_targets():
                break
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        waited_seconds = min(seconds, max(0.0, time.monotonic() - started_at))
        return context.json_result(
            {
                "waited_seconds": waited_seconds,
                "commands": [
                    context.runtime._command_state_payload(command)
                    for command in context.runtime._command_states()
                ],
                "subagents": [
                    context.runtime._subagent_runtime_payload(subagent)
                    for subagent in context.runtime._subagent_states()
                ],
            }
        )
