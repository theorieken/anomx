"""Optional questions use a nonblocking host callback, never an implicit approval."""

import json

from anomx.agent.runtime import AgentRuntime, QuestionResponse, RuntimeCallbacks
from anomx.agent.store import AnomxHome


def test_optional_question_does_not_call_blocking_callback(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    requests = []

    def optional(request):
        requests.append(request)
        return QuestionResponse(answered=False, pending=True, kind=request.kind)

    def blocking(request):
        raise AssertionError("Optional questions must never block")

    result = json.loads(
        runtime._execute_tool(
            "ask_question",
            {
                "statement": "Optional preference",
                "question": "Which label?",
                "kind": "text",
                "wait_for_response": False,
            },
            RuntimeCallbacks(question=blocking, question_async=optional),
        )
    )
    assert requests[0].wait_for_response is False
    assert result["pending"] and not result["answered"] and not result["cancelled"]


def test_unsupported_host_does_not_turn_optional_into_required(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    result = json.loads(
        runtime._execute_tool(
            "ask_question",
            {"question": "Preference?", "kind": "text", "wait_for_response": False},
            RuntimeCallbacks(question=lambda request: None),
        )
    )
    assert "does not support optional" in result["error"]


def test_host_task_prompt_replaces_general_identity_and_workflow(tmp_path):
    runtime = AgentRuntime(
        AnomxHome(tmp_path / "home"), tmp_path,
        task_instructions="Only import associated dataset data_dataset-example read-only.",
    )
    instructions = runtime._instructions()
    assert instructions.startswith("Only import associated dataset data_dataset-example")
    assert "Always check the skill catalog" not in instructions
    assert "Available Tools" in instructions
