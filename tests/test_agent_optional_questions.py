"""Optional questions use a nonblocking host callback, never an implicit approval."""

import json
from queue import Queue

from anomx.agent.base.interactions import QuestionRequest
from anomx.agent.runtime import AgentRuntime, QuestionResponse, RuntimeCallbacks
from anomx.agent.runtime_process import RuntimeProcessClient
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
                "title": "Label decision",
                "body": "Use **short labels** for this dataset.",
                "object_references": ["data_dataset-example"],
            },
            RuntimeCallbacks(question=blocking, question_async=optional),
        )
    )
    assert requests[0].wait_for_response is False
    assert requests[0].title == "Label decision"
    assert requests[0].body == "Use **short labels** for this dataset."
    assert requests[0].object_references == ("data_dataset-example",)
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


def test_invalid_attachments_do_not_reach_host(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)

    def unexpected(request):
        raise AssertionError("Invalid requests must not reach the host")

    result = json.loads(runtime._execute_tool(
        "ask_question",
        {"question": "Review?", "kind": "text", "object_references": "data_dataset-example"},
        RuntimeCallbacks(question=unexpected),
    ))
    assert "object_references" in result["error"]


def test_optional_question_ipc_preserves_pending_response(tmp_path, monkeypatch):
    client = RuntimeProcessClient(AnomxHome(tmp_path / "home"), tmp_path)
    replies = Queue()
    requests = []

    def optional(request):
        requests.append(request)
        return QuestionResponse(answered=False, pending=True, kind=request.kind)

    monkeypatch.setattr(client, "_send", replies.put)
    client._callbacks["turn"] = client._callback_map(RuntimeCallbacks(question_async=optional))
    request = QuestionRequest(
        question="Which dataset?", kind="text", wait_for_response=False,
        body="Review **this** dataset.", object_references=("data_dataset-example",),
    )
    client._dispatch_event({
        "request_id": "turn", "callback": "question_async",
        "interaction_id": "question", "args": [request],
    })
    reply = replies.get(timeout=2)
    assert requests == [request]
    assert reply["interaction_id"] == "question"
    assert reply["value"].pending and not reply["value"].cancelled


def test_host_task_prompt_replaces_general_identity_and_workflow(tmp_path):
    runtime = AgentRuntime(
        AnomxHome(tmp_path / "home"), tmp_path,
        task_instructions="Only import associated dataset data_dataset-example read-only.",
    )
    instructions = runtime._instructions()
    assert instructions.startswith("Only import associated dataset data_dataset-example")
    assert "Always check the skill catalog" not in instructions
    assert "Available Tools" in instructions
