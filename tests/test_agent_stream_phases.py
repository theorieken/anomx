"""Streaming phase events follow provider events without token-by-token flooding."""

import json
from types import SimpleNamespace

import pytest

from anomx.agent import AnomxHome
from anomx.agent.backends import backend_for_provider
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks


class Stream:
    def __init__(self, events, *, sse=True):
        self.lines = [
            (("data: " if sse else "") + json.dumps(event) + "\n").encode()
            for event in events
        ]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self.lines)


def stream_response(backend, provider, observed):
    def status(value):
        observed.append(("status", value))

    def thought(value):
        observed.append(("thought", value))

    def delta(value):
        observed.append(("text", value))
    if provider == "openai":
        return backend._stream_openai_response("key", {}, delta, status, thought)
    if provider == "anthropic":
        return backend._stream_response("key", {}, delta, status, thought)
    if provider == "ollama":
        return backend._stream_ollama_response(
            "model", [], RuntimeCallbacks(status=status, thought=thought, delta=delta),
        )
    return backend._stream_chat_completion("key", {}, delta, status, thought)


def native_events(provider):
    if provider == "openai":
        return [
            {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "r1"}},
            *[{"type": "response.reasoning_summary_text.delta", "item_id": "r1", "delta": part}
              for part in ("Inspect ", "the file.")],
            {"type": "response.output_item.done", "item": {"type": "reasoning", "id": "r1"}},
            {"type": "response.output_item.added", "item": {"type": "function_call"}},
            *[{"type": "response.function_call_arguments.delta", "delta": part}
              for part in ('{"command":', '"ls"}')],
        ]
    if provider == "anthropic":
        return [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
            *[{"type": "content_block_delta", "index": 0,
               "delta": {"type": "thinking_delta", "thinking": part}}
              for part in ("Inspect ", "the file.")],
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1,
             "content_block": {"type": "tool_use", "id": "c1", "name": "run_command"}},
            *[{"type": "content_block_delta", "index": 1,
               "delta": {"type": "input_json_delta", "partial_json": part}}
              for part in ('{"command":', '"ls"}')],
            {"type": "content_block_stop", "index": 1},
        ]
    if provider == "ollama":
        return [
            *[{"message": {"thinking": part}} for part in ("Inspect ", "the file.")],
            {"message": {"tool_calls": [{"function": {
                "name": "run_command", "arguments": {"command": "ls"},
            }}]}},
        ]
    return [
        *[{"choices": [{"delta": {"reasoning_content": part}}]}
          for part in ("Inspect ", "the file.")],
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1",
          "function": {"name": "run_command", "arguments": '{"command":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0,
          "function": {"arguments": '"ls"}'}}]}}]},
    ]


@pytest.mark.parametrize("provider", ["openai", "anthropic", "blablador", "ollama"])
def test_tool_argument_generation_stays_thinking_and_deduplicates_tokens(tmp_path, monkeypatch, provider):
    events = native_events(provider)
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: Stream(events, sse=provider != "ollama"),
    )
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    observed = []
    stream_response(backend_for_provider(provider, runtime), provider, observed)
    assert [item for item in observed if item != ("status", "Loading model")] == [
        ("status", "Thinking"),
        ("thought", "Inspect the file."),
        ("status", "Thinking"),
        ("status", "Executing"),
    ]


@pytest.mark.parametrize("provider", ["openai", "anthropic", "blablador", "ollama"])
def test_tagged_thought_and_prose_phases_restart_per_request(tmp_path, monkeypatch, provider):
    chunks = ["<thi", "nk>Inspect", " the file.", "</think>Answer", " continues."]
    if provider == "openai":
        events = [{"type": "response.output_text.delta", "delta": part} for part in chunks]
    elif provider == "anthropic":
        events = [{"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}]
        events += [{"type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": part}} for part in chunks]
    elif provider == "ollama":
        events = [{"message": {"content": part}} for part in chunks]
    else:
        events = [{"choices": [{"delta": {"content": part}}]} for part in chunks]
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: Stream(events, sse=provider != "ollama"),
    )
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    backend = backend_for_provider(provider, runtime)
    for _ in range(2):
        observed = []
        response = stream_response(backend, provider, observed)
        assert response.text == "Answer continues."
        phases = [item for item in observed
                  if item[0] != "text" and item != ("status", "Loading model")]
        assert phases == [
            ("status", "Thinking"), ("thought", "Inspect the file."), ("status", "Thinking"),
            ("status", "Executing"),
        ]
        assert ("status", "Writing") not in observed
        assert observed.index(("status", "Thinking")) < observed.index(("text", "Answer"))


def test_next_reasoning_block_emits_thinking_after_completed_thought(tmp_path, monkeypatch):
    events = []
    for index in range(2):
        events.extend([
            {"type": "content_block_start", "index": index, "content_block": {"type": "thinking"}},
            {"type": "content_block_delta", "index": index,
             "delta": {"type": "thinking_delta", "thinking": f"Thought {index}"}},
            {"type": "content_block_stop", "index": index},
        ])
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Stream(events))
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    observed = []
    stream_response(backend_for_provider("anthropic", runtime), "anthropic", observed)
    assert observed == [
        ("status", "Thinking"), ("thought", "Thought 0"),
        ("status", "Thinking"), ("thought", "Thought 1"),
        ("status", "Executing"),
    ]


def test_retry_resets_phase_even_when_previous_attempt_was_thinking(tmp_path, monkeypatch):
    calls = 0

    def open_stream(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("transient")
        return Stream([{"type": "response.output_text.delta", "delta": "Answer"}])

    monkeypatch.setattr("urllib.request.urlopen", open_stream)
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    backend = backend_for_provider("openai", runtime)
    monkeypatch.setattr(backend, "_sleep_before_model_request_retry", lambda *args: True)
    observed = []
    assert stream_response(backend, "openai", observed).text == "Answer"
    assert [item for item in observed if item[0] == "status"] == [
        ("status", "Thinking"), ("status", "Executing"),
        ("status", "Thinking"), ("status", "Executing"),
    ]


@pytest.mark.parametrize(("provider", "method"), [
    ("openai", "_execute_requested_tools"),
    ("anthropic", "_execute_anthropic_requested_tools"),
    ("ollama", "_execute_ollama_requested_tools"),
    ("blablador", "_execute_chat_completion_tools"),
])
def test_tool_batches_leave_thinking_before_execution(tmp_path, monkeypatch, provider, method):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    backend = backend_for_provider(provider, runtime)
    observed = []

    def execute_tool(*args, **kwargs):
        observed.append(("tool", args[0]))
        return "done"

    monkeypatch.setattr(runtime, "_execute_tool", execute_tool)
    callbacks = RuntimeCallbacks(status=lambda value: observed.append(("status", value)))
    execute = getattr(backend, method)
    assert execute(SimpleNamespace(tool_calls=()), callbacks, tmp_path / "session") == []
    assert not observed
    call = SimpleNamespace(
        name="run_command", call_id="c1", tool_use_id="c1", input={"command": "ls"},
        arguments={"command": "ls"} if provider == "ollama" else '{"command":"ls"}',
    )
    execute(SimpleNamespace(tool_calls=(call, call)), callbacks, tmp_path / "session")
    assert observed == [
        ("status", "Executing"), ("tool", "run_command"), ("tool", "run_command"),
    ]


@pytest.mark.parametrize("provider", ["openai", "anthropic", "blablador", "ollama"])
@pytest.mark.parametrize("outcome", ["aborted", "failed"])
def test_request_phase_clears_when_stream_does_not_complete(tmp_path, monkeypatch, provider, outcome):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    aborted = False
    monkeypatch.setattr(runtime, "_turn_aborted", lambda: aborted)

    class InterruptedStream(Stream):
        def __iter__(self):
            nonlocal aborted
            if outcome == "failed":
                raise ValueError("malformed stream")
            aborted = True
            yield from super().__iter__()

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: InterruptedStream(
        native_events(provider), sse=provider != "ollama",
    ))
    observed = []
    if outcome == "failed":
        with pytest.raises(ValueError, match="malformed stream"):
            stream_response(backend_for_provider(provider, runtime), provider, observed)
    else:
        stream_response(backend_for_provider(provider, runtime), provider, observed)
    assert observed[-1] == ("status", "Executing")
    assert ("status", "Thinking") in observed
