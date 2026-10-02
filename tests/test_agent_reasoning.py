"""Provider reasoning stays in thought events, never assistant text."""

import json

import pytest

from anomx.agent import AnomxHome
from anomx.agent.backends.anthropic import AnthropicBackend
from anomx.agent.backends.desy_assistant import DesyAssistantBackend
from anomx.agent.backends.ollama import OllamaBackend
from anomx.agent.backends.openai import OpenAIBackend
from anomx.agent.base.backends import ThinkingTagStreamFilter
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks


class Stream:
    def __init__(self, events, *, sse=True):
        self.lines = [
            (("data: " if sse else "") + json.dumps(event) + "\n").encode() for event in events
        ]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self.lines)


@pytest.mark.parametrize("backend_class", [AnthropicBackend, DesyAssistantBackend])
def test_messages_thought_blocks_preserve_replay_and_order(tmp_path, monkeypatch, backend_class):
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "First "},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "thought.\n\nStill reasoning."},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig-"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "tail"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "redacted_thinking", "data": "opaque"},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "text", "text": "Answer."},
        },
        {"type": "content_block_stop", "index": 2},
    ]
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Stream(events))
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    observed = []
    response = backend_class(runtime)._stream_response(
        "test-key",
        {},
        lambda text: observed.append(("text", text)),
        None,
        lambda text: observed.append(("thought", text)),
    )
    assert observed == [("thought", "First thought.\n\nStill reasoning."), ("text", "Answer.")]
    assert response.text == "Answer."
    assert response.content[0]["signature"] == "sig-tail"
    assert response.content[1] == {"type": "redacted_thinking", "data": "opaque"}


def test_desy_coding_tagged_thought_is_forwarded_by_generate(tmp_path, monkeypatch):
    chunks = ["<thi", "nk>Check the index.", "</th", "ink>Use xs[-1]."]
    events = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
    ]
    events.extend(
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": chunk}}
        for chunk in chunks
    )
    events.append({"type": "content_block_stop", "index": 0})
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Stream(events))
    home = AnomxHome(tmp_path / "home")
    home.set_api_key("desy", "test-key")
    session = home.create_session(tmp_path, provider="desy", model="coding")
    home.append_session_event(session.path, "user_message", {"message": "Fix the index."})
    runtime = AgentRuntime(home, tmp_path)
    thoughts, deltas = [], []
    result = DesyAssistantBackend(runtime).generate(
        session.path,
        "coding",
        RuntimeCallbacks(thought=thoughts.append, delta=deltas.append),
    )
    assert result == "Use xs[-1]."
    assert thoughts == ["Check the index."]
    assert "".join(deltas) == result


def test_openai_groups_summaries_per_reasoning_item_without_duplicates(tmp_path, monkeypatch):
    events = [
        {
            "type": "response.reasoning_summary_text.delta",
            "item_id": "r1",
            "summary_index": 0,
            "delta": "First",
        },
        {
            "type": "response.reasoning_summary_text.done",
            "item_id": "r1",
            "summary_index": 0,
            "text": "First",
        },
        {
            "type": "response.reasoning_summary_text.done",
            "item_id": "r1",
            "summary_index": 1,
            "text": "Second paragraph",
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "reasoning",
                "id": "r1",
                "summary": [
                    {"type": "summary_text", "text": "First"},
                    {"type": "summary_text", "text": "Second paragraph"},
                ],
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "reasoning",
                "id": "r2",
                "summary": [{"type": "summary_text", "text": "Another thought"}],
            },
        },
        {"type": "response.output_text.delta", "delta": "Answer"},
        {"type": "response.completed", "response": {"id": "resp1"}},
    ]
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Stream(events))
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    observed = []
    response = OpenAIBackend(runtime)._stream_openai_response(
        "test-key",
        {},
        lambda text: observed.append(("text", text)),
        None,
        lambda text: observed.append(("thought", text)),
    )
    assert response.text == "Answer"
    assert observed == [
        ("thought", "First\n\nSecond paragraph"),
        ("thought", "Another thought"),
        ("text", "Answer"),
    ]


def test_ollama_thought_arrives_before_text(tmp_path, monkeypatch):
    events = [
        {"message": {"thinking": "Check ", "content": ""}},
        {"message": {"thinking": "the result.", "content": ""}},
        {"message": {"content": "Answer"}, "done": True},
    ]
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Stream(events, sse=False))
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    observed = []
    response = OllamaBackend(runtime)._stream_ollama_response(
        "qwen3",
        [],
        RuntimeCallbacks(
            delta=lambda text: observed.append(("text", text)),
            thought=lambda text: observed.append(("thought", text)),
        ),
    )
    assert response.text == "Answer"
    assert response.message["thinking"] == "Check the result."
    assert observed == [("thought", "Check the result."), ("text", "Answer")]


def test_thinking_tags_are_safe_at_every_chunk_boundary():
    text = "<think>I should inspect this.\n\nThis is still reasoning.</think>Answer"
    for split in range(len(text) + 1):
        parser = ThinkingTagStreamFilter()
        first, _ = parser.feed(text[:split])
        second, _ = parser.feed(text[split:])
        assert first + second + parser.finish() == "Answer"
        assert parser.drain_completed_thoughts() == (
            "I should inspect this.\n\nThis is still reasoning.",
        )
    parser = ThinkingTagStreamFilter()
    assert parser.feed("<think>I should inspect this.\n\nStill reasoning.</thi")[0] == ""
    assert parser.finish() == ""
    assert parser.drain_completed_thoughts() == ("I should inspect this.\n\nStill reasoning.</thi",)


def test_subagent_thought_is_saved_in_work_history(tmp_path, monkeypatch):
    from anomx.agent.base.agents import AgentKind
    from anomx.agent.base.subagents import SubagentRuntimeState

    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="desy", model="coding")
    runtime = AgentRuntime(home, tmp_path)
    child = AgentRuntime(home, tmp_path)
    state = SubagentRuntimeState(
        agent_id="child",
        kind=AgentKind.SUB,
        name="Child",
        prompt="Inspect",
        status="running",
        statement="",
        started_at="",
        runtime=child,
    )

    def respond(session_path, callbacks, **kwargs):
        callbacks.thought("Inspect the index.\n\nCheck the bounds.")
        return "Done"

    monkeypatch.setattr(child, "backend_response", respond)
    runtime._run_subagent_turn(state, "Inspect", session.path, RuntimeCallbacks())
    assert state.status == "ready", state.error
    thoughts = [
        event["payload"]
        for event in home.read_session_events(state.session_path)
        if event["payload"].get("type") == "work_message"
        and event["payload"].get("role") == "thought"
    ]
    assert thoughts[0]["command"] == "Inspect the index.\n\nCheck the bounds."
    assert state.command_history[0]["kind"] == "thought"
    assert state.command_history[0]["thought"] == thoughts[0]["command"]


@pytest.mark.parametrize("opening,closing", [
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<reason>", "</reason>"),
    ("<reasoning>", "</reasoning>"),
    ("<thought>", "</thought>"),
    ("<|begin_of_thought|>", "<|end_of_thought|>"),
    ('<details type="reasoning" open>', "</details>"),
])
def test_explicit_reasoning_markers_at_every_chunk_boundary(opening, closing):
    text = f"{opening}Private deliberation.{closing}Answer."
    for split in range(len(text) + 1):
        parser = ThinkingTagStreamFilter()
        left, _ = parser.feed(text[:split])
        right, _ = parser.feed(text[split:])
        assert left + right + parser.finish() == "Answer."
        assert parser.drain_completed_thoughts() == ("Private deliberation.",)
    parser = ThinkingTagStreamFilter()
    assert "".join(parser.feed(character)[0] for character in text) + parser.finish() == "Answer."
    assert parser.drain_completed_thoughts() == ("Private deliberation.",)


def test_ordinary_details_and_unmarked_prose_are_not_classified_as_thoughts():
    text = "I have enough context. <details><summary>Evidence</summary>Useful data</details>"
    parser = ThinkingTagStreamFilter()
    assert "".join(parser.feed(character)[0] for character in text) + parser.finish() == text
    assert parser.drain_completed_thoughts() == ()


@pytest.mark.parametrize("provider", ["openai", "anthropic", "desy", "ollama", "blablador", "kimi"])
def test_every_backend_routes_tagged_reasoning_to_thought_callback(tmp_path, monkeypatch, provider):
    from anomx.agent.backends import backend_for_provider

    chunks = ['<details type="rea', 'soning"><summary>Thinking</summary>',
              'Inspect the result.', '</det', 'ails>Answer.']
    observed = []
    delta = lambda value: observed.append(("text", value))
    thought = lambda value: observed.append(("thought", value))
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    backend = backend_for_provider(provider, runtime)
    if provider in {"anthropic", "desy"}:
        events = [{"type": "content_block_start", "index": 0,
                   "content_block": {"type": "text", "text": ""}}]
        events.extend({"type": "content_block_delta", "index": 0,
                       "delta": {"type": "text_delta", "text": chunk}} for chunk in chunks)
        events.append({"type": "content_block_stop", "index": 0})
    elif provider == "openai":
        events = [{"type": "response.output_text.delta", "delta": chunk} for chunk in chunks]
    elif provider == "ollama":
        events = [{"message": {"content": chunk}} for chunk in chunks]
    else:
        events = [{"choices": [{"delta": {"content": chunk}}]} for chunk in chunks]
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Stream(events, sse=provider != "ollama"))
    if provider in {"anthropic", "desy"}:
        response = backend._stream_response("key", {}, delta, None, thought)
    elif provider == "openai":
        response = backend._stream_openai_response("key", {}, delta, None, thought)
    elif provider == "ollama":
        response = backend._stream_ollama_response("model", [], RuntimeCallbacks(delta=delta, thought=thought))
    else:
        response = backend._stream_chat_completion("key", {}, delta, None, thought)
    assert response.text == "Answer."
    assert observed == [("thought", "Inspect the result."), ("text", "Answer.")]


@pytest.mark.parametrize("provider", ["blablador", "kimi"])
@pytest.mark.parametrize("field", ["reasoning_content", "reasoning", "thinking"])
def test_chat_backend_native_reasoning_is_separate(tmp_path, monkeypatch, provider, field):
    from anomx.agent.backends import backend_for_provider

    events = [{"choices": [{"delta": {field: "Inspect."}}]},
              {"choices": [{"delta": {"content": "Answer."}}]}]
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Stream(events))
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    observed = []
    response = backend_for_provider(provider, runtime)._stream_chat_completion(
        "key", {}, lambda value: observed.append(("text", value)), None,
        lambda value: observed.append(("thought", value)),
    )
    assert response.text == "Answer."
    assert observed == [("thought", "Inspect."), ("text", "Answer.")]
