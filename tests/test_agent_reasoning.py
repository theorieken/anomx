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
            (("data: " if sse else "") + json.dumps(event) + "\n").encode()
            for event in events
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
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "First "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "thought.\n\nStill reasoning."}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig-"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "tail"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "redacted_thinking", "data": "opaque"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": "Answer."}},
        {"type": "content_block_stop", "index": 2},
    ]
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Stream(events))
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    observed = []
    response = backend_class(runtime)._stream_response(
        "test-key", {}, lambda text: observed.append(("text", text)), None,
        lambda text: observed.append(("thought", text)),
    )
    assert observed == [("thought", "First thought.\n\nStill reasoning."), ("text", "Answer.")]
    assert response.text == "Answer."
    assert response.content[0]["signature"] == "sig-tail"
    assert response.content[1] == {"type": "redacted_thinking", "data": "opaque"}


def test_desy_coding_tagged_thought_is_forwarded_by_generate(tmp_path, monkeypatch):
    chunks = ["<thi", "nk>Check the index.", "</th", "ink>Use xs[-1]."]
    events = [{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}]
    events.extend({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": chunk}} for chunk in chunks)
    events.append({"type": "content_block_stop", "index": 0})
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Stream(events))
    home = AnomxHome(tmp_path / "home")
    home.set_api_key("desy", "test-key")
    session = home.create_session(tmp_path, provider="desy", model="coding")
    home.append_session_event(session.path, "user_message", {"message": "Fix the index."})
    runtime = AgentRuntime(home, tmp_path)
    thoughts, deltas = [], []
    result = DesyAssistantBackend(runtime).generate(
        session.path, "coding", RuntimeCallbacks(thought=thoughts.append, delta=deltas.append),
    )
    assert result == "Use xs[-1]."
    assert thoughts == ["Check the index."]
    assert "".join(deltas) == result


def test_openai_groups_summaries_per_reasoning_item_without_duplicates(tmp_path, monkeypatch):
    events = [
        {"type": "response.reasoning_summary_text.delta", "item_id": "r1", "summary_index": 0, "delta": "First"},
        {"type": "response.reasoning_summary_text.done", "item_id": "r1", "summary_index": 0, "text": "First"},
        {"type": "response.reasoning_summary_text.done", "item_id": "r1", "summary_index": 1, "text": "Second paragraph"},
        {"type": "response.output_item.done", "item": {"type": "reasoning", "id": "r1", "summary": [{"type": "summary_text", "text": "First"}, {"type": "summary_text", "text": "Second paragraph"}]}},
        {"type": "response.output_item.done", "item": {"type": "reasoning", "id": "r2", "summary": [{"type": "summary_text", "text": "Another thought"}]}},
        {"type": "response.output_text.delta", "delta": "Answer"},
        {"type": "response.completed", "response": {"id": "resp1"}},
    ]
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Stream(events))
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    observed = []
    response = OpenAIBackend(runtime)._stream_openai_response(
        "test-key", {}, lambda text: observed.append(("text", text)), None,
        lambda text: observed.append(("thought", text)),
    )
    assert response.text == "Answer"
    assert observed == [("thought", "First\n\nSecond paragraph"), ("thought", "Another thought"), ("text", "Answer")]


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
        {}, RuntimeCallbacks(delta=lambda text: observed.append(("text", text)), thought=lambda text: observed.append(("thought", text))),
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
        assert parser.drain_completed_thoughts() == ("I should inspect this.\n\nThis is still reasoning.",)
    parser = ThinkingTagStreamFilter()
    assert parser.feed("<think>I should inspect this.\n\nStill reasoning.</thi")[0] == ""
    assert parser.finish() == ""
    assert parser.drain_completed_thoughts() == ("I should inspect this.\n\nStill reasoning.</thi",)
