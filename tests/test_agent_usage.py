import json

from anomx.agent import AnomxHome
from anomx.agent.backends.anthropic import AnthropicBackend
from anomx.agent.backends.kimi import KimiBackend
from anomx.agent.backends.ollama import OllamaBackend
from anomx.agent.backends.openai import OpenAIBackend
from anomx.agent.base.backends import (
    TokenUsage,
    UsageSnapshot,
    anthropic_token_usage,
    chat_completion_token_usage,
    format_token_count,
    ollama_token_usage,
    openai_token_usage,
)
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks


class FakeStreamResponse:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self.lines)


def test_format_token_count_uses_compact_suffixes():
    assert format_token_count(0) == "0"
    assert format_token_count(132) == "132"
    assert format_token_count(999) == "999"
    assert format_token_count(1_000) == "1k"
    assert format_token_count(1_500) == "1.5k"
    assert format_token_count(12_000) == "12k"
    assert format_token_count(324_000) == "324k"
    assert format_token_count(999_499) == "999k"
    assert format_token_count(999_999) == "1M"
    assert format_token_count(1_000_000) == "1M"
    assert format_token_count(1_200_000) == "1.2M"


def test_usage_parsers_normalize_provider_payloads():
    assert anthropic_token_usage(
        {
            "input_tokens": 2_100,
            "cache_creation_input_tokens": 3_000,
            "cache_read_input_tokens": 15_000,
            "output_tokens": 503,
        }
    ) == TokenUsage(
        input_tokens=20_100,
        output_tokens=503,
        cached_tokens=15_000,
        cache_creation_tokens=3_000,
        total_tokens=20_603,
    )
    assert openai_token_usage(
        {
            "input_tokens": 328,
            "input_tokens_details": {"cached_tokens": 128},
            "output_tokens": 52,
            "total_tokens": 380,
        }
    ) == TokenUsage(
        input_tokens=328,
        output_tokens=52,
        cached_tokens=128,
        cache_creation_tokens=0,
        total_tokens=380,
    )
    assert chat_completion_token_usage(
        {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
    ) == TokenUsage(
        input_tokens=10,
        output_tokens=20,
        cached_tokens=4,
        cache_creation_tokens=0,
        total_tokens=30,
    )
    assert ollama_token_usage({"done": True, "prompt_eval_count": 26, "eval_count": 298}) == (
        TokenUsage(
            input_tokens=26,
            output_tokens=298,
            cached_tokens=0,
            cache_creation_tokens=0,
            total_tokens=324,
        )
    )
    assert anthropic_token_usage(None) is None
    assert openai_token_usage({}) is None
    assert chat_completion_token_usage({"prompt_tokens": 0}) is None
    assert ollama_token_usage({}) is None


def test_usage_snapshot_serialization_round_trip():
    snapshot = UsageSnapshot(
        total=TokenUsage(
            input_tokens=20_100,
            output_tokens=503,
            cached_tokens=15_000,
            cache_creation_tokens=3_000,
            total_tokens=20_603,
        ),
        context_tokens=20_100,
    )
    assert UsageSnapshot.from_dict(snapshot.to_dict()) == snapshot
    assert UsageSnapshot.from_dict(None) is None
    assert UsageSnapshot.from_dict({"context_tokens": 10}) is None


def test_chat_completion_stream_reports_usage(tmp_path, monkeypatch):
    import anomx.agent.backends.openai_chat as openai_chat_module

    responses = iter(
        (
            [
                b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n',
                (
                    b'data: {"choices":[],"usage":{"prompt_tokens":12000,"completion_tokens":5,'
                    b'"total_tokens":12005,"prompt_tokens_details":{"cached_tokens":4000}}}\n'
                ),
                b"data: [DONE]\n",
            ],
        )
    )
    home = AnomxHome(tmp_path / "home")
    home.set_api_key("kimi", "kimi-key")
    session = home.create_session(tmp_path, provider="kimi", model="kimi-k3")
    home.append_session_event(session.path, "user_message", {"message": "Hello."})
    runtime = AgentRuntime(home, tmp_path)
    payloads = []

    def fake_urlopen(request, timeout):
        payloads.append(json.loads(request.data.decode("utf-8")))
        return FakeStreamResponse(next(responses))

    monkeypatch.setattr(openai_chat_module.urllib.request, "urlopen", fake_urlopen)

    snapshots = []
    response = KimiBackend(runtime).generate(
        session.path,
        "kimi-k3",
        RuntimeCallbacks(usage=snapshots.append),
    )

    assert response == "Hi"
    assert payloads[0]["stream_options"] == {"include_usage": True}
    assert snapshots == [
        UsageSnapshot(
            total=TokenUsage(
                input_tokens=12_000,
                output_tokens=5,
                cached_tokens=4_000,
                cache_creation_tokens=0,
                total_tokens=12_005,
            ),
            context_tokens=12_000,
        )
    ]


def test_anthropic_stream_reports_merged_usage(tmp_path, monkeypatch):
    import anomx.agent.backends.anthropic as anthropic_module

    responses = iter(
        (
            [
                (
                    b'data: {"type":"message_start","message":{"id":"msg_1","usage":'
                    b'{"input_tokens":100,"cache_creation_input_tokens":40,'
                    b'"cache_read_input_tokens":60,"output_tokens":1}}}\n'
                ),
                (
                    b'data: {"type":"content_block_delta","index":0,'
                    b'"delta":{"type":"text_delta","text":"Done."}}\n'
                ),
                (
                    b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
                    b'"usage":{"output_tokens":12}}\n'
                ),
                b'data: {"type":"message_stop"}\n',
            ],
        )
    )
    home = AnomxHome(tmp_path / "home")
    home.set_api_key("anthropic", "anthropic-key")
    session = home.create_session(tmp_path, provider="anthropic", model="claude-sonnet-4-6")
    home.append_session_event(session.path, "user_message", {"message": "Hello."})
    runtime = AgentRuntime(home, tmp_path)

    def fake_urlopen(request, timeout):
        return FakeStreamResponse(next(responses))

    monkeypatch.setattr(anthropic_module.urllib.request, "urlopen", fake_urlopen)

    snapshots = []
    response = AnthropicBackend(runtime).generate(
        session.path,
        "claude-sonnet-4-6",
        RuntimeCallbacks(usage=snapshots.append),
    )

    assert response == "Done."
    assert snapshots == [
        UsageSnapshot(
            total=TokenUsage(
                input_tokens=200,
                output_tokens=12,
                cached_tokens=60,
                cache_creation_tokens=40,
                total_tokens=212,
            ),
            context_tokens=200,
        )
    ]


def test_openai_stream_reports_usage(tmp_path, monkeypatch):
    import anomx.agent.backends.openai as openai_module

    responses = iter(
        (
            [
                b'data: {"type":"response.output_text.delta","delta":"Done."}\n',
                (
                    b'data: {"type":"response.completed","response":{"id":"resp_1","usage":'
                    b'{"input_tokens":328,"input_tokens_details":{"cached_tokens":128},'
                    b'"output_tokens":52,"total_tokens":380}}}\n'
                ),
            ],
        )
    )
    home = AnomxHome(tmp_path / "home")
    home.set_api_key("openai", "openai-key")
    session = home.create_session(tmp_path, provider="openai", model="gpt-5.5")
    home.append_session_event(session.path, "user_message", {"message": "Hello."})
    runtime = AgentRuntime(home, tmp_path)

    def fake_urlopen(request, timeout):
        return FakeStreamResponse(next(responses))

    monkeypatch.setattr(openai_module.urllib.request, "urlopen", fake_urlopen)

    snapshots = []
    response = OpenAIBackend(runtime).generate(
        session.path,
        "gpt-5.5",
        RuntimeCallbacks(usage=snapshots.append),
    )

    assert response == "Done."
    assert snapshots == [
        UsageSnapshot(
            total=TokenUsage(
                input_tokens=328,
                output_tokens=52,
                cached_tokens=128,
                cache_creation_tokens=0,
                total_tokens=380,
            ),
            context_tokens=328,
        )
    ]


def test_ollama_stream_reports_usage(tmp_path, monkeypatch):
    import anomx.agent.backends.ollama as ollama_module

    responses = iter(
        (
            [
                b'{"message":{"role":"assistant","content":"Done."},"done":false}\n',
                (
                    b'{"message":{"role":"assistant"},"done":true,'
                    b'"prompt_eval_count":26,"eval_count":298}\n'
                ),
            ],
        )
    )
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="ollama", model="qwen3.6")
    home.append_session_event(session.path, "user_message", {"message": "Hello."})
    runtime = AgentRuntime(home, tmp_path)

    def fake_urlopen(request, timeout):
        return FakeStreamResponse(next(responses))

    monkeypatch.setattr(ollama_module.urllib.request, "urlopen", fake_urlopen)

    snapshots = []
    response = OllamaBackend(runtime).generate(
        session.path,
        "qwen3.6",
        RuntimeCallbacks(usage=snapshots.append),
    )

    assert response == "Done."
    assert snapshots == [
        UsageSnapshot(
            total=TokenUsage(
                input_tokens=26,
                output_tokens=298,
                cached_tokens=0,
                cache_creation_tokens=0,
                total_tokens=324,
            ),
            context_tokens=26,
        )
    ]


def test_runtime_tracks_last_usage_snapshot(tmp_path, monkeypatch):
    import anomx.agent.backends.openai_chat as openai_chat_module

    responses = iter(
        (
            [
                b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n',
                (
                    b'data: {"choices":[],"usage":{"prompt_tokens":64,"completion_tokens":2,'
                    b'"total_tokens":66}}\n'
                ),
                b"data: [DONE]\n",
            ],
        )
    )
    home = AnomxHome(tmp_path / "home")
    home.set_api_key("kimi", "kimi-key")
    config = home.load_config()
    config["provider"] = "kimi"
    config["model"] = "kimi-k3"
    home.save_config(config)
    session = home.create_session(tmp_path, provider="kimi", model="kimi-k3")
    home.append_session_event(session.path, "user_message", {"message": "Hello."})
    runtime = AgentRuntime(home, tmp_path)

    def fake_urlopen(request, timeout):
        return FakeStreamResponse(next(responses))

    monkeypatch.setattr(openai_chat_module.urllib.request, "urlopen", fake_urlopen)

    snapshots = []
    response = runtime.backend_response(
        session.path,
        callbacks=RuntimeCallbacks(usage=snapshots.append),
    )

    assert response == "Hi"
    assert runtime.last_usage_snapshot is not None
    assert runtime.last_usage_snapshot.context_tokens == 64
    assert runtime.last_usage_snapshot.total.total_tokens == 66
    assert snapshots and snapshots[-1] == runtime.last_usage_snapshot
