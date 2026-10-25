import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from anomx.agent import AgentMode, AnomxHome
from anomx.agent.context_management import ContextMessage
from anomx.agent.helpers.tool_results import MAX_INLINE_READ_CHARACTERS, store_tool_result
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks


@pytest.fixture
def runtime(tmp_path):
    home = AnomxHome(tmp_path / "home")
    home.save_config({**home.load_config(), "maximum_context_tokens": 32_000})
    return AgentRuntime(home, tmp_path, mode=AgentMode.AUTONOMOUS)


@pytest.mark.parametrize("tool", ["search", "get_details", "get_history", "custom_tool"])
def test_large_result_preserves_file_and_protocol_on_resume(runtime, tmp_path, monkeypatch, tool):
    raw = json.dumps(
        {"ok": True, "channels": [{"identifier": "FLASH/X/Y/Z", "last_value": "1," * 150000}]}
    )
    session = runtime.home.create_session(tmp_path, provider="desy", model="coding")
    monkeypatch.setattr(
        runtime, "_tool_for_call", lambda _: SimpleNamespace(name=tool, execute=lambda *_: raw)
    )
    runtime.context_optimizer = lambda *_: pytest.fail("No model needed to store large results")
    result = runtime._execute_tool(
        tool, {}, RuntimeCallbacks(), session.path, tool_call_id="native-1"
    )
    compact = json.loads(result)
    assert len(result) < 6000 and compact["ok"] is True
    assert Path(compact["result_path"]).read_text() == raw
    assert result.optimized
    resumed = AgentRuntime(runtime.home, tmp_path).backend_conversation_entries(session.path)
    assert resumed[-1].payload["tool_call_id"] == "native-1"
    assert resumed[-1].payload["content"] == result
    assert runtime.conversation_messages(session.path)[-1]["content"] == raw
    assert runtime.tool_manager.path_inside_workspace(Path(compact["result_path"]))


def test_oversized_single_record_in_old_history_needs_no_model(runtime, tmp_path):
    session = runtime.home.create_session(tmp_path, provider="desy", model="coding")
    raw = json.dumps({"channels": [{"last_value": "x" * 287000}]})
    entries = [ContextMessage("t1", {"role": "tool", "content": raw, "tool_call_id": "a"})]
    result, changed = runtime._optimize_tool_blocks(
        session.path, entries, maximum=32000, current_context_tokens=80000
    )
    assert changed
    assert json.loads(
        Path(json.loads(result[0].payload["content"])["result_path"]).read_text()
    ) == json.loads(raw)


def test_long_line_is_readable_in_bounded_pages(runtime, tmp_path):
    text = "A" * 8000 + "B" * 8000 + "C" * 1000
    path = tmp_path / "one-line.txt"
    path.write_text(text)
    parts = []
    offset = 0
    while offset is not None:
        result = json.loads(
            runtime._execute_tool(
                "read",
                {"path": str(path), "start_line": 1, "max_lines": 1, "start_character": offset},
                RuntimeCallbacks(),
            )
        )
        assert len(result["content"]) <= 8000
        parts.append(result["content"])
        offset = result["next_character"]
    assert "".join(parts) == text


def test_large_read_stays_inline_and_survives_session_resume(runtime, tmp_path):
    runtime.home.save_config({**runtime.home.load_config(), "maximum_context_tokens": 1_000_000})
    path = tmp_path / "large.json"
    text = "\n".join(f'"field{index}": {index},' for index in range(3000))
    path.write_text(text)
    session = runtime.home.create_session(tmp_path, provider="desy", model="coding")
    result = runtime._execute_tool(
        "read", {"path": str(path), "max_lines": 10_000}, RuntimeCallbacks(), session.path,
        tool_call_id="large-read",
    )
    payload = json.loads(result)
    assert 16_000 < len(result) < MAX_INLINE_READ_CHARACTERS
    assert payload["content"] == text
    assert payload["line_count"] == 3000
    assert payload["eof"] and payload["next_read"] is None
    resumed = AgentRuntime(runtime.home, tmp_path).backend_conversation_entries(session.path)
    assert resumed[-1].payload["content"] == result
    assert resumed[-1].payload["tool_call_id"] == "large-read"
    entries, changed = runtime._optimize_tool_blocks(
        session.path, resumed, maximum=1_000_000, current_context_tokens=30_000
    )
    assert not changed
    assert entries[-1].payload["content"] == result


def test_read_uses_larger_default_and_clamps_line_limit(runtime, tmp_path):
    path = tmp_path / "many-lines.txt"
    path.write_text("x\n" * 10_001)
    first = json.loads(runtime._execute_tool("read", {"path": str(path)}, RuntimeCallbacks()))
    assert first["line_count"] == 2000
    assert first["next_read"] == {
        "path": str(path), "start_line": 2001, "max_lines": 2000, "start_character": 0,
    }
    assert not first["eof"]
    runtime.home.save_config({**runtime.home.load_config(), "maximum_context_tokens": 1_000_000})
    capped = json.loads(runtime._execute_tool(
        "read", {"path": str(path), "max_lines": 50_000}, RuntimeCallbacks()
    ))
    assert capped["line_count"] == 10_000
    assert capped["next_read"]["start_line"] == 10_001
    assert capped["next_read"]["max_lines"] == 10_000


@pytest.mark.parametrize("large_context", [False, True])
@pytest.mark.parametrize("character", ["A", "\x00"])
def test_read_cursor_reconstructs_long_lines_without_nested_results(
    runtime, tmp_path, large_context, character,
):
    if large_context:
        runtime.home.save_config({
            **runtime.home.load_config(), "maximum_context_tokens": 1_000_000,
        })
    path = tmp_path / "long-line.txt"
    text = character * 140_000
    path.write_text(text)
    arguments = {"path": str(path), "start_line": 1, "max_lines": 1}
    parts = []
    while arguments is not None:
        result = runtime._execute_tool("read", arguments, RuntimeCallbacks())
        assert len(result) < MAX_INLINE_READ_CHARACTERS
        page = json.loads(result)
        assert page["content"]
        assert not page.get("repeated_read")
        parts.append(page["content"])
        arguments = page["next_read"]
        assert page["eof"] == (arguments is None)
        if arguments:
            assert arguments["start_line"] == 1
            assert arguments["max_lines"] == 1
            assert arguments["start_character"] == sum(map(len, parts))
    assert "".join(parts) == text


def test_alternating_eof_reads_report_repetition_without_hiding_content(runtime, tmp_path):
    path = tmp_path / "channel.json"
    path.write_text("metadata\n" * 1182)
    for attempt in range(2):
        for start in (1120, 1240):
            page = json.loads(runtime._execute_tool(
                "read", {"path": str(path), "start_line": start, "max_lines": 120},
                RuntimeCallbacks(),
            ))
            assert page["eof"] and page["next_read"] is None
            assert page["line_count"] == (63 if start == 1120 else 0)
            assert bool(page.get("repeated_read")) == (attempt == 1)
            assert "End of file" in page["read_hint"]
    path.write_text("changed\n" * 1182)
    changed = json.loads(runtime._execute_tool(
        "read", {"path": str(path), "start_line": 1120, "max_lines": 120}, RuntimeCallbacks()
    ))
    assert not changed.get("repeated_read")
    assert changed["content"].startswith("changed")
    path.write_text("changed\n" * 1300)
    appended = json.loads(runtime._execute_tool(
        "read", {"path": str(path), "start_line": 1240, "max_lines": 120}, RuntimeCallbacks()
    ))
    assert not appended.get("repeated_read")
    assert appended["content"]


def test_repeated_clipped_range_points_to_character_continuation(runtime, tmp_path):
    path = tmp_path / "job.json"
    path.write_text("x" * 20_000)
    arguments = {"path": str(path), "start_line": 1, "max_lines": 120}
    runtime._execute_tool("read", arguments, RuntimeCallbacks())
    page = json.loads(runtime._execute_tool("read", arguments, RuntimeCallbacks()))
    assert page["repeated_read"] and not page["eof"]
    assert page["next_read"]["start_character"] == len(page["content"])
    assert "start_character" in page["read_hint"]
    runtime._with_usage_tracking(RuntimeCallbacks())
    next_turn = json.loads(runtime._execute_tool("read", arguments, RuntimeCallbacks()))
    assert not next_turn.get("repeated_read")


def test_invalid_character_cursor_is_not_reported_as_eof(runtime, tmp_path):
    path = tmp_path / "short.txt"
    path.write_text("abc")
    page = json.loads(runtime._execute_tool(
        "read", {"path": str(path), "start_character": 100}, RuntimeCallbacks()
    ))
    assert "error" in page
    assert "eof" not in page


def test_command_file_retains_middle_rows_and_deduplicates(runtime):
    text = "\n".join(f"measurement-{index}" for index in range(1000))
    result = json.loads(runtime.tool_manager._abbreviate_command_output(text))
    assert Path(result["result_path"]).read_text() == text
    assert (
        json.loads(runtime.tool_manager._abbreviate_command_output(text))["result_path"]
        == result["result_path"]
    )


def test_48_messages_do_not_trigger_summary_without_token_pressure(runtime, tmp_path):
    session = runtime.home.create_session(tmp_path, provider="desy", model="coding")
    runtime.context_summarizer = lambda *_: "Keep the verified measurement and the current goal."
    for index in range(48):
        runtime.home.append_session_event(
            session.path, "user_message", {"message_id": str(index), "message": "measurement " * 30}
        )
    runtime._context_evaluations[session.path] = (50, 40)
    _, changed = runtime.compress_in_turn_context(
        session.path,
        runtime.backend_conversation_entries(session.path),
        current_context_tokens=17000,
        status_callback=None,
    )
    assert not changed and runtime.context_compression_state(session.path) is None


def test_short_but_large_history_summarizes_before_limit(runtime, tmp_path):
    session = runtime.home.create_session(tmp_path, provider="desy", model="coding")
    runtime.context_summarizer = lambda *_: "Keep the original goal and verified outcomes."
    for index in range(12):
        runtime.home.append_session_event(
            session.path,
            "user_message" if index % 2 == 0 else "agent_message",
            {"message_id": str(index), "message": "context " * (14000 if index == 1 else 2)},
        )
    _, changed = runtime.compress_in_turn_context(
        session.path,
        runtime.backend_conversation_entries(session.path),
        current_context_tokens=30000,
        status_callback=None,
    )
    assert changed
    assert runtime.context_compression_state(session.path).reason == "context_pressure"
    assert runtime.backend_conversation_entries(session.path)[-1].message_id == "11"


def test_search_bar_tool_returns_compact_hits_and_raw_response(runtime, monkeypatch):
    runtime.home.set_platform_connection(url="https://platform.test", token="test")
    payload = {
        "items": [
            {
                "value": "FLASH.RF/X/Y/Z",
                "channel": {
                    "identifier": "FLASH.RF/X/Y/Z",
                    "_anomx": {"object_reference": "data_channel-id"},
                    "last_value": "x" * 300000,
                },
            },
            {"value": "FLASH.RF/X/Y/W", "source": "typed", "channel": None},
        ],
        "page": 1,
        "next_page": 2,
        "has_more": True,
        "loading": True,
        "discovery_complete": False,
    }
    requests = []

    def request(req, **_):
        requests.append(req.full_url)
        response = io.BytesIO(json.dumps(payload).encode())
        response.status = 200
        response.headers = {"content-type": "application/json"}
        return response

    monkeypatch.setattr("anomx.agent.helpers.anomx_api.urlopen", request)
    result = json.loads(
        runtime._execute_tool(
            "search_anomx_data_channels",
            {"query": "FLASH.RF/*/Y/", "refresh": False},
            RuntimeCallbacks(),
        )
    )
    assert "/channels/search?" in requests[0] and "refresh=false" in requests[0]
    assert len(json.dumps(result)) < 4000
    assert result["matches"][0]["object_reference"] == "data_channel-id"
    assert result["matches"][1]["availability"] == "unverified"
    assert "last_value" not in json.dumps(result)
    assert result["loading"] and result["next_page"] == 2
    assert json.loads(Path(result["request"]["response_path"]).read_text()) == payload


def test_result_files_preserve_failure_status_and_unbounded_keys(tmp_path):
    result = json.loads(
        store_tool_result(
            json.dumps({"ok": False, "error": "denied", "x" * 100000: "y" * 100000}), tmp_path
        )
    )
    assert result["ok"] is False and result["error"] == "denied"
    assert len(json.dumps(result)) < 6000


def test_artifact_digest_matches_saved_bytes_and_failed_write_is_retryable(tmp_path, monkeypatch):
    from anomx.agent.helpers import tool_results

    original_replace = tool_results.os.replace

    def fail(*_):
        raise OSError("disk full")

    monkeypatch.setattr(tool_results.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        store_tool_result('{"measurements":[1,2,3]}', tmp_path)
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(tool_results.os, "replace", original_replace)
    result = json.loads(store_tool_result('{"measurements":[1,2,3]}', tmp_path))
    saved = Path(result["result_path"]).read_bytes()
    assert result["result_sha256"] == hashlib.sha256(saved).hexdigest()
    assert result["result_bytes"] == len(saved)
    exact = '{"precise": 0.12345678901234567890123456789, "repeated": 1, "repeated": 2}'
    artifact = json.loads(store_tool_result(exact, tmp_path))
    assert Path(artifact["result_path"]).read_text() == exact


def test_artifact_failure_does_not_return_an_oversized_result(runtime, monkeypatch):
    from anomx.agent.exceptions import AgentBackendError

    monkeypatch.setattr(
        runtime, "_tool_for_call",
        lambda _: SimpleNamespace(name="custom_tool", execute=lambda *_: "x" * 300000),
    )

    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("anomx.agent.runtime.store_tool_result", fail)
    with pytest.raises(AgentBackendError, match="Cannot preserve a large tool result"):
        runtime._execute_tool("custom_tool", {}, RuntimeCallbacks())


def test_escaped_file_content_remains_directly_readable(runtime, tmp_path):
    path = tmp_path / "control-characters.txt"
    path.write_text("\x00" * 10000)
    result = runtime._execute_tool("read", {"path": str(path)}, RuntimeCallbacks())
    payload = json.loads(result)
    assert len(result) < 16000 and payload["content"]
    assert payload["next_character"] == len(payload["content"])


def test_async_output_keeps_complete_artifact(runtime):
    state = SimpleNamespace(
        process_id="async-1", output_chunks=["start\n", "middle" * 1000, "\nend"], output="end"
    )
    runtime._processes[state.process_id] = state
    payload = json.loads(runtime._current_process_output(state))
    assert Path(payload["result_path"]).read_text() == "".join(state.output_chunks)
