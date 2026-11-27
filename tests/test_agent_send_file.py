from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from anomx.agent import MainAgent, SubAgent
from anomx.agent.base.tools import ToolExecutionContext
from anomx.agent.runtime import RuntimeCallbacks
from anomx.agent.runtime_process import RuntimeProcessClient
from anomx.agent.tools.send_file import SendFileTool


def context_for(root: Path, *, enabled=True, denied=False, result=None):
    delivered = []
    authorization = []

    def authorize(*args, **kwargs):
        authorization.append((args, kwargs))
        return (
            SimpleNamespace(blocked_by_mode=True, output="Writes are disabled") if denied else None
        )

    def send(payload):
        delivered.append(payload)
        return (
            result
            if result is not None
            else {"ok": True, "object_reference": "core_file-38df2566-9198-4b37-a8ad-2036a5d8d6cc"}
        )

    manager = SimpleNamespace(
        resolve_trusted_path=lambda value: (root / value).resolve(),
        path_inside_workspace=lambda value: value.is_relative_to(root),
        authorize_platform_api_request=authorize,
    )
    runtime = SimpleNamespace(
        can_output_response=lambda: enabled,
        tool_manager=manager,
        home=SimpleNamespace(brain_dir=root / "brain"),
    )
    callbacks = RuntimeCallbacks(send_file=send)
    return ToolExecutionContext(runtime=runtime, callbacks=callbacks), delivered, authorization


def execute(context, path="result.bin", **kwargs):
    return json.loads(
        SendFileTool(statement_description="Working").execute(
            {"statement": "Delivering the result", "path": path, **kwargs},
            context,
        )
    )


def test_binary_delivery_passes_a_path_and_confirmed_reference_without_file_bytes(tmp_path):
    (tmp_path / "result.bin").write_bytes(b"\x00private binary content\xff")
    context, delivered, authorization = context_for(tmp_path)
    result = execute(context, filename="export.bin")
    assert result["ok"] is True
    assert result["object_reference"].startswith("core_file-")
    assert delivered == [
        {"path": str(tmp_path / "result.bin"), "filename": "export.bin", "description": ""}
    ]
    assert authorization[0][0][:2] == ("POST", "/files/upload")
    assert "private binary content" not in json.dumps(result)


@pytest.mark.parametrize("enabled,denied", [(False, False), (True, True)])
def test_delivery_requires_a_platform_host_and_writable_mode(tmp_path, enabled, denied):
    (tmp_path / "result.bin").write_bytes(b"result")
    context, delivered, _ = context_for(tmp_path, enabled=enabled, denied=denied)
    assert execute(context)["ok"] is False
    assert delivered == []


def test_outside_workspace_and_symlink_escape_are_not_delivered(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("must not be published")
    (root / "linked.txt").symlink_to(secret)
    context, delivered, _ = context_for(root)
    assert execute(context, "../secret.txt")["ok"] is False
    assert execute(context, "linked.txt")["ok"] is False
    assert delivered == []


@pytest.mark.parametrize(
    "result", [{"ok": False, "error": "Storage unavailable"}, {"ok": True}, {}]
)
def test_unconfirmed_upload_never_claims_success(tmp_path, result):
    (tmp_path / "result.bin").write_bytes(b"result")
    context, delivered, _ = context_for(tmp_path, result=result)
    assert execute(context)["ok"] is False
    assert len(delivered) == 1


def test_directory_and_invalid_download_name_do_not_reach_host(tmp_path):
    (tmp_path / "result.bin").write_bytes(b"result")
    context, delivered, _ = context_for(tmp_path)
    assert execute(context, ".")["ok"] is False
    assert execute(context, filename="../result.bin")["ok"] is False
    assert delivered == []


def test_delivery_is_main_agent_only_and_callback_is_in_process_bridge():
    assert "send_file" in {tool.name for tool in MainAgent().tools}
    assert "send_file" not in {tool.name for tool in SubAgent().tools}

    def callback(payload):
        return payload

    assert (
        RuntimeProcessClient._callback_map(RuntimeCallbacks(send_file=callback))["send_file"]
        is callback
    )
