import json
from uuid import uuid4

import pytest

import anomx
from anomx.execution import CallbackBackend, Run, current_run, init, log, use_run
from anomx.integrations.workspaces import mount, save


def test_steps_pending_values_and_context_restoration():
    events = []
    backend = CallbackBackend(events.append)
    with init(project="forecast", backend=backend) as outer:
        log("loss", 3, commit=False)
        log({"accuracy": 0.7})
        assert outer.summary == {"loss": 3, "accuracy": 0.7}
        assert outer.step == 1
        with init(project="inner", backend=backend):
            log("loss", 1, step=7)
        assert current_run() is outer
        with pytest.raises(ValueError):
            log("loss", 2, step=0)
        with pytest.raises(ValueError):
            log("loss", float("nan"))
    assert current_run() is None
    history = [event for event in events if event["kind"] == "history"]
    assert history[0]["step"] == 0
    assert history[0]["values"] == {"loss": 3, "accuracy": 0.7}
    assert history[1]["step"] == 7
    assert [event["status"] for event in events if event["kind"] == "finish"] == [
        "succeeded",
        "succeeded",
    ]


def test_retry_preserves_id_and_pending_values():
    events = []
    failed = False

    def write(event):
        nonlocal failed
        events.append(event)
        if event["kind"] == "history" and not failed:
            failed = True
            raise OSError("ack lost")

    run = Run(project="offline", backend=CallbackBackend(write))
    with pytest.raises(OSError):
        run.log("score", 2)
    assert run.step == 0
    run.flush()
    assert run.step == 1
    assert events[-1]["event_id"] == events[-2]["event_id"]
    assert run.summary == {"score": 2}


def test_hosted_run_reuses_worker_and_does_not_finish_lifecycle():
    events = []
    run = Run(project="project-id", managed=True, backend=CallbackBackend(events.append))
    with use_run(run):
        assert anomx.init() is run
        anomx.log("score", 4)
        anomx.finish()
    assert [event["kind"] for event in events] == ["history"]
    assert events[0]["step"] is None
    assert current_run() is None


def test_failed_context_and_offline_jsonl(tmp_path):
    with pytest.raises(RuntimeError), init(project="test", directory=tmp_path):
        log("count", 2)
        raise RuntimeError("operation failed")
    events = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    assert events[-1]["status"] == "failed"
    assert events[-1]["error"] == "operation failed"
    assert current_run() is None


def test_workspace_round_trip_and_conflict_keeps_local_edits(tmp_path):
    identifier = str(uuid4())

    class Client:
        revision = 3
        files = {
            "main.py": "from lib.code import answer\nresult = answer",
            "lib/code.py": "answer = 42",
        }

        def request(self, path, payload=None, *, method="POST"):
            assert path == f"/job-works/{identifier}/workspace"
            if method == "GET":
                return {"files": self.files, "entrypoint": "main.py", "revision": self.revision}
            if payload["revision"] != self.revision:
                raise RuntimeError("409 conflict")
            self.files = payload["files"]
            self.revision += 1
            return {"entrypoint": "main.py", "revision": self.revision}

    client = Client()
    target = tmp_path / "work"
    mount(identifier, target, client=client)
    (target / "lib/code.py").write_text("answer = 7")
    assert save(target, client=client)["revision"] == 4
    assert client.files["lib/code.py"] == "answer = 7"
    client.revision = 5
    with pytest.raises(RuntimeError, match="conflict"):
        save(target, client=client)
    assert json.loads((target / ".anomx-workspace.json").read_text())["revision"] == 4
    assert (target / "lib/code.py").read_text() == "answer = 7"


def test_workspace_rejects_traversal_before_writing(tmp_path):
    class Client:
        def request(self, *args, **kwargs):
            return {"files": {"../escape.py": "bad"}, "revision": 0, "entrypoint": "main.py"}

    with pytest.raises(ValueError):
        mount(str(uuid4()), tmp_path / "work", client=Client())
    assert not (tmp_path / "escape.py").exists()


def test_managed_pending_values_keep_explicit_step():
    events = []
    run = Run(project="project", managed=True, backend=CallbackBackend(events.append))
    run.log("loss", 0.3, step=20, commit=False)
    run.log("accuracy", 0.8)
    run.log("tail", 1, commit=False)
    run.finish()
    assert [(event["step"], event["values"]) for event in events] == [
        (20, {"loss": 0.3, "accuracy": 0.8}),
        (None, {"tail": 1}),
    ]
