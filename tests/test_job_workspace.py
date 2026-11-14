import json
from uuid import uuid4

import pytest

from anomx.integrations.platform import PlatformError
from anomx.integrations.workspaces import commit, mount, save


@pytest.mark.parametrize(
    "url,prefix", [("http://localhost/api", ""), ("http://localhost/api/v1", "/jobs")]
)
def test_job_workspace_roundtrip_and_conflict_preserve_checkout(tmp_path, url, prefix):
    identifier = str(uuid4())

    class Client:
        revision = 2
        files = {"main.py": "print('shared')"}

        def request(self, path, payload=None, *, method="POST"):
            assert path == f"{prefix}/jobs/{identifier}/workspace"
            if method == "GET":
                return {"files": self.files, "entrypoint": "main.py", "revision": self.revision}
            if payload["revision"] != self.revision:
                raise RuntimeError("409 conflict")
            self.files = payload["files"]
            self.revision += 1
            return {"entrypoint": "main.py", "revision": self.revision}

    client = Client()
    client.url = url
    manifest = mount(f"jobs_job-{identifier}", tmp_path, client=client)
    assert manifest["job"] == identifier
    (tmp_path / "main.py").write_text("print('changed')")
    assert save(tmp_path, client=client)["revision"] == 3
    assert client.files["main.py"] == "print('changed')"
    client.revision += 1
    with pytest.raises(RuntimeError, match="409"):
        save(tmp_path, client=client)
    assert json.loads((tmp_path / ".anomx-workspace.json").read_text())["revision"] == 3
    assert (tmp_path / "main.py").read_text() == "print('changed')"


@pytest.mark.parametrize("component", [False, True])
def test_commit_saved_workspace_checks_revision_and_configuration(tmp_path, component):
    job_id, component_id = str(uuid4()), str(uuid4())
    calls = []

    class Client:
        url = "https://example.invalid/api/v1"

        def request(self, path, payload=None, *, method="POST"):
            calls.append((path, method, payload))
            if path.endswith("/workspace"):
                return {
                    "files": {"main.py": "print('shared')"},
                    "entrypoint": "main.py",
                    "revision": 3,
                }
            if path.endswith(component_id):
                return {"job": {"object_reference": f"jobs_job-{job_id}"}}
            if path.endswith("/version-control"):
                return {"digest": "reviewed-config"}
            assert path == f"/jobs/jobs/{job_id}/commit"
            assert payload == {
                "message": "Improve detection",
                "expected_revision": 3,
                "expected_digest": "reviewed-config",
            }
            return {"head": "a" * 40, "has_changes": False}

    client = Client()
    reference = f"jobs_job_component-{component_id}" if component else f"jobs_job-{job_id}"
    mount(reference, tmp_path, client=client)
    assert commit(tmp_path, "Improve detection", client=client)["head"] == "a" * 40
    assert [method for _, method, _ in calls].count("POST") == 1
    assert all(method != "PUT" for _, method, _ in calls)
    (tmp_path / "main.py").write_text("print('unsaved')")
    with pytest.raises(PlatformError, match="Save or merge"):
        commit(tmp_path, "Unsaved", client=client)
    assert [method for _, method, _ in calls].count("POST") == 1


def test_commit_preserves_checkout_on_concurrent_remote_revision(tmp_path):
    job_id = str(uuid4())

    class Client:
        revision = 1

        def request(self, path, payload=None, *, method="POST"):
            assert method == "GET"
            return {
                "files": {"main.py": "pass\n"},
                "entrypoint": "main.py",
                "revision": self.revision,
            }

    client = Client()
    mount(f"jobs_job-{job_id}", tmp_path, client=client)
    client.revision = 2
    with pytest.raises(PlatformError, match="Save or merge"):
        commit(tmp_path, "Stale", client=client)
    assert json.loads((tmp_path / ".anomx-workspace.json").read_text())["revision"] == 1
    assert (tmp_path / "main.py").read_text() == "pass\n"
