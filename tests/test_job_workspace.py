import json
from uuid import uuid4

import pytest

from anomx.integrations.workspaces import mount, save


@pytest.mark.parametrize("url,prefix", [("http://localhost/api", ""), ("http://localhost/api/v1", "/jobs")])
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
