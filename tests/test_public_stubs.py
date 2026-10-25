from pathlib import Path
from runpy import run_path

import pytest


def test_lazy_api_stubs_match_every_declared_export():
    root = Path(__file__).resolve().parents[1]
    generate = run_path(str(root / "tools" / "generate_public_stubs.py"))["generate"]
    outputs = generate(root / "src" / "anomx")
    assert outputs
    for path, content in outputs.items():
        assert path.read_text() == content, f"Regenerate the public API stubs: {path}"


def test_lazy_exports_support_static_completion_and_signatures(tmp_path, monkeypatch):
    jedi = pytest.importorskip("jedi")
    import anomx

    monkeypatch.setattr(jedi.settings, "cache_directory", str(tmp_path / "jedi-cache"))
    source = "from anomx import "
    names = {item.name for item in jedi.Script(source).complete(1, len(source))}
    assert set(anomx.__all__) <= names
    source = "from anomx import Dataset as Samples\nSamples.from_values("
    signatures = jedi.Script(source).get_signatures(2, len("Samples.from_values("))
    assert signatures and "values" in signatures[0].to_string()
    source = "from anomx import get_work_context\njob = get_work_context()\njob."
    names = {item.name for item in jedi.Script(source).complete(3, 4)}
    assert {"abort", "check_cancelled", "finding", "artifact", "metric", "log"} <= names
    source = "from anomx import WorkContext\nwork: WorkContext\nwork."
    names = {item.name for item in jedi.Script(source).complete(3, 5)}
    assert {"abort", "log", "metric"} <= names
