"""Behavior and dependency checks for the package's domain boundaries."""

import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from anomx.base import BaseComponent, iter_component_classes
from anomx.data import Dataset
from anomx.data.base import current_dataset_client, use_dataset_client
from anomx.detection import Detector, Scorer
from anomx.detection.detectors import ThresholdDetector
from anomx.detection.scorers import AbsoluteErrorScorer
from anomx.execution import CallbackBackend, Run, WorkContext, current_run, init, log, use_run
from anomx.models import ConstantBaselineModel, Model, NormalityModel


def test_domain_exports_share_the_native_contracts():
    assert issubclass(ConstantBaselineModel, NormalityModel)
    assert issubclass(NormalityModel, Model)
    assert issubclass(Model, BaseComponent)
    assert issubclass(AbsoluteErrorScorer, Scorer)
    assert issubclass(ThresholdDetector, Detector)
    assert importlib.import_module("anomx").Scorer is Scorer
    assert importlib.import_module("anomx").Detector is Detector


def test_inference_only_models_do_not_require_training_or_persistence():
    class Identity(Model):
        def predict(self, data):
            return data

    model = Identity()
    assert model.predict([1, 2]) == [1, 2]
    assert not hasattr(model, "fit")


def test_native_discovery_keeps_catalog_and_resolvable_paths():
    components = list(iter_component_classes())
    keys = [component.get_component_key() for component in components]
    assert len(keys) == len(set(keys))
    assert set(keys) == {
        "absolute_error",
        "composite_scorer",
        "constant_baseline",
        "darts_naive_seasonal",
        "isolation_forest",
        "pca_reconstruction",
        "pipeline",
        "quantile_threshold",
        "rolling_window_forecast",
        "threshold",
        "torch_autoencoder",
        "zscore",
    }
    for component in components:
        module, _, name = component.get_component_import_path().rpartition(".")
        assert getattr(importlib.import_module(module), name) is component
        assert not module.startswith("anomx.components")
        assert component.get_component_source_path().startswith("src/anomx/")


def test_core_imports_and_discovery_do_not_load_optional_runtimes():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import json
import sys

class DenyOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'darts', 'torch', 'onnx', 'onnxruntime', 'django'}:
            raise AssertionError('Core imported optional runtime: ' + fullname)
        if fullname in {'anomx.agent', 'anomx_platform'}:
            raise AssertionError('Core imported application: ' + fullname)

sys.meta_path.insert(0, DenyOptional())
from anomx.data import Dataset
from anomx.data.processing import ProcessingPipeline, Resample
from anomx.models import Model
from anomx.detection import Scorer, Detector, AnomalyPipeline
from anomx.execution import Run, JobRunner, WorkContext
from anomx.base import iter_component_classes
print(json.dumps(sorted(c.get_component_key() for c in iter_component_classes())))
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "pca_reconstruction" in json.loads(result.stdout)


def test_domain_dependencies_do_not_point_to_execution_or_agent():
    source = Path(__file__).resolve().parents[1] / "src/anomx"
    forbidden = {
        "data": ("anomx.models", "anomx.detection", "anomx.execution", "anomx.agent"),
        "models": ("anomx.detection", "anomx.execution", "anomx.agent"),
        "detection": ("anomx.execution", "anomx.agent"),
    }
    for domain, prefixes in forbidden.items():
        for path in (source / domain).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                modules = []
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                assert not any(
                    module == prefix or module.startswith(prefix + ".")
                    for module in modules
                    for prefix in prefixes
                ), path


class _Client:
    def __init__(self, value):
        self.value = value

    def describe_dataset(self, reference):
        return {"version": "v1", "kind": "independent_samples", "mode": "fixed"}

    def read_dataset(self, reference, **kwargs):
        return {"version": "v1", "columns": ["x"], "records": [{"x": self.value}]}


def test_dataset_clients_are_scoped_and_explicit_client_wins():
    outer, inner = _Client(1), _Client(2)
    assert current_dataset_client() is None
    with use_dataset_client(outer):
        assert Dataset("dataset").to_dataframe().x.tolist() == [1]
        with pytest.raises(RuntimeError), use_dataset_client(inner):
            assert Dataset("dataset").to_dataframe().x.tolist() == [2]
            assert Dataset("dataset", client=outer).to_dataframe().x.tolist() == [1]
            raise RuntimeError("Leave nested context")
        assert current_dataset_client() is outer
    assert current_dataset_client() is None


def test_work_context_uses_the_existing_run_and_restores_dataset_client():
    events = []
    run = Run(project="example", managed=True, backend=CallbackBackend(events.append))
    with use_dataset_client(_Client(9)), use_run(run), WorkContext().activate():
        assert current_run() is run
        assert init() is run
        log("loss", 0.25)
        with pytest.raises(RuntimeError, match="does not support describe_dataset"):
            Dataset("dataset").describe()
    assert current_run() is None
    assert current_dataset_client() is None
    assert [event["kind"] for event in events] == ["history"]


def test_standalone_dataset_does_not_import_execution():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from anomx.data import Dataset
data = Dataset.from_values([1, 2, 3])
assert len(data.to_dataframe()) == 3
assert not any(name.startswith('anomx.execution') for name in sys.modules)
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
