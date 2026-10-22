"""Execute bundled training assets on synthetic data without a platform or network."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import anomx
from anomx import Dataset
from anomx.models import LinearRegressionModel

ASSETS = Path(anomx.__file__).parent / "agent/skills/train-models/assets"


def load_asset(name, module_name):
    spec = importlib.util.spec_from_file_location(module_name, ASSETS / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class RecordingWork:
    def __init__(self):
        self.metrics = []
        self.artifacts = []

    def log(self, *args, **kwargs):
        pass

    def metric(self, name, value, step=None):
        assert np.isfinite(value)
        self.metrics.append((name, value, step))

    def artifact(self, path):
        assert Path(path).is_file()
        self.artifacts.append(path)
        return {"name": path}


@pytest.fixture
def template(monkeypatch):
    module = load_asset("native-forecast.py", "native_training_fixture")
    times = pd.date_range("2026-10-01", periods=240, freq="min", tz="UTC")
    frame = pd.DataFrame({"detuning": np.sin(np.arange(240) / 9)}, index=times)

    class FixtureDataset(Dataset):
        def describe(self, **kwargs):
            return {**super().describe(**kwargs), "version": "fixture-v1"}

    def source(reference, **kwargs):
        assert reference == "fixture"
        return FixtureDataset(data=frame, kind="time_series", **kwargs)

    source.from_dataframe = Dataset.from_dataframe
    monkeypatch.setattr(module, "Dataset", source)
    inputs = {
        "dataset_reference": "fixture",
        "target_columns": ["detuning"],
        "start": times[0].isoformat(),
        "end": (times[-1] + pd.Timedelta(minutes=1)).isoformat(),
        "lags": [3, 6],
        "horizon": 2,
        "max_rows": 240,
    }
    return module, frame, inputs


def test_native_template_selects_on_validation_and_preserves_replay(
    template, tmp_path, monkeypatch
):
    module, frame, inputs = template
    monkeypatch.chdir(tmp_path)
    first = module.run(inputs, RecordingWork())
    model = LinearRegressionModel().load("forecast.pkl")
    training_values = model.training_series.values().copy()
    procedure = json.loads(Path("procedure.json").read_text())
    forecasts = pd.read_csv("test-predictions.csv")
    assert procedure["split_positions"] == [0, 168, 204, 240]
    assert first["dataset_version"] == "fixture-v1"
    assert first["metrics"]["test_forecasts"] == 35
    assert pd.Timestamp(forecasts["timestamp_utc"].iloc[0]) == frame.index[205].tz_localize(None)
    np.testing.assert_allclose(
        forecasts["detuning:absolute_error"],
        abs(forecasts["detuning:actual"] - forecasts["detuning:prediction"]),
    )
    assert first["metrics"]["quality_gate_passed"] == float(
        first["metrics"]["validation_rmse"] < first["metrics"]["validation_persistence_rmse"] * 0.95
    )
    assert {item["name"] for item in first["artifacts"]} == {
        "procedure.json",
        "test-predictions.csv",
        "forecast.pkl",
        "scaler.pkl",
    }
    frame.iloc[204:, 0] += 1000
    second = module.run(inputs, RecordingWork())
    assert second["selected"] == first["selected"]
    assert second["metrics"]["test_rmse"] != first["metrics"]["test_rmse"]
    np.testing.assert_array_equal(
        LinearRegressionModel().load("forecast.pkl").training_series.values(), training_values
    )


def test_native_template_pushes_selection_and_aggregation_before_row_budget(
    template, tmp_path, monkeypatch
):
    module, frame, inputs = template
    monkeypatch.chdir(tmp_path)
    inputs.update(start=frame.index[20].isoformat(), frequency="2min", max_rows=110)
    module.run(inputs, RecordingWork())
    procedure = json.loads(Path("procedure.json").read_text())
    assert procedure["rows"] == 110
    assert procedure["selection"]["frequency"] == "120000000000ns"
    assert procedure["selection"]["aggregation"] == "mean"


@pytest.mark.parametrize("invalid", ["oversized", "gap", "nonfinite", "architecture", "unbounded"])
def test_native_template_rejects_silent_data_or_model_substitutions(
    template, tmp_path, monkeypatch, invalid
):
    module, frame, inputs = template
    monkeypatch.chdir(tmp_path)
    if invalid == "oversized":
        inputs["max_rows"] = 200
        error = "exceeds max_rows"
    elif invalid == "gap":
        frame.drop(frame.index[30], inplace=True)
        error = "gaps or irregular"
    elif invalid == "nonfinite":
        frame.iloc[30, 0] = np.nan
        error = "Missing/nonfinite"
    elif invalid == "unbounded":
        inputs["start"] = None
        error = "start must be an explicit"
    else:
        inputs["architecture"] = "unavailable_lstm"
        error = "adapt the native model factory"
    with pytest.raises(ValueError, match=error):
        module.run(inputs, RecordingWork())
    assert not Path("forecast.pkl").exists()


def test_lstm_template_uses_native_training_and_retains_requested_architecture(
    template, tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    from anomx.models import TorchForecastingModel

    load_asset("lstm-model.py", "lstm_model")
    module, _, inputs = template
    monkeypatch.chdir(tmp_path)
    inputs.update(architecture="lstm", lags=[3], epochs=1, hidden_size=2, batch_size=64)
    work = RecordingWork()
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        result = module.run(inputs, work)
    finally:
        torch.set_num_threads(old_threads)
    model = TorchForecastingModel().load("forecast.pkl")
    assert isinstance(model.module.recurrent, torch.nn.LSTM)
    assert len(model.training_history) == 1
    assert any(name == "candidate_1_train_loss" and step == 1 for name, _, step in work.metrics)
    assert json.loads(Path("procedure.json").read_text())["architecture"] == "lstm"
    assert result["metrics"]["test_forecasts"] == 35


def test_compute_entrypoint_defers_scientific_module_to_selected_compute():
    class Work:
        def compute(self, code, inputs):
            assert "from training import run" in code
            assert inputs == {"architecture": "lstm"}
            compile(code, "remote", "exec")
            return {"metrics": {"test_rmse": 1.0}}

        def log(self, *args, **kwargs):
            pass

    namespace = {"work": Work(), "inputs": {"architecture": "lstm"}}
    exec(compile((ASSETS / "compute-work.py").read_text(), "main.py", "exec"), namespace)
    assert namespace["result"] == {"metrics": {"test_rmse": 1.0}}
