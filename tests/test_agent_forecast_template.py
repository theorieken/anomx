"""Execute the optional ML skill asset, including its exported inference contract."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("with_gap", [False, True])
def test_transformer_template_trains_and_round_trips_onnx(tmp_path, monkeypatch, with_gap):
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import anomx

    class Dataset:
        kind = "time_series"

        def __init__(self, reference):
            assert reference == "fixture"

        def describe(self):
            return {"version": "fixed-fixture-v1"}

        def iter_batches(self, batch_size):
            frame = pd.DataFrame({"value": np.sin(np.arange(1400) / 15)},
                                 index=pd.date_range("2026-01-01", periods=1400, freq="s", name="timestamp"))
            if with_gap:
                frame.index = frame.index + pd.to_timedelta((np.arange(1400) >= 700).astype(int), unit="h")
                frame.index.name = "timestamp"
            for offset in range(0, len(frame), batch_size):
                yield frame.iloc[offset:offset + batch_size]

    class Work:
        def log(self, *args, **kwargs):
            pass

        def metric(self, *args, **kwargs):
            pass

        def publish_model(self, artifact, **kwargs):
            assert Path(artifact).stat().st_size > 1000
            assert kwargs["metadata"]["architecture"] == "transformer"
            assert kwargs["metadata"]["normalization"]["fitted_on"] == "training only"
            return {"reference": "verified-fixture-model"}

    monkeypatch.setattr(anomx, "Dataset", Dataset)
    monkeypatch.chdir(tmp_path)
    asset = Path(anomx.__file__).parent / "agent/skills/train-models/assets/forecast.py"
    namespace = {"work": Work(), "inputs": {
        "dataset_reference": "fixture", "target_column": "value", "model_name": "Fixture",
        "order_column": "timestamp", "max_gap_seconds": 2, "lookback": 24, "horizon": 3,
        "architecture": "transformer", "epochs": 3, "hidden_size": 8, "attention_heads": 2,
        "layers": 1, "batch_size": 32,
    }}
    exec(compile(asset.read_text(), str(asset), "exec"), namespace)
    result = namespace["result"]
    assert result["metrics"]["test_windows"] > 100
    assert np.isfinite(result["metrics"]["test_rmse"])
    metrics = result["metrics"]
    assert bool(metrics["quality_gate_passed"]) == (
        metrics["validation_model_mse"] < metrics["validation_persistence_mse"] * 0.95
    )
    assert metrics["validation_model_mse"] == pytest.approx(metrics["validation_rmse"] ** 2)
    assert result["dataset_version"] == "fixed-fixture-v1"
    np.testing.assert_allclose(namespace["train_values"], np.sin(np.arange(980) / 15), atol=1e-7)
    if with_gap:
        assert namespace["split_details"][0]["boundaries"] == [0, 700, 700, 700]
        assert namespace["split_details"][1]["boundaries"] == [0, 280, 490, 700]
    assert (tmp_path / "training-metadata.json").is_file()
