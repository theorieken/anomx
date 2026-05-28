import pandas as pd

from anomx import discover_component_payloads
from anomx.components.algorithms import JobSpec, PipelineOrchestrator
from anomx.components.models import (
    IsolationForestModel,
    PcaReconstructionModel,
    RollingWindowForecastModel,
)


def test_component_discovery_exposes_all_approaches():
    payloads = discover_component_payloads()
    keys = {payload["key"] for payload in payloads}

    assert "pipeline" in keys
    assert "rolling_window_forecast" in keys
    assert "pca_reconstruction" in keys
    assert "isolation_forest" in keys
    assert "threshold" in keys
    assert "zscore" in keys


def test_approach_models_emit_model_score(tmp_path):
    frame = pd.DataFrame(
        {
            "value": [1.0, 1.1, 1.2, 5.0, 1.1, 1.0],
            "other": [0.5, 0.6, 0.55, 0.7, 0.52, 0.48],
        }
    )
    models = [
        RollingWindowForecastModel({"window": 2}),
        PcaReconstructionModel({"n_components": 1}),
        IsolationForestModel({"random_state": 7}),
    ]

    for model in models:
        model.fit(frame)
        prediction = model.predict(frame)
        assert "model_score" in prediction.columns

        artifact_path = tmp_path / f"{model.get_component_key()}.pkl"
        model.save(str(artifact_path))
        restored = type(model)({}).load(str(artifact_path))
        assert restored.get_component_key() == model.get_component_key()


def test_pipeline_orchestrator_runs_end_to_end(tmp_path):
    frame = pd.DataFrame(
        {
            "feature_a": [0.1, 0.2, 0.15, 4.0, 0.18, 0.19],
            "feature_b": [1.0, 1.1, 1.05, 3.5, 1.08, 1.02],
        }
    )
    csv_path = tmp_path / "signals.csv"
    frame.to_csv(csv_path, index=False)

    orchestrator = PipelineOrchestrator()
    result = orchestrator.run_job(
        JobSpec(
            connector="local_fs",
            model="pca_reconstruction",
            scorer="zscore",
            detector="threshold",
            config={
                "connector": {"path": str(csv_path)},
                "model": {"n_components": 1},
                "detector": {"threshold": 1.0},
            },
        )
    )

    assert result.status == "completed"
    assert result.summary.rows_processed == len(frame)
    assert result.summary.score_column == "zscore"
    assert any("is_anomaly" in record for record in result.records)
