"""Default offline train-score-detect orchestration."""

from __future__ import annotations

from time import perf_counter
from typing import Any, cast
from uuid import uuid4

import pandas as pd

from anomx._shared import dataframe_to_records, ensure_dataframe
from anomx.components.algorithms.base import BaseAlgorithm
from anomx.components.algorithms.contracts import JobResult, JobSpec, JobSummary
from anomx.components.algorithms.offline.catalog import resolve_implementation


class PipelineOrchestrator(BaseAlgorithm):
    """Train, score, and classify a dataset with built-in library components."""

    component_key = "pipeline"
    component_name = "Pipeline"
    component_description = "Default offline orchestration pipeline for model, scorer, and detector components."
    component_config_schema = {
        "connector": {"type": "object"},
        "detector": {"type": "object"},
        "model": {"type": "object"},
        "scorer": {"type": "object"},
    }

    def run_job(self, job_spec: JobSpec | dict[str, Any]) -> JobResult:
        spec = job_spec if isinstance(job_spec, JobSpec) else JobSpec.from_mapping(job_spec)
        started_at = perf_counter()
        config = spec.config

        connector = cast(type, resolve_implementation(spec.connector, "connector"))()
        model = cast(type, resolve_implementation(spec.model, "model"))(config.get("model", {}))
        scorer = cast(type, resolve_implementation(spec.scorer, "scorer"))(config.get("scorer", {}))
        detector = cast(type, resolve_implementation(spec.detector, "detector"))(config.get("detector", {}))

        raw_data = connector.read(config.get("connector", {}))
        frame = ensure_dataframe(raw_data)
        model.fit(frame)
        predictions = ensure_dataframe(model.predict(frame))
        scored = ensure_dataframe(scorer.score(predictions))
        detected = ensure_dataframe(detector.detect(scored))

        feature_columns = list(config.get("model", {}).get("feature_columns") or [])
        if not feature_columns:
            feature_columns = frame.select_dtypes(include=["number"]).columns.tolist()

        anomaly_series = detected.get("is_anomaly", pd.Series(dtype=bool))
        score_column = "zscore" if "zscore" in detected.columns else "model_score"
        anomaly_count = int(anomaly_series.sum())
        duration_ms = int((perf_counter() - started_at) * 1000)

        return JobResult(
            job_id=str(uuid4()),
            status="completed",
            connector=spec.connector,
            model=spec.model,
            scorer=spec.scorer,
            detector=spec.detector,
            summary=JobSummary(
                rows_processed=len(detected),
                anomaly_count=anomaly_count,
                feature_columns=feature_columns,
                score_column=score_column,
                duration_ms=duration_ms,
            ),
            records=dataframe_to_records(detected),
        )


PipelineAlgorithm = PipelineOrchestrator
