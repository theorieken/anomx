"""Native forecast comparison for one verified, regular numeric time-series segment.

Copy this file to the shared job workspace as training.py and compute-work.py as
main.py. Required inputs: dataset_reference, target_columns, start, end.
Optional: frequency (server-side aggregation), aggregation, max_rows, lags,
ridge, horizon, minimum_improvement, architecture=linear/lstm. For LSTM, also
copy lstm-model.py as lstm_model.py and configure epochs/device on real compute.
This produces research artifacts; it does
not export ONNX, publish a Model, enable a schedule, or create findings.
"""

from __future__ import annotations

import importlib.metadata
import json
import pickle
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anomx import Dataset, TimeSeries
from anomx.data.processing.transformers import Scaler
from anomx.detection import NormScorer
from anomx.execution.work import WorkContext
from anomx.metrics import mae, rmse
from anomx.models import ForecastingModel, LinearRegressionModel, NaiveSeasonal


def run(inputs: dict[str, Any], work: WorkContext) -> dict[str, Any]:
    """Fit on training, select on validation, and evaluate one held-out range."""
    for name in ("dataset_reference", "start", "end"):
        if not isinstance(inputs.get(name), str) or not inputs[name].strip():
            raise ValueError(f"{name} must be an explicit nonempty string.")
    architecture = inputs.get("architecture", "linear")
    if architecture not in {"linear", "lstm"}:
        raise ValueError("Use linear or lstm, or adapt the native model factory explicitly.")
    columns = inputs["target_columns"]
    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(column, str) for column in columns)
        or len(set(columns)) != len(columns)
    ):
        raise ValueError("target_columns must name distinct verified numeric columns.")
    max_rows = inputs.get("max_rows", 20000)
    horizon = inputs.get("horizon", 1)
    lags = inputs.get("lags", [12])
    if not isinstance(lags, list) or not lags or len(lags) > 12:
        raise ValueError("lags must be a bounded list of candidate positive lag counts.")
    for name, values in (("max_rows", [max_rows]), ("horizon", [horizon]), ("lags", lags)):
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values
        ):
            raise ValueError(f"{name} must contain positive integers.")
    improvement = float(inputs.get("minimum_improvement", 0.05))
    if not 0 <= improvement < 1:
        raise ValueError("minimum_improvement must be in [0, 1).")

    # Selection/aggregation precede transfer. Do not download all history first.
    dataset = Dataset(inputs["dataset_reference"], progress=False).select(
        start=inputs["start"], end=inputs["end"]
    )
    if inputs.get("frequency"):
        dataset = dataset.downsample(
            inputs["frequency"], aggregation=inputs.get("aggregation", "mean")
        )
    frame = dataset.to_dataframe(max_rows=max_rows + 1, progress=False)
    definition = dataset.describe()
    if len(frame) > max_rows:
        raise ValueError(
            "Selected data exceeds max_rows; narrow the range or aggregate explicitly."
        )
    if dataset.kind != "time_series" or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("This template requires verified time_series data with timestamps.")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and increasing; resolve ordering explicitly.")
    selected = frame[columns]
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in selected.dtypes):
        raise ValueError(
            "Select scalar numeric columns; tensor reduction needs a documented adapter."
        )
    if not np.isfinite(selected.to_numpy(dtype=float)).all():
        raise ValueError("Missing/nonfinite observations need an explicit preprocessing policy.")
    if len(frame) < 3 or pd.infer_freq(frame.index) is None:
        raise ValueError("Data has gaps or irregular sampling; use separate regular segments.")
    # Convert the already bounded frame without a second remote read or hidden fill.
    series = Dataset.from_dataframe(selected, kind="time_series").to_timeseries(progress=False)
    train_end, validation_end = len(series) * 70 // 100, len(series) * 85 // 100
    if (
        train_end < max(lags) + horizon
        or min(validation_end - train_end, len(series) - validation_end) < horizon + 9
    ):
        raise ValueError("Splits need enough training history and at least ten evaluation origins.")
    scaler = Scaler().fit(series[:train_end])
    scaled = scaler.transform(series)
    work.log(
        "Loaded bounded dataset on compute", rows=len(series), version=definition.get("version")
    )

    def forecast(model: ForecastingModel, start: int, stop: int) -> TimeSeries:
        predicted = model.historical_forecasts(
            scaled[:stop],
            start=start,
            forecast_horizon=horizon,
            retrain=False,
            last_points_only=True,
            verbose=False,
        )
        return scaler.inverse_transform(predicted)

    baseline = NaiveSeasonal(K=1).fit(scaled[:train_end])
    validation_baseline = forecast(baseline, train_end, validation_end)
    validation_baseline_rmse = float(rmse(series, validation_baseline))
    candidates = []
    fitted = []
    for step, lag in enumerate(lags, start=1):
        if architecture == "lstm":
            from lstm_model import create_model

            model = create_model(inputs, lag, horizon, len(columns))
        else:
            model = LinearRegressionModel(lags=lag, ridge=float(inputs.get("ridge", 0.01)))
        model.fit(scaled[:train_end])
        for point in getattr(model, "training_history", []):
            work.metric(f"candidate_{step}_train_loss", point["loss"], step=int(point["epoch"]))
        value = float(rmse(series, forecast(model, train_end, validation_end)))
        work.metric("candidate_validation_rmse", value, step=step)
        candidates.append({"lags": lag, "validation_rmse": value})
        fitted.append(model)
    selected_index = min(
        range(len(candidates)), key=lambda index: candidates[index]["validation_rmse"]
    )
    model = fitted[selected_index]
    # Selection is frozen before any test forecast or test metric is evaluated.
    predicted = forecast(model, validation_end, len(series))
    baseline_predicted = forecast(baseline, validation_end, len(series))
    actual = series.slice_intersect(predicted)
    scores = NormScorer(component_wise=True).score_from_prediction(actual, predicted)
    validation_rmse = candidates[selected_index]["validation_rmse"]
    metrics = {
        "validation_rmse": validation_rmse,
        "validation_persistence_rmse": validation_baseline_rmse,
        "test_rmse": float(rmse(actual, predicted)),
        "test_mae": float(mae(actual, predicted)),
        "persistence_rmse": float(rmse(actual, baseline_predicted)),
        "test_forecasts": len(predicted),
        "quality_gate_passed": float(
            validation_rmse < validation_baseline_rmse * (1 - improvement)
        ),
    }
    for name, value in metrics.items():
        work.metric(name, float(value))
    procedure = {
        "dataset_reference": inputs["dataset_reference"],
        "dataset_version": definition.get("version"),
        "selection": frame.attrs.get("anomx_selection", definition.get("selection", {})),
        "cache": dataset.cache_info,
        "inputs": dict(inputs),
        "target_columns": columns,
        "rows": len(series),
        "observed_time": {
            "start_utc": str(series.start_time()),
            "last_utc": str(series.end_time()),
            "frequency": str(series.freq),
        },
        "split_positions": [0, train_end, validation_end, len(series)],
        "normalization": "Scaler fitted on training only; metrics restored to original units",
        "evaluation": "rolling observed history; fixed fitted weights; final horizon point",
        "architecture": architecture,
        "checkpoint_policy": "fixed epoch budget per candidate; validation selects candidate",
        "candidates": candidates,
        "selected": candidates[selected_index],
        "metrics": metrics,
        "runtime": {
            "python": platform.python_version(),
            "anomx": importlib.metadata.version("anomx"),
        },
        "publication": "research artifacts only; no registered Model or operational threshold",
    }
    Path("procedure.json").write_text(
        json.dumps(procedure, indent=2, allow_nan=False), encoding="utf-8"
    )
    prediction_frame = pd.DataFrame(index=actual.time_index)
    for index, column in enumerate(columns):
        prediction_frame[f"{column}:actual"] = actual.values()[:, index]
        prediction_frame[f"{column}:prediction"] = predicted.values()[:, index]
        prediction_frame[f"{column}:absolute_error"] = scores.values()[:, index]
    prediction_frame.to_csv("test-predictions.csv", index_label="timestamp_utc")
    model.save("forecast.pkl")
    # These are locally generated trusted replay artifacts, not portable model publications.
    with Path("scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)
    artifacts = [
        work.artifact(path)
        for path in ("procedure.json", "test-predictions.csv", "forecast.pkl", "scaler.pkl")
    ]
    work.log(
        "Native forecast comparison completed", selected_lags=candidates[selected_index]["lags"]
    )
    return {
        "metrics": metrics,
        "dataset_version": definition.get("version"),
        "selected": candidates[selected_index],
        "artifacts": artifacts,
        "preview": prediction_frame.head(5)
        .reset_index(names="timestamp")
        .astype({"timestamp": str})
        .to_dict("records"),
    }
