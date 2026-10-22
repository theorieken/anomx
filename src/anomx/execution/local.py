"""Inspectable local experiments with chronological evaluation and Matplotlib.

This module has no platform, job-worker, Darts, or database dependency. Models
remain responsible for fitting and inference; the executor records the split,
metrics, timings, and outputs of a local experiment.
"""

from __future__ import annotations

# ruff: noqa: ANN401 - execution accepts user-defined component contracts
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd

from anomx.base._arrays import as_array, restore_output
from anomx.console import operation
from anomx.data.timeseries import TimeSeries


def _frame(data: Any) -> pd.DataFrame:  # noqa: ANN401
    """Normalize display/export without changing model inputs."""
    if isinstance(data, TimeSeries):
        return cast("pd.DataFrame", data.to_dataframe())
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, pd.Series):
        return data.to_frame()
    return pd.DataFrame(data)


@dataclass
class LocalResult:
    """One experiment's observations, predictions, metrics, and provenance.

    ``prediction`` is a forecast, reconstruction, embedding, or score depending
    on ``family``. Representation outputs need not share observation features;
    the executor only calculates default error metrics for forecasting and
    reconstruction. ``plot()`` returns ordinary Matplotlib axes and ``save()``
    writes portable CSV/JSON, never executable model pickle files.
    """

    name: str
    family: str
    train: Any
    test: Any
    prediction: Any
    metrics: dict[str, float] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def plot(
        self, *, ax: Any = None, history: int = 100, max_components: int = 4, **kwargs: Any
    ) -> Any:  # noqa: ANN401
        """Plot a forecast/reconstruction against observations, or learned outputs.

        ``history`` limits the training context shown before the held-out data.
        Pass ``ax`` to compose with existing Matplotlib figures. For multivariate
        output each component is rendered separately with a distinct label.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
        if max_components < 1:
            raise ValueError("max_components must be positive.")
        if self.family in {"forecasting", "reconstruction"}:
            if history > 0:
                training = _frame(self.train).iloc[:, :max_components].tail(history)
                training.rename(columns=lambda name: f"train · {name}").plot(
                    ax=ax,
                    **{"alpha": 0.5, **kwargs},
                )
            actual = _frame(self.test).iloc[:, :max_components]
            actual.rename(columns=lambda name: f"actual · {name}").plot(ax=ax, **kwargs)
        predicted = _frame(self.prediction).iloc[:, :max_components]
        predicted.rename(columns=lambda name: f"{self.family} · {name}").plot(
            ax=ax,
            **{"linestyle": "--", **kwargs},
        )
        component_count = len(_frame(self.prediction).columns)
        legend_title = (
            f"Showing {max_components} of {component_count} components"
            if component_count > max_components
            else None
        )
        ax.legend(fontsize="x-small", ncol=3, title=legend_title, title_fontsize="x-small")
        ax.set_title(
            self.name
            + (
                " · " + ", ".join(f"{k}={v:.3g}" for k, v in self.metrics.items())
                if self.metrics
                else ""
            )
        )
        ax.set_xlabel("time / observation")
        return ax

    def save(self, directory: str | Path, *, plot: bool = False) -> Path:
        """Save a receipt and tables to a dedicated local directory.

        Existing filenames in this directory are replaced deliberately. Model
        objects and training data are not serialized; receipts include test and
        prediction data plus the split metadata needed to inspect the run.
        """
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        _frame(self.test).to_csv(destination / "observations.csv")
        _frame(self.prediction).to_csv(destination / "predictions.csv")
        receipt = {
            "name": self.name,
            "family": self.family,
            "metrics": self.metrics,
            "timings": self.timings,
            "metadata": self.metadata,
        }
        (destination / "run.json").write_text(
            json.dumps(receipt, indent=2, default=str, allow_nan=False) + "\n"
        )
        if plot:
            import matplotlib.pyplot as plt

            axes = self.plot()
            axes.figure.savefig(destination / "result.png", dpi=150)
            plt.close(axes.figure)
        return destination


class LocalExecutor:
    """Run and inspect local models with a deliberately small execution contract.

    :meth:`forecast` fits a model on a chronological training prefix and predicts
    the untouched suffix. :meth:`fit_predict` supports reconstruction and
    representation models on explicitly supplied train/test arrays or tables.
    Transformers fit on training data only. No platform service is required.
    """

    def __init__(self, *, verbose: bool = True) -> None:
        self.verbose = verbose

    def forecast(
        self,
        model: Any,
        series: TimeSeries,
        *,
        horizon: int | None = None,
        train_fraction: float = 0.8,
        transformer: Any = None,
        metrics: Mapping[str, Callable[..., Any]] | None = None,
        name: str | None = None,
    ) -> LocalResult:  # noqa: ANN401
        """Fit and evaluate a deterministic model on a chronological holdout.

        ``model`` implements ``fit(TimeSeries)`` and ``predict(n)``. ``horizon``
        selects the test suffix; otherwise ``train_fraction`` chooses the split.
        Optional transformers implement fit_transform/transform/inverse_transform;
        metrics and saved predictions use the original data scale.
        """
        from anomx.metrics import mae, rmse

        if not isinstance(series, TimeSeries) or not series.is_deterministic:
            raise TypeError("Local forecasting requires a deterministic Anomx TimeSeries.")
        if not math.isfinite(train_fraction) or not 0 < train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1.")
        if horizon is None:
            horizon = len(series) - int(len(series) * train_fraction)
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or not 0 < horizon < len(series)
        ):
            raise ValueError("horizon must leave nonempty training and test sets.")
        train, test = series[:-horizon], series[-horizon:]
        if not np.isfinite(series.all_values(copy=False)).all():
            raise ValueError(
                "Forecast data contains NaN/inf; select a complete interval "
                "or explicitly fill gaps."
            )
        label = name or type(model).__name__
        start = perf_counter()
        with operation(f"{label} · fit {len(train):,} observations", enabled=self.verbose):
            training = transformer.fit_transform(train) if transformer is not None else train
            model.fit(training)
        fit_seconds = perf_counter() - start
        start = perf_counter()
        with operation(f"{label} · forecast {len(test):,} observations", enabled=self.verbose):
            prediction = model.predict(len(test))
            if transformer is not None:
                prediction = transformer.inverse_transform(prediction)
        if not isinstance(prediction, TimeSeries) or not prediction.time_index.equals(
            test.time_index
        ):
            raise ValueError("Forecast must cover the complete held-out time index.")
        if not prediction.is_deterministic or not prediction.components.equals(test.components):
            raise ValueError("Forecast must preserve ordered components and be deterministic.")
        if not np.isfinite(prediction.all_values(copy=False)).all():
            raise ValueError("Forecast predictions must be finite.")
        predict_seconds = perf_counter() - start
        measured = {
            key: float(fn(test, prediction))
            for key, fn in (metrics if metrics is not None else {"mae": mae, "rmse": rmse}).items()
        }
        if not all(math.isfinite(value) for value in measured.values()):
            raise ValueError("Evaluation metrics must be finite.")
        return LocalResult(
            label,
            "forecasting",
            train,
            test,
            prediction,
            measured,
            {"fit_seconds": fit_seconds, "predict_seconds": predict_seconds},
            {
                "model": f"{type(model).__module__}.{type(model).__name__}",
                "train_rows": len(train),
                "test_rows": len(test),
                "train_end": str(train.end_time()),
                "test_start": str(test.start_time()),
                "series": series.metadata,
                "transformer": type(transformer).__name__ if transformer is not None else None,
            },
        )

    def fit_predict(
        self,
        model: Any,
        train: Any,
        test: Any,
        *,
        family: str = "reconstruction",
        name: str | None = None,
        metrics: Mapping[str, Callable[..., Any]] | None = None,
    ) -> LocalResult:  # noqa: ANN401
        """Fit an array/table model and inspect held-out reconstructions or representations.

        GenericModel can adapt sklearn estimators, callables, and PyTorch modules.
        For representation models supply metrics explicitly if desired; embedding
        coordinates are not compared to observation values as reconstruction errors.
        """
        if family not in {"reconstruction", "representation"}:
            raise ValueError(
                "family must be reconstruction or representation; "
                "use forecast for temporal forecasts."
            )
        if len(train) == 0 or len(test) == 0:
            raise ValueError("Training and test data must both be nonempty.")
        label = name or type(model).__name__
        start = perf_counter()
        with operation(f"{label} · fit {len(train):,} observations", enabled=self.verbose):
            model.fit(train)
        fit_seconds = perf_counter() - start
        start = perf_counter()
        with operation(f"{label} · {family} {len(test):,} observations", enabled=self.verbose):
            prediction = model.predict(test)
        predict_seconds = perf_counter() - start
        if isinstance(prediction, (pd.DataFrame, pd.Series, TimeSeries)):
            expected_index = (
                test.time_index if isinstance(test, TimeSeries) else getattr(test, "index", None)
            )
            output_index = (
                prediction.time_index if isinstance(prediction, TimeSeries) else prediction.index
            )
            if expected_index is not None and not output_index.equals(expected_index):
                raise ValueError("Prediction index must match test observations in order.")
            if (
                family == "reconstruction"
                and isinstance(test, (TimeSeries, pd.DataFrame))
                and not _frame(prediction).columns.equals(_frame(test).columns)
            ):
                raise ValueError("Reconstruction components must match test columns in order.")
        else:
            prediction = restore_output(test, prediction, name="prediction")
        actual_values = as_array(test)
        predicted_values = as_array(prediction, name="prediction")
        if len(actual_values) != len(predicted_values):
            raise ValueError("Prediction must return one output per test observation.")
        if family == "reconstruction" and actual_values.shape != predicted_values.shape:
            raise ValueError("Reconstruction output must have the observation shape.")
        measured = {}
        if metrics is not None:
            measured = {key: float(fn(test, prediction)) for key, fn in metrics.items()}
        elif family == "reconstruction":
            if actual_values.shape != predicted_values.shape:
                raise ValueError("Reconstruction output must have the observation shape.")
            difference = actual_values - predicted_values
            measured = {
                "mae": float(np.mean(np.abs(difference))),
                "rmse": float(np.sqrt(np.mean(difference**2))),
            }
        if not all(math.isfinite(value) for value in measured.values()):
            raise ValueError("Evaluation metrics must be finite.")
        return LocalResult(
            label,
            family,
            train,
            test,
            prediction,
            measured,
            {"fit_seconds": fit_seconds, "predict_seconds": predict_seconds},
            {
                "model": f"{type(model).__module__}.{type(model).__name__}",
                "train_rows": len(train),
                "test_rows": len(test),
            },
        )
