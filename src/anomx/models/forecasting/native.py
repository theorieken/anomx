"""Native, inspectable forecasting baselines operating on Anomx TimeSeries.

These implementations do not import Darts. Their fit/predict convention makes
chronological experiments easy to read while keeping statistical assumptions
explicit: inputs are regular, deterministic, finite series; forecasts begin
immediately after the supplied history; no scaling or imputation is implicit.
"""

from __future__ import annotations

import copy
import math
import pickle
from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

import numpy as np
import pandas as pd

from anomx.base.capabilities import ComponentCapabilities, Predictive
from anomx.data.timeseries import TimeSeries
from anomx.models.base.normality import NormalityModel


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _series(data: object, minimum: int) -> TimeSeries:
    if not isinstance(data, TimeSeries):
        converter = getattr(data, "to_timeseries", None)
        if converter is None:
            raise TypeError("Expected an Anomx TimeSeries or a dataset with to_timeseries().")
        data = converter()
    if not isinstance(data, TimeSeries) or not data.is_deterministic:
        raise ValueError("Forecasting inputs must be deterministic TimeSeries.")
    if len(data) < minimum or not np.isfinite(data.values(copy=False)).all():
        raise ValueError(f"At least {minimum} finite observations are required.")
    return data


def _forecast(source: TimeSeries, values: np.ndarray) -> TimeSeries:
    if not np.isfinite(values).all():
        raise ValueError("Forecast produced nonfinite values.")
    n = len(values)
    if source.has_datetime_index:
        index = pd.date_range(source.end_time() + source.freq, periods=n, freq=source.freq)
    else:
        start, step = int(source.end_time()) + int(source.freq), int(source.freq)
        index = pd.RangeIndex(start, start + n * step, step)
    return TimeSeries.from_times_and_values(
        index,
        values,
        columns=source.components,
        static_covariates=source.static_covariates,
        hierarchy=source.hierarchy,
        metadata=source.metadata,
    )


class ForecastingModel(NormalityModel, Predictive):
    """Base for deterministic local forecasts with explicit horizons.

    Fit returns the model; ``predict(n, series=...)`` returns the next ``n``
    time steps. A supplied series changes the prediction context without
    changing fitted parameters. Subclasses declare ``min_train_series_length``
    and implement ``_fit_values`` and ``_predict_values``.
    """

    min_train_series_length = 1
    min_predict_series_length = 1
    preferred_view = "timeseries"
    capabilities = ComponentCapabilities(
        supports_unordered_samples=False,
        supports_sequences=True,
        supports_irregular_sampling=False,
        requires_regular_sampling=True,
    )

    def __init__(self) -> None:
        self.training_series: TimeSeries | None = None

    def fit(self, series: TimeSeries, **kwargs: object) -> ForecastingModel:
        """Fit one series, recording the exact training time span and components."""
        if kwargs:
            raise TypeError(f"Unsupported fit arguments: {', '.join(kwargs)}")
        data = _series(series, self.min_train_series_length)
        self.training_series = None
        self._fit_values(data.values(copy=False))
        self.training_series = data.copy()
        return self

    @abstractmethod
    def _fit_values(self, values: np.ndarray) -> None:
        """Fit model state from a finite time-by-component matrix."""

    @abstractmethod
    def _predict_values(self, n: int, history: np.ndarray) -> np.ndarray:
        """Forecast a finite time-by-component matrix of length ``n``."""

    def predict(
        self, n: int, series: TimeSeries | None = None, *, num_samples: int = 1
    ) -> TimeSeries:
        """Forecast after the context's end; no test observations are consumed."""
        _positive_int(n, "n")
        if self.training_series is None:
            raise RuntimeError("Fit the forecasting model before calling predict().")
        if num_samples != 1:
            raise ValueError("This baseline is deterministic and supports num_samples=1.")
        source = _series(
            series if series is not None else self.training_series, self.min_predict_series_length
        )
        if not source.components.equals(self.training_series.components):
            raise ValueError("Prediction components must match the ordered training components.")
        if source.freq != self.training_series.freq:
            raise ValueError("Prediction frequency must match the training frequency.")
        return _forecast(source, self._predict_values(n, source.values(copy=False)))

    def forecast(self, data: TimeSeries, horizon: int) -> TimeSeries:
        """Implement the predictive capability using an explicit history and horizon."""
        return self.predict(horizon, series=data)

    def historical_forecasts(
        self,
        series: TimeSeries,
        *,
        start: int | float = 0.5,
        forecast_horizon: int = 1,
        stride: int = 1,
        retrain: bool = True,
        last_points_only: bool = True,
        verbose: bool = False,
    ) -> TimeSeries | list[TimeSeries]:
        """Walk forward without training on validation/future observations.

        ``start`` is the first forecast origin (an integer position or fraction
        of series length). Only complete forecasts within ``series`` are made.
        Retraining uses a fresh model copy on each expanding history; with
        ``retrain=False`` the existing fitted parameters are reused.
        """
        data = _series(series, self.min_train_series_length + 1)
        _positive_int(forecast_horizon, "forecast_horizon")
        _positive_int(stride, "stride")
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            raise ValueError("start must be an integer position or fraction in (0, 1).")
        if isinstance(start, float):
            if not 0 < start < 1:
                raise ValueError("A fractional start must be in (0, 1).")
            start = int(len(data) * start)
        if start < self.min_train_series_length or start + forecast_horizon > len(data):
            raise ValueError("start must allow sufficient training history and a full forecast.")
        from anomx.console import progress_bar

        forecasts = []
        origins = progress_bar(
            "historical forecasts",
            enabled=verbose,
            iterable=range(start, len(data) - forecast_horizon + 1, stride),
            unit="forecast",
        )
        for origin in origins:
            context = data[:origin]
            model = copy.deepcopy(self).fit(context) if retrain else self
            forecasts.append(model.predict(forecast_horizon, series=context))
        if not last_points_only:
            return forecasts
        times = pd.Index([item.end_time() for item in forecasts])
        if isinstance(data.time_index, pd.DatetimeIndex):
            times = pd.DatetimeIndex(times, freq=data.freq * stride)
        else:
            times = pd.RangeIndex(
                int(times[0]), int(times[-1]) + int(data.freq) * stride, int(data.freq) * stride
            )
        return TimeSeries.from_times_and_values(
            times,
            np.stack([item.values(copy=False)[-1] for item in forecasts]),
            columns=data.components,
        )

    def backtest(
        self,
        series: TimeSeries,
        *,
        metric: Callable[[TimeSeries, TimeSeries], float] | None = None,
        **kwargs: object,
    ) -> float:
        """Evaluate historical forecasts with a metric (mean absolute error by default)."""
        if metric is None:
            from anomx.metrics import mae

            metric = mae
        forecasts = self.historical_forecasts(series, **kwargs)
        if isinstance(forecasts, list):
            return float(np.mean([metric(series, forecast) for forecast in forecasts]))
        return metric(series, forecasts)

    def save(self, path: str) -> None:
        """Save fitted state with pickle; only load artifacts from trusted sources."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self, handle)

    def load(self, path: str) -> ForecastingModel:
        """Restore a trusted artifact of this same concrete model class."""
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, type(self)):
            raise TypeError("Saved model must have the same concrete class.")
        self.__dict__.update(model.__dict__)
        return self


class NaiveMean(ForecastingModel):
    """Repeat the fitted per-component mean for every forecast step."""

    component_key = "naive_mean"

    def _fit_values(self, values: np.ndarray) -> None:
        self.mean_ = values.mean(axis=0)

    def _predict_values(self, n: int, history: np.ndarray) -> np.ndarray:
        return np.repeat(self.mean_[None], n, axis=0)


class NaiveSeasonal(ForecastingModel):
    """Repeat the last ``K`` observations; ``K=1`` is persistence."""

    component_key = "naive_seasonal"

    def __init__(self, K: int = 1) -> None:
        super().__init__()
        self.K = _positive_int(K, "K")
        self.min_train_series_length = self.K
        self.min_predict_series_length = self.K

    def _fit_values(self, values: np.ndarray) -> None:
        pass

    def _predict_values(self, n: int, history: np.ndarray) -> np.ndarray:
        return history[-self.K :][np.arange(n) % self.K]


class NaiveDrift(ForecastingModel):
    """Extrapolate the line joining the first and last context observation."""

    component_key = "naive_drift"
    min_train_series_length = 2
    min_predict_series_length = 2

    def _fit_values(self, values: np.ndarray) -> None:
        pass

    def _predict_values(self, n: int, history: np.ndarray) -> np.ndarray:
        slope = (history[-1] - history[0]) / (len(history) - 1)
        return history[-1] + np.arange(1, n + 1)[:, None] * slope


class LinearRegressionModel(ForecastingModel):
    """Fit an autoregression with jointly modeled components and ridge regularization.

    A positive integer ``lags`` uses the preceding ``lags`` observations. A
    sequence of distinct negative offsets selects specific history positions.
    ``ridge=0`` uses ordinary least squares; the intercept is never penalized.
    Prediction recursively appends each one-step forecast to the context.
    Covariates and probabilistic forecasts are deliberately not implicit.
    """

    component_key = "linear_regression"

    def __init__(
        self, lags: int | list[int] = 12, *, ridge: float = 0.0, fit_intercept: bool = True
    ) -> None:
        super().__init__()
        if isinstance(lags, int) and not isinstance(lags, bool):
            self.lags = list(range(-_positive_int(lags, "lags"), 0))
        else:
            self.lags = list(lags) if isinstance(lags, (list, tuple)) else []
            if (
                not self.lags
                or any(
                    isinstance(lag, bool) or not isinstance(lag, int) or lag >= 0
                    for lag in self.lags
                )
                or len(set(self.lags)) != len(self.lags)
            ):
                raise ValueError(
                    "lags must be a positive count or distinct negative integer offsets."
                )
            self.lags.sort()
        if not math.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and nonnegative.")
        self.ridge, self.fit_intercept = ridge, fit_intercept
        self.min_train_series_length = -self.lags[0] + 1
        self.min_predict_series_length = -self.lags[0]

    def _fit_values(self, values: np.ndarray) -> None:
        positions = np.arange(-self.lags[0], len(values))
        x = values[positions[:, None] + np.array(self.lags)].reshape(len(positions), -1)
        if self.fit_intercept:
            x = np.column_stack([x, np.ones(len(x))])
        y = values[positions]
        if self.ridge:
            penalty = np.eye(x.shape[1]) * np.sqrt(self.ridge)
            if self.fit_intercept:
                penalty[-1, -1] = 0
            x = np.concatenate([x, penalty])
            y = np.concatenate([y, np.zeros((len(penalty), y.shape[1]))])
        self.coef_ = np.linalg.lstsq(x, y, rcond=None)[0]

    def _predict_values(self, n: int, history: np.ndarray) -> np.ndarray:
        values = list(history)
        forecast = []
        for _ in range(n):
            x = np.concatenate([values[lag] for lag in self.lags])
            if self.fit_intercept:
                x = np.append(x, 1.0)
            value = x @ self.coef_
            values.append(value)
            forecast.append(value)
        return np.asarray(forecast)


class TorchForecastingModel(ForecastingModel):
    """Train any PyTorch module on chronological time-series windows.

    The module receives ``(batch, input_chunk_length, components)`` and must
    return ``(batch, output_chunk_length, components)``. If omitted, a small
    linear module is constructed on first fit. The model uses GenericModel's
    training adapter and recursively forecasts in output-sized chunks.
    """

    component_key = "torch_forecasting"

    def __init__(
        self,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        module: Module | None = None,
        *,
        n_epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        random_state: int = 42,
        device: str = "cpu",
        loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.input_chunk_length = _positive_int(input_chunk_length, "input_chunk_length")
        self.output_chunk_length = _positive_int(output_chunk_length, "output_chunk_length")
        self.min_train_series_length = input_chunk_length + output_chunk_length
        self.min_predict_series_length = input_chunk_length
        self.module = module
        self._module_template = copy.deepcopy(module)
        self.n_epochs, self.batch_size = n_epochs, batch_size
        self.learning_rate, self.random_state = learning_rate, random_state
        self.device, self.loss_fn, self.verbose = device, loss_fn, verbose
        self.training_history: list[dict[str, float]] = []

    def _fit_values(self, values: np.ndarray) -> None:
        import torch

        from anomx.models.generic import GenericModel

        width = values.shape[1]
        # A new fit is independent of previous training, including walk-forward refits.
        self.module = copy.deepcopy(self._module_template)
        if self.module is None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.random_state)
                self.module = torch.nn.Sequential(
                    torch.nn.Flatten(start_dim=1),
                    torch.nn.Linear(
                        self.input_chunk_length * width, self.output_chunk_length * width
                    ),
                    torch.nn.Unflatten(1, (self.output_chunk_length, width)),
                )
        starts = range(len(values) - self.min_train_series_length + 1)
        x = np.stack([values[i : i + self.input_chunk_length] for i in starts])
        y = np.stack(
            [values[i + self.input_chunk_length : i + self.min_train_series_length] for i in starts]
        )
        self.adapter_ = GenericModel(
            self.module,
            family="forecasting",
            epochs=self.n_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            device=self.device,
            loss_fn=self.loss_fn,
            random_state=self.random_state,
        ).fit(x, y, verbose=self.verbose)
        self.training_history = self.adapter_.training_history

    def _predict_values(self, n: int, history: np.ndarray) -> np.ndarray:
        result = []
        while sum(len(chunk) for chunk in result) < n:
            output = self.adapter_.predict(history[-self.input_chunk_length :][None])[0]
            expected = (self.output_chunk_length, history.shape[1])
            if output.shape != expected:
                raise ValueError(f"Module output must have shape {expected} for each batch item.")
            chunk = output[: n - sum(len(chunk) for chunk in result)]
            result.append(chunk)
            history = np.concatenate([history, chunk])
        return np.concatenate(result)
