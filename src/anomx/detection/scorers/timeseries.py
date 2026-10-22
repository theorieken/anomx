"""Native time-series residual, distribution, and learned anomaly scorers.

Scores preserve the time axis and component names. Window scorers learn only
from explicit training data. Their default trailing windows are causal; setting
``window_agg=True`` averages all windows covering a point and is therefore an
offline operation that can use later observations.
"""

from __future__ import annotations

import copy
from abc import abstractmethod
from collections.abc import Sequence
from typing import Protocol, Self

import numpy as np

from anomx.base.capabilities import ComponentCapabilities
from anomx.data.timeseries import TimeSeries
from anomx.detection.base.scorers import Scorer

SeriesLike = TimeSeries | Sequence[TimeSeries]


class OutlierEstimator(Protocol):
    """Structural contract accepted by the optional PyOD adapter."""

    def fit(self, values: np.ndarray) -> Self: ...
    def decision_function(self, values: np.ndarray) -> np.ndarray: ...


def _inputs(series: TimeSeries | Sequence[TimeSeries]) -> list[TimeSeries]:
    items = [series] if isinstance(series, TimeSeries) else list(series)
    if not items or any(not isinstance(item, TimeSeries) for item in items):
        raise ValueError("Expected a TimeSeries or a nonempty sequence of TimeSeries.")
    return items


def _values(series: TimeSeries) -> np.ndarray:
    if not isinstance(series, TimeSeries) or not series.is_deterministic:
        raise ValueError("Expected a deterministic Anomx TimeSeries.")
    values = series.values(copy=False)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Scorer inputs must be nonempty and finite.")
    return values


def _aligned(actual: TimeSeries, predicted: TimeSeries) -> tuple[TimeSeries, TimeSeries]:
    if not isinstance(actual, TimeSeries) or not isinstance(predicted, TimeSeries):
        raise TypeError("Actual and predicted values must be TimeSeries.")
    if not actual.components.equals(predicted.components):
        raise ValueError("Actual and predicted component names must match in order.")
    actual, predicted = actual.slice_intersect(predicted), predicted.slice_intersect(actual)
    if not len(actual) or not actual.time_index.equals(predicted.time_index):
        raise ValueError("Actual and predicted series must share aligned timestamps.")
    return actual, predicted


def _result(source: TimeSeries, values: np.ndarray, *, offset: int = 0) -> TimeSeries:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if not np.isfinite(values).all():
        raise ValueError("Scorer produced nonfinite values.")
    same_width = values.shape[1] == source.n_components
    return TimeSeries.from_times_and_values(
        source.time_index[offset:],
        values,
        columns=source.components if same_width else ["score"],
        static_covariates=source.static_covariates if same_width else None,
        hierarchy=source.hierarchy if same_width else None,
        metadata=source.metadata,
    )


class AnomalyScorer(Scorer):
    """TimeSeries scorer contract, distinct from legacy scored-table components.

    ``score`` consumes observations or residuals. ``score_from_prediction``
    aligns actual and predicted timestamps, then scores their absolute
    residuals unless a concrete scorer specifies another transformation.
    All methods accept one series or equal-length sequences of series.
    """

    is_trainable = False
    capabilities = ComponentCapabilities(
        supports_sequences=True, supports_unordered_samples=False, requires_training=False
    )

    @abstractmethod
    def _score(self, series: TimeSeries) -> TimeSeries:
        """Score a single series."""

    def score(self, series: TimeSeries | Sequence[TimeSeries]) -> TimeSeries | list[TimeSeries]:
        results = [self._score(item) for item in _inputs(series)]
        return results[0] if isinstance(series, TimeSeries) else results

    def _residual(self, actual: TimeSeries, predicted: TimeSeries) -> TimeSeries:
        actual, predicted = _aligned(actual, predicted)
        return _result(actual, np.abs(_values(actual) - _values(predicted)))

    def score_from_prediction(
        self,
        actual_series: TimeSeries | Sequence[TimeSeries],
        pred_series: TimeSeries | Sequence[TimeSeries],
    ) -> TimeSeries | list[TimeSeries]:
        actual, predicted = _inputs(actual_series), _inputs(pred_series)
        if isinstance(actual_series, TimeSeries) != isinstance(pred_series, TimeSeries) or len(
            actual
        ) != len(predicted):
            raise ValueError("Actual and predicted inputs must have the same sequence structure.")
        result = [self._score(self._residual(a, p)) for a, p in zip(actual, predicted, strict=True)]
        return result[0] if isinstance(actual_series, TimeSeries) else result


class DifferenceScorer(AnomalyScorer):
    """Signed per-component difference ``actual - predicted`` (no fitting)."""

    component_key = "difference"
    scorer_family = "residual"

    def _residual(self, actual: TimeSeries, predicted: TimeSeries) -> TimeSeries:
        actual, predicted = _aligned(actual, predicted)
        return _result(actual, _values(actual) - _values(predicted))

    def _score(self, series: TimeSeries) -> TimeSeries:
        return _result(series, _values(series))


class NormScorer(AnomalyScorer):
    """Absolute component residuals or their vector norm at each timestamp.

    ``component_wise=True`` returns one absolute score per component.
    Otherwise ``ord`` chooses the numpy vector norm (default Euclidean).
    """

    component_key = "norm"
    scorer_family = "residual"

    def __init__(self, ord: float = 2, component_wise: bool = False) -> None:
        if ord <= 0 or np.isnan(ord):
            raise ValueError("ord must be positive.")
        self.ord, self.component_wise = ord, component_wise

    def _score(self, series: TimeSeries) -> TimeSeries:
        values = _values(series)
        return _result(
            series,
            np.abs(values) if self.component_wise else np.linalg.norm(values, ord=self.ord, axis=1),
        )


class WindowedAnomalyScorer(AnomalyScorer):
    """Fit and score sliding windows without crossing series boundaries."""

    is_trainable = True
    capabilities = ComponentCapabilities(
        supports_sequences=True, supports_unordered_samples=False, requires_training=True
    )

    def __init__(
        self, window: int = 1, component_wise: bool = False, window_agg: bool = False
    ) -> None:
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("window must be a positive integer.")
        self.window, self.component_wise, self.window_agg = window, component_wise, window_agg
        self._components: list[str] | None = None

    def _windows(self, series: TimeSeries) -> np.ndarray:
        values = _values(series)
        if len(values) < self.window:
            raise ValueError(f"At least {self.window} observations are required for a window.")
        return np.lib.stride_tricks.sliding_window_view(values, self.window, axis=0).transpose(
            0, 2, 1
        )

    def fit(self, series: TimeSeries | Sequence[TimeSeries]) -> WindowedAnomalyScorer:
        """Learn normal windows from one or more independent training series."""
        items = _inputs(series)
        components = list(items[0].components)
        if any(list(item.components) != components for item in items):
            raise ValueError("Training series must have identical ordered components.")
        windows = np.concatenate([self._windows(item) for item in items])
        self._components = None
        self._fit_windows(windows)
        self._components = components
        return self

    def fit_from_prediction(
        self, actual_series: SeriesLike, pred_series: SeriesLike
    ) -> WindowedAnomalyScorer:
        """Learn the distribution of absolute training prediction errors."""
        actual, predicted = _inputs(actual_series), _inputs(pred_series)
        if len(actual) != len(predicted):
            raise ValueError("Actual and predicted sequence counts must match.")
        return self.fit([self._residual(a, p) for a, p in zip(actual, predicted, strict=True)])

    def fit_score(self, series: SeriesLike) -> TimeSeries | list[TimeSeries]:
        """Fit and score the same series (use separate validation data for evaluation)."""
        return self.fit(series).score(series)

    @abstractmethod
    def _fit_windows(self, windows: np.ndarray) -> None:
        """Fit time-window-by-component values."""

    @abstractmethod
    def _score_windows(self, windows: np.ndarray) -> np.ndarray:
        """Return one scalar or component vector for each window."""

    def _score(self, series: TimeSeries) -> TimeSeries:
        if self._components is None:
            raise RuntimeError("Fit the scorer before scoring.")
        if list(series.components) != self._components:
            raise ValueError("Scoring components must match the training components.")
        scores = np.asarray(self._score_windows(self._windows(series)))
        if scores.ndim == 1:
            scores = scores[:, None]
        if not self.window_agg:
            return _result(series, scores, offset=self.window - 1)
        totals, counts = np.zeros((len(series), scores.shape[1])), np.zeros(len(series))
        for shift in range(self.window):
            totals[shift : shift + len(scores)] += scores
            counts[shift : shift + len(scores)] += 1
        return _result(series, totals / counts[:, None])


class KMeansScorer(WindowedAnomalyScorer):
    """Euclidean distance to the closest fitted normal-window cluster center."""

    component_key = "kmeans"
    scorer_family = "representation"

    def __init__(
        self,
        window: int = 1,
        k: int = 8,
        component_wise: bool = False,
        window_agg: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(window, component_wise, window_agg)
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        self.k, self.kwargs = k, {"random_state": 42, "n_init": 10, **kwargs}

    def _fit_windows(self, windows: np.ndarray) -> None:
        from sklearn.cluster import KMeans

        if len(windows) < self.k:
            raise ValueError("The number of training windows must be at least k.")
        inputs = (
            [windows[:, :, i] for i in range(windows.shape[2])]
            if self.component_wise
            else [windows.reshape(len(windows), -1)]
        )
        self.models_ = [KMeans(n_clusters=self.k, **self.kwargs).fit(values) for values in inputs]

    def _score_windows(self, windows: np.ndarray) -> np.ndarray:
        inputs = (
            [windows[:, :, i] for i in range(windows.shape[2])]
            if self.component_wise
            else [windows.reshape(len(windows), -1)]
        )
        return np.column_stack(
            [
                model.transform(values).min(axis=1)
                for model, values in zip(self.models_, inputs, strict=True)
            ]
        )


class WassersteinScorer(WindowedAnomalyScorer):
    """One-dimensional Wasserstein distance from each window to training values.

    Components are flattened into one empirical distribution unless
    ``component_wise=True``. This is not multivariate optimal transport.
    """

    component_key = "wasserstein"
    scorer_family = "direct_data"

    def __init__(
        self, window: int = 10, component_wise: bool = False, window_agg: bool = False
    ) -> None:
        super().__init__(window, component_wise, window_agg)

    def fit(self, series: TimeSeries | Sequence[TimeSeries]) -> WassersteinScorer:
        items = _inputs(series)
        super().fit(items)
        # Use each observation once; overlapping windows would bias the empirical distribution.
        self.training_values_ = np.concatenate([_values(item) for item in items])
        return self

    def _fit_windows(self, windows: np.ndarray) -> None:
        pass

    def _score_windows(self, windows: np.ndarray) -> np.ndarray:
        from scipy.stats import wasserstein_distance

        if self.component_wise:
            return np.asarray(
                [
                    [
                        wasserstein_distance(self.training_values_[:, i], window[:, i])
                        for i in range(windows.shape[2])
                    ]
                    for window in windows
                ]
            )
        training = self.training_values_.ravel()
        return np.asarray([wasserstein_distance(training, window.ravel()) for window in windows])


class PyODScorer(WindowedAnomalyScorer):
    """Adapt a PyOD-compatible estimator's fit/decision_function to windows.

    ``model`` must implement ``fit(X)`` and ``decision_function(X)`` with larger
    values meaning more anomalous. A deep copy is fitted per component so that
    models never share fitted state. Import/install PyOD only when choosing a
    PyOD model; this adapter itself has no PyOD dependency.
    """

    component_key = "pyod"
    component_discoverable = False
    scorer_family = "representation"

    def __init__(
        self,
        model: OutlierEstimator,
        window: int = 1,
        component_wise: bool = False,
        window_agg: bool = False,
    ) -> None:
        super().__init__(window, component_wise, window_agg)
        if not callable(getattr(model, "fit", None)) or not callable(
            getattr(model, "decision_function", None)
        ):
            raise TypeError("model must implement fit and decision_function.")
        self.model = model

    def _fit_windows(self, windows: np.ndarray) -> None:
        inputs = (
            [windows[:, :, i] for i in range(windows.shape[2])]
            if self.component_wise
            else [windows.reshape(len(windows), -1)]
        )
        self.models_ = [copy.deepcopy(self.model).fit(values) for values in inputs]

    def _score_windows(self, windows: np.ndarray) -> np.ndarray:
        inputs = (
            [windows[:, :, i] for i in range(windows.shape[2])]
            if self.component_wise
            else [windows.reshape(len(windows), -1)]
        )
        outputs = [
            np.asarray(model.decision_function(values))
            for model, values in zip(self.models_, inputs, strict=True)
        ]
        if any(output.shape != (len(windows),) for output in outputs):
            raise ValueError("decision_function must return one score per window.")
        return np.column_stack(outputs)


class NLLScorer(AnomalyScorer):
    """Negative log likelihood fitted separately to prediction samples per point.

    Requires deterministic observations and stochastic predictions (at least
    two samples). Degenerate scales are floored by ``min_scale``; observations
    outside a distribution's support receive the finite cap ``-log(tiny)``.
    ``window`` averages successive point scores over a causal trailing window.
    """

    scorer_family = "likelihood"

    def __init__(self, window: int = 1, min_scale: float = 1e-8) -> None:
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("window must be a positive integer.")
        if not np.isfinite(min_scale) or min_scale <= 0:
            raise ValueError("min_scale must be finite and positive.")
        self.window, self.min_scale = window, min_scale

    def _score(self, series: TimeSeries) -> TimeSeries:
        raise TypeError("NLL scorers need score_from_prediction(actual, stochastic_prediction).")

    @abstractmethod
    def _logpdf(self, actual: np.ndarray, samples: np.ndarray) -> np.ndarray:
        """Compute per-point log likelihood from a distribution estimated from samples."""

    def score_from_prediction(
        self, actual_series: SeriesLike, pred_series: SeriesLike
    ) -> TimeSeries | list[TimeSeries]:
        actual, predicted = _inputs(actual_series), _inputs(pred_series)
        if isinstance(actual_series, TimeSeries) != isinstance(pred_series, TimeSeries) or len(
            actual
        ) != len(predicted):
            raise ValueError("Actual and predicted inputs must have matching structure.")
        results = []
        for a, p in zip(actual, predicted, strict=True):
            a, p = _aligned(a, p)
            values, samples = _values(a), p.all_values(copy=False)
            if p.n_samples < 2 or not np.isfinite(samples).all():
                raise ValueError("NLL predictions require at least two finite samples per point.")
            if len(a) < self.window:
                raise ValueError("Series is shorter than the scoring window.")
            scores = -self._logpdf(values, samples)
            cap = -np.log(np.finfo(float).tiny)
            scores = np.clip(scores, -cap, cap)
            if self.window > 1:
                scores = np.lib.stride_tricks.sliding_window_view(scores, self.window, axis=0).mean(
                    axis=-1
                )
            results.append(_result(a, scores, offset=self.window - 1))
        return results[0] if isinstance(actual_series, TimeSeries) else results


class GaussianNLLScorer(NLLScorer):
    """Gaussian negative log likelihood using sample mean and standard deviation."""

    component_key = "gaussian_nll"

    def _logpdf(self, actual: np.ndarray, samples: np.ndarray) -> np.ndarray:
        from scipy.stats import norm

        return norm.logpdf(
            actual, samples.mean(axis=-1), np.maximum(samples.std(axis=-1), self.min_scale)
        )


class LaplaceNLLScorer(NLLScorer):
    """Laplace negative log likelihood using sample median and absolute deviation."""

    component_key = "laplace_nll"

    def _logpdf(self, actual: np.ndarray, samples: np.ndarray) -> np.ndarray:
        from scipy.stats import laplace

        location = np.median(samples, axis=-1)
        scale = np.maximum(np.abs(samples - location[..., None]).mean(axis=-1), self.min_scale)
        return laplace.logpdf(actual, location, scale)


class CauchyNLLScorer(NLLScorer):
    """Cauchy negative log likelihood using robust median and half-IQR estimates."""

    component_key = "cauchy_nll"

    def _logpdf(self, actual: np.ndarray, samples: np.ndarray) -> np.ndarray:
        from scipy.stats import cauchy

        location = np.median(samples, axis=-1)
        quartiles = np.quantile(samples, [0.25, 0.75], axis=-1)
        scale = np.maximum((quartiles[1] - quartiles[0]) / 2, self.min_scale)
        return cauchy.logpdf(actual, location, scale)


class ExponentialNLLScorer(NLLScorer):
    """Exponential negative log likelihood for nonnegative prediction samples."""

    component_key = "exponential_nll"

    def _logpdf(self, actual: np.ndarray, samples: np.ndarray) -> np.ndarray:
        from scipy.stats import expon

        if (samples < 0).any():
            raise ValueError("Exponential predictions must be nonnegative.")
        return expon.logpdf(actual, scale=np.maximum(samples.mean(axis=-1), self.min_scale))


class GammaNLLScorer(NLLScorer):
    """Gamma negative log likelihood with method-of-moments sample parameters."""

    component_key = "gamma_nll"

    def _logpdf(self, actual: np.ndarray, samples: np.ndarray) -> np.ndarray:
        from scipy.stats import gamma

        if (samples <= 0).any():
            raise ValueError("Gamma prediction samples must be strictly positive.")
        mean = np.maximum(samples.mean(axis=-1), self.min_scale)
        variance = np.maximum(samples.var(axis=-1), self.min_scale**2)
        return gamma.logpdf(actual, a=mean**2 / variance, scale=variance / mean)


class PoissonNLLScorer(NLLScorer):
    """Poisson negative log likelihood for nonnegative integer observations."""

    component_key = "poisson_nll"

    def _logpdf(self, actual: np.ndarray, samples: np.ndarray) -> np.ndarray:
        from scipy.stats import poisson

        if (
            (samples < 0).any()
            or (actual < 0).any()
            or not np.equal(actual, np.floor(actual)).all()
        ):
            raise ValueError("Poisson needs nonnegative samples and integer observations.")
        return poisson.logpmf(actual, mu=np.maximum(samples.mean(axis=-1), self.min_scale))
