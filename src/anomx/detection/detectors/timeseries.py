"""Native component-wise decisions for deterministic anomaly-score series."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from anomx.base.capabilities import ComponentCapabilities
from anomx.data.timeseries import TimeSeries
from anomx.detection.base.detectors import AdaptiveThresholdDetector, StaticThresholdDetector
from anomx.detection.scorers.timeseries import _inputs, _result, _values

Bounds = float | Sequence[float] | np.ndarray | None
SeriesLike = TimeSeries | Sequence[TimeSeries]


def _bounds(value: Bounds, width: int, default: float, name: str) -> np.ndarray:
    if value is None:
        return np.full(width, default)
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        values = np.repeat(values, width)
    if values.shape != (width,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite and scalar or match the number of components.")
    return values


class TimeSeriesThresholdDetector(StaticThresholdDetector):
    """Flag scores strictly below or above explicit component-wise bounds.

    At least one of ``low_threshold`` and ``high_threshold`` is required.
    Each can be a scalar or one value per component. Equality is in bounds.
    Returns a binary TimeSeries (1 anomalous, 0 normal), retaining coordinates.
    This differs from the older table-oriented ``ThresholdDetector``, which
    adds a column to a record table and uses an absolute-value threshold.
    """

    component_key = "timeseries_threshold"
    capabilities = ComponentCapabilities(
        supports_sequences=True, supports_unordered_samples=False, requires_training=False
    )

    def __init__(self, low_threshold: Bounds = None, high_threshold: Bounds = None) -> None:
        if low_threshold is None and high_threshold is None:
            raise ValueError("At least one threshold is required.")
        self.low_threshold, self.high_threshold = low_threshold, high_threshold

    def detect(self, series: TimeSeries | Sequence[TimeSeries]) -> TimeSeries | list[TimeSeries]:
        """Apply configured bounds to one or several independent score series."""
        result = []
        for item in _inputs(series):
            values = _values(item)
            low = _bounds(self.low_threshold, item.n_components, -np.inf, "low_threshold")
            high = _bounds(self.high_threshold, item.n_components, np.inf, "high_threshold")
            if (low > high).any():
                raise ValueError("Low thresholds cannot exceed high thresholds.")
            result.append(_result(item, ((values < low) | (values > high)).astype(float)))
        return result[0] if isinstance(series, TimeSeries) else result


class QuantileDetector(AdaptiveThresholdDetector):
    """Learn lower/upper per-component quantiles from explicit training scores.

    Unlike the legacy ``QuantileThresholdDetector``, calibration is separate
    from detection, so evaluation data never changes its own threshold.
    Quantiles can be scalars or one probability per component; use ``None``
    to disable either side. Fitting on a sequence pools observations while
    preserving component order. Probabilities must lie in [0, 1].
    """

    component_key = "quantile"
    capabilities = ComponentCapabilities(
        supports_sequences=True, supports_unordered_samples=False, requires_training=True
    )

    def __init__(self, low_quantile: Bounds = None, high_quantile: Bounds = 0.99) -> None:
        if low_quantile is None and high_quantile is None:
            raise ValueError("At least one quantile is required.")
        self.low_quantile, self.high_quantile = low_quantile, high_quantile
        self._components: list[str] | None = None
        self.low_threshold: np.ndarray | None = None
        self.high_threshold: np.ndarray | None = None

    def _fit_bounds(self, values: np.ndarray) -> None:
        width = values.shape[1]
        low = _bounds(self.low_quantile, width, 0.0, "low_quantile")
        high = _bounds(self.high_quantile, width, 1.0, "high_quantile")
        if (low < 0).any() or (high > 1).any() or (low > high).any():
            raise ValueError("Quantiles must satisfy 0 <= low <= high <= 1.")
        self.low_threshold = (
            None
            if self.low_quantile is None
            else np.asarray([np.quantile(values[:, i], low[i]) for i in range(width)])
        )
        self.high_threshold = (
            None
            if self.high_quantile is None
            else np.asarray([np.quantile(values[:, i], high[i]) for i in range(width)])
        )

    def fit(self, series: TimeSeries | Sequence[TimeSeries]) -> QuantileDetector:
        """Fit thresholds using training scores only."""
        items = _inputs(series)
        components = list(items[0].components)
        if any(list(item.components) != components for item in items):
            raise ValueError("Training series must have identical ordered components.")
        values = np.concatenate([_values(item) for item in items])
        self._components = None
        self._fit_bounds(values)
        self._components = components
        return self

    def detect(self, series: TimeSeries | Sequence[TimeSeries]) -> TimeSeries | list[TimeSeries]:
        """Apply the fixed training thresholds to new observations."""
        if self._components is None:
            raise RuntimeError("Fit the detector before detection.")
        if any(list(item.components) != self._components for item in _inputs(series)):
            raise ValueError("Detection components must match the training components.")
        return TimeSeriesThresholdDetector(self.low_threshold, self.high_threshold).detect(series)

    def fit_detect(self, series: SeriesLike) -> TimeSeries | list[TimeSeries]:
        """Fit and label the same input; use separate data when evaluating performance."""
        return self.fit(series).detect(series)


class IQRDetector(QuantileDetector):
    """Learn Tukey fences ``Q1 - scale*IQR`` and ``Q3 + scale*IQR``.

    ``scale`` is nonnegative (default 1.5) and may be supplied per component.
    With constant training scores, any different value lies outside the fence.
    """

    component_key = "iqr"

    def __init__(self, scale: Bounds = 1.5) -> None:
        super().__init__(0.25, 0.75)
        self.scale = scale

    def _fit_bounds(self, values: np.ndarray) -> None:
        scale = _bounds(self.scale, values.shape[1], 1.5, "scale")
        if (scale < 0).any():
            raise ValueError("IQR scale must be nonnegative.")
        q1, q3 = np.quantile(values, [0.25, 0.75], axis=0)
        self.low_threshold = q1 - scale * (q3 - q1)
        self.high_threshold = q3 + scale * (q3 - q1)
