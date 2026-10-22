# ruff: noqa: ANN401
# Adapter boundaries intentionally accept arbitrary third-party model/tensor types.
"""Explicit adapters for user supplied scoring and detection algorithms."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from anomx.base._arrays import as_array, call_array, component_names, restore_output
from anomx.detection.base.detectors import Detector
from anomx.detection.base.scorers import Scorer


class GenericScorer(Scorer):
    """Wrap a scoring callable or estimator with explicit anomaly direction.

    The callable receives a finite numeric array with observations on axis 0
    and returns one scalar or vector per observation. Alternatively provide an
    estimator with ``score_samples``, ``decision_function``, or ``score``.
    ``score_fn(model, values)`` overrides method selection. Set ``higher_is_anomaly=False``
    for sklearn density/inlier scores; the adapter negates those outputs.
    For DataFrames, a scalar output is added as ``score``; TimeSeries outputs
    preserve time coordinates. Fitting, when available, is always explicit.
    """

    component_key = "generic_scorer"
    component_discoverable = False

    def __init__(
        self,
        model: Any = None,
        *,
        score_fn: Callable | None = None,
        fit_fn: Callable | None = None,
        higher_is_anomaly: bool = True,
    ) -> None:
        if model is None and score_fn is None:
            raise ValueError("Provide a scoring callable/model or score_fn.")
        self.model, self.score_fn, self.fit_fn = model, score_fn, fit_fn
        self.higher_is_anomaly = higher_is_anomaly
        self._input_shape: tuple[int, ...] | None = None
        self._components: tuple[object, ...] | None = None

    def fit(self, data: Any) -> GenericScorer:
        """Fit a user-provided scorer on the given training observations."""
        values = as_array(data)
        if self.fit_fn is not None:
            self.fit_fn(self.model, values)
        elif callable(getattr(self.model, "fit", None)):
            self.model.fit(values)
        else:
            raise TypeError("This scorer has no fit method; provide fit_fn or use score directly.")
        self._input_shape = values.shape[1:]
        self._components = component_names(data)
        return self

    def score(self, data: Any) -> Any:
        """Return finite anomaly scores; larger values always mean more anomalous."""
        values = as_array(data)
        if self._input_shape is not None and values.shape[1:] != self._input_shape:
            raise ValueError("Scoring input shape must match training.")
        if self._components is not None and component_names(data) != self._components:
            raise ValueError("Scoring component names must match the training components.")
        if self.score_fn is not None:
            result = self.score_fn(self.model, values)
        elif callable(self.model):
            result = call_array(self.model, values)
        else:
            method = next(
                (
                    getattr(self.model, name)
                    for name in ("score_samples", "decision_function", "score")
                    if callable(getattr(self.model, name, None))
                ),
                None,
            )
            if method is None:
                raise TypeError("Provide a scoring method, callable, or score_fn.")
            result = method(values)
        scores = as_array(result, name="scores") * (1 if self.higher_is_anomaly else -1)
        restored = restore_output(data, scores, name="score")
        if isinstance(data, pd.DataFrame) and scores.shape[1:] == (1,):
            restored = data.copy()
            restored["score"] = scores[:, 0]
        return restored


class GenericDetector(Detector):
    """Wrap a detector callable or estimator and normalize binary labels.

    ``detect_fn(model, values)`` overrides ``detect``/``predict``. Output must
    have one row per input row. By default labels are bool or {0, 1}; set
    ``anomaly_label=-1`` for sklearn's {-1, 1} convention. Unknown labels raise
    instead of silently changing their meaning. DataFrame results add an
    ``is_anomaly`` column, while TimeSeries results are numeric binary series.
    """

    component_key = "generic_detector"
    component_discoverable = False

    def __init__(
        self,
        model: Any = None,
        *,
        detect_fn: Callable | None = None,
        fit_fn: Callable | None = None,
        anomaly_label: int = 1,
    ) -> None:
        if model is None and detect_fn is None:
            raise ValueError("Provide a detector callable/model or detect_fn.")
        if anomaly_label not in (-1, 1):
            raise ValueError("anomaly_label must be 1 for {0,1}, or -1 for sklearn labels.")
        self.model, self.detect_fn, self.fit_fn = model, detect_fn, fit_fn
        self.anomaly_label = anomaly_label
        self._input_shape: tuple[int, ...] | None = None
        self._components: tuple[object, ...] | None = None

    def fit(self, data: Any) -> GenericDetector:
        """Fit the custom detector explicitly on training data."""
        values = as_array(data)
        if self.fit_fn is not None:
            self.fit_fn(self.model, values)
        elif callable(getattr(self.model, "fit", None)):
            self.model.fit(values)
        else:
            raise TypeError(
                "This detector has no fit method; provide fit_fn or use detect directly."
            )
        self._input_shape = values.shape[1:]
        self._components = component_names(data)
        return self

    def detect(self, data: Any) -> Any:
        """Return binary labels with exactly the input observation alignment."""
        values = as_array(data)
        if self._input_shape is not None and values.shape[1:] != self._input_shape:
            raise ValueError("Detection input shape must match training.")
        if self._components is not None and component_names(data) != self._components:
            raise ValueError("Detection component names must match the training components.")
        if self.detect_fn is not None:
            result = self.detect_fn(self.model, values)
        elif callable(self.model):
            result = call_array(self.model, values)
        else:
            method = next(
                (
                    getattr(self.model, name)
                    for name in ("detect", "predict")
                    if callable(getattr(self.model, name, None))
                ),
                None,
            )
            if method is None:
                raise TypeError("Provide a detect/predict method, callable, or detect_fn.")
            result = method(values)
        labels = as_array(result, name="labels")
        allowed = (0, 1) if self.anomaly_label == 1 else (-1, 1)
        if not np.isin(labels, allowed).all():
            raise ValueError(f"Detector labels must use {allowed}.")
        labels = labels == self.anomaly_label
        restored = restore_output(data, labels, name="is_anomaly")
        if isinstance(data, pd.DataFrame) and labels.shape[1:] == (1,):
            restored = data.copy()
            restored["is_anomaly"] = labels[:, 0]
        return restored
