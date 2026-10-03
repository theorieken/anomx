"""Inference from portable ONNX model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import ArrayLike


class ONNXModel:
    """Run a published graph without importing or unpickling Python model code."""

    def __init__(self, path: str | Path, *, providers: list[str] | None = None) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("Install `anomx[onnx]` for ONNX inference.") from exc
        self.session = ort.InferenceSession(
            str(path),
            providers=providers or ["CPUExecutionProvider"],
        )

    def predict(self, observations: ArrayLike) -> np.ndarray:
        """Predict a batch of ``(batch, input_window, channels)`` observations."""
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("Use the ONNX Runtime session directly for multi-input models.")
        values = np.asarray(observations, dtype=np.float32)
        if values.ndim != 3 or not np.isfinite(values).all():
            raise ValueError("Expected finite (batch, input_window, channels) observations.")
        return cast(np.ndarray, np.asarray(self.session.run(None, {inputs[0].name: values})[0]))
