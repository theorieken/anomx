"""Native models and lazy optional model integrations."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ARIMA": "anomx.integrations.darts.models",
    "BaseAnomalyModel": "anomx.models.base",
    "BlockRNNModel": "anomx.integrations.darts.models",
    "ConstantBaselineModel": "anomx.models.forecasting.constant",
    "DLinearModel": "anomx.integrations.darts.models",
    "DartsNaiveSeasonalModel": "anomx.models.forecasting.darts_naive_seasonal",
    "ExponentialSmoothing": "anomx.integrations.darts.models",
    "InferenceKind": "anomx.models.results",
    "InferenceOutput": "anomx.models.results",
    "IsolationForestModel": "anomx.models.representation.isolation_forest",
    "LinearRegressionModel": "anomx.integrations.darts.models",
    "Model": "anomx.models.base",
    "NBEATSModel": "anomx.integrations.darts.models",
    "NHiTSModel": "anomx.integrations.darts.models",
    "NLinearModel": "anomx.integrations.darts.models",
    "NaiveDrift": "anomx.integrations.darts.models",
    "NaiveMean": "anomx.integrations.darts.models",
    "NaiveSeasonal": "anomx.integrations.darts.models",
    "NormalityModel": "anomx.models.base.normality",
    "ONNXModel": "anomx.models.onnx",
    "PcaReconstructionModel": "anomx.models.reconstruction.pca",
    "PyTorchModel": "anomx.models.torch",
    "RNNModel": "anomx.integrations.darts.models",
    "RandomForest": "anomx.integrations.darts.models",
    "RollingWindowForecastModel": "anomx.models.forecasting.rolling_window",
    "TCNModel": "anomx.integrations.darts.models",
    "TFTModel": "anomx.integrations.darts.models",
    "TiDEModel": "anomx.integrations.darts.models",
    "TorchAutoencoderModel": "anomx.models.reconstruction.torch_autoencoder",
    "TransformerModel": "anomx.integrations.darts.models",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:  # noqa: ANN401 - lazy public API
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
