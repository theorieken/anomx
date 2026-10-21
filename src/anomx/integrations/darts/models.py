"""Common Darts model foundation."""

from importlib import import_module
from typing import Any

from darts.models.forecasting.forecasting_model import (  # type: ignore[import-untyped]
    GlobalForecastingModel,
)

from anomx.models.base.model import Model


class DartsModel(GlobalForecastingModel, Model):  # type: ignore[misc]  # Darts ships no typing marker
    """Abstract Anomx forecasting foundation inheriting the Darts contract.

    Custom Darts implementations supply ``fit``, ``predict``, their supported
    inputs and lag properties. Existing Darts models can also be used directly.
    """


_CATALOG = [
    "NaiveMean",
    "NaiveSeasonal",
    "NaiveDrift",
    "LinearRegressionModel",
    "RandomForest",
    "ExponentialSmoothing",
    "ARIMA",
    "NBEATSModel",
    "NHiTSModel",
    "RNNModel",
    "BlockRNNModel",
    "TCNModel",
    "TransformerModel",
    "TFTModel",
    "DLinearModel",
    "NLinearModel",
    "TiDEModel",
]

__all__ = ["DartsModel", *_CATALOG]


def __getattr__(name: str) -> Any:  # noqa: ANN401 - optional catalog
    if name not in _CATALOG:
        raise AttributeError(name)
    value = getattr(import_module("darts.models"), name)
    globals()[name] = value
    return value
