"""Optional integrations with external time-series ecosystems."""

from anomx.integrations.darts import DartsForecastingModel, is_darts_available

__all__ = [
    "DartsForecastingModel",
    "is_darts_available",
]
