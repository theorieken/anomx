"""Base contract for component-oriented anomaly models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from anomx.components.base import BaseComponent


class BaseAnomalyModel(BaseComponent, ABC):
    """Base class for trainable pipeline models."""

    component_type = "model"

    @abstractmethod
    def fit(self, data: Any) -> None:
        """Train the model using input data."""

    @abstractmethod
    def predict(self, data: Any) -> Any:
        """Generate model outputs for input data."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model state."""

    @abstractmethod
    def load(self, path: str) -> "BaseAnomalyModel":
        """Load model state and return the model instance."""
