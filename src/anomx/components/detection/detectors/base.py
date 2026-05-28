"""Base detector contract for component pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from anomx.components.base import BaseComponent


class BaseDetector(BaseComponent, ABC):
    """Convert scores into anomaly labels."""

    component_type = "detector"

    @abstractmethod
    def detect(self, predictions: Any) -> Any:
        """Return predictions with anomaly labels."""
