"""Base scorer contract for component pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from anomx.components.base import BaseComponent


class BaseScorer(BaseComponent, ABC):
    """Convert model outputs into standardized anomaly scores."""

    component_type = "scorer"

    @abstractmethod
    def score(self, data: Any) -> Any:
        """Return scored prediction records."""
