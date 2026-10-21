"""Base connector contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable


class BaseConnector(ABC):
    """Read and write datasets for component workflows."""

    @abstractmethod
    def read(self, config: dict[str, Any]) -> Any:
        """Read data from a source."""

    @abstractmethod
    def write(self, data: Any, config: dict[str, Any]) -> None:
        """Write data to a target."""

    @abstractmethod
    def stream(self, config: dict[str, Any]) -> Iterable[Any]:
        """Stream records from a source."""

    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """Return the connector configuration schema."""
