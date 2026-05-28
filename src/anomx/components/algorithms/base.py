"""Base algorithm contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from anomx.components.algorithms.contracts import JobResult, JobSpec
from anomx.components.base import BaseComponent


class BaseAlgorithm(BaseComponent, ABC):
    """Base class for reusable anomaly workflows."""

    component_type = "algorithm"

    @abstractmethod
    def run_job(self, job_spec: JobSpec | dict[str, Any]) -> JobResult:
        """Execute an anomaly-detection workflow."""
