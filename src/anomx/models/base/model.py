"""Inference contract shared by native models and optional adapters."""

from abc import ABC, abstractmethod
from typing import Any

from anomx.base import BaseComponent


class Model(BaseComponent, ABC):
    """Produce model outputs without assuming training or a forecasting horizon.

    Input and output shapes belong to the concrete model's signature. Trainable
    normality models extend this contract with fitting and persistence; an
    inference-only model does not need to implement either operation.
    """

    @abstractmethod
    def predict(self, data: Any) -> Any:  # noqa: ANN401 - model-specific tensor contract
        """Return predictions, reconstructions, embeddings, or other model outputs."""
