"""Contracts for explicit, reusable DataFrame preprocessing."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Self

import pandas as pd


class Transform(ABC):
    """Transform materialized observations while retaining their index/metadata.

    Fitting receives training data only. Implementations decide explicitly how
    to handle missing values and schema changes. Stateful temporal transforms
    must override ``transform_batches`` to carry context across batch boundaries.
    """

    def fit(self, frame: pd.DataFrame) -> Self:
        """Fit on training observations; stateless transforms need no fitting."""
        return self

    @abstractmethod
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform a frame without modifying the caller's input."""

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fit on a training frame and transform that same frame."""
        self.fit(frame)
        return self.transform(frame)

    def transform_batches(self, batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        """Transform independent batches without refitting or materializing all rows."""
        for frame in batches:
            yield self.transform(frame)
