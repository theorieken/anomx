"""Composition of fitted preprocessing steps."""

from collections.abc import Iterator, Sequence
from typing import Self

import pandas as pd

from anomx.data.base.transform import Transform


class ProcessingPipeline(Transform):
    """Fit transforms sequentially on training data and reuse them for inference.

    Steps are owned by the caller and retain their fitted state. Frames are
    supplied explicitly, so applying a pipeline never triggers remote dataset
    loading. An empty pipeline is an identity transform that returns a copy.
    """

    def __init__(self, steps: Sequence[Transform]) -> None:
        self.steps = tuple(steps)

    def fit(self, frame: pd.DataFrame) -> Self:
        current = frame.copy()
        for index, step in enumerate(self.steps):
            step.fit(current)
            if index < len(self.steps) - 1:
                current = step.transform(current)
        return self

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        current = frame.copy()
        for step in self.steps:
            current = step.fit_transform(current)
        return current

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        current = frame.copy()
        for step in self.steps:
            current = step.transform(current)
        return current

    def transform_batches(self, batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        current: Iterator[pd.DataFrame] = (frame.copy() for frame in batches)
        for step in self.steps:
            current = step.transform_batches(current)
        yield from current
