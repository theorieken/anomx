"""Reusable time-series resampling."""

from collections.abc import Iterator

import pandas as pd

from anomx.data.base.transform import Transform


class Resample(Transform):
    """Aggregate a complete time-indexed selection at an explicit frequency.

    Independent batch aggregation can split an interval and change its result.
    Batch iteration is therefore intentionally unsupported until a streaming
    accumulator with explicit ordering/completion semantics is provided.
    """

    def __init__(self, rule: str, *, method: str = "mean") -> None:
        self.rule = rule
        self.method = method

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        resampler = frame.resample(self.rule)
        if not hasattr(resampler, self.method):
            raise ValueError(f"Unsupported resample method: {self.method}")
        result = getattr(resampler, self.method)()
        if not isinstance(result, pd.DataFrame):
            raise TypeError("Resampling must produce a DataFrame.")
        result.attrs = frame.attrs.copy()
        return result

    def transform_batches(self, batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        raise NotImplementedError(
            "Resampling requires a complete selection, not independent batches."
        )
