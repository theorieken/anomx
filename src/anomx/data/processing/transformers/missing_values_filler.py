# Copyright 2019 Unit8 SA; additional contributions by Darts contributors.
# Licensed under the Apache License, Version 2.0 (see licenses/DARTS-APACHE-2.0.txt).
# Derived from Unit8 Darts, commit dcdde2414f07f13b3bbb095428d9c6cc41492588.
# Modified for Anomx: native namespace and integration with Anomx data APIs.

"""
Missing Values Filler
---------------------
"""

from collections.abc import Mapping
from typing import Any

from anomx._series_logging import raise_log
from anomx._series_utils.missing_values import fill_missing_values
from anomx.data.processing.transformers.base_data_transformer import BaseDataTransformer
from anomx.data.timeseries import TimeSeries


class MissingValuesFiller(BaseDataTransformer):
    def __init__(
        self,
        fill: str | float = "auto",
        name: str = "MissingValuesFiller",
        n_jobs: int = 1,
        verbose: bool = False,
        columns: str | list[str] | None = None,
    ):
        """Data transformer to fill missing values from a (sequence of) deterministic ``TimeSeries``.

        Parameters
        ----------
        fill
            The value used to replace the missing values.
            If set to 'auto', will auto-fill missing values using the :meth:`pandas.DataFrame.interpolate` method.
        name
            A specific name for the transformer
        n_jobs
            The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
            passed as input to a method, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
            (sequential). Setting the parameter to `-1` means using all the available processors.
            Note: for a small amount of data, the parallelisation overhead could end up increasing the total
            required amount of time.
        verbose
            Optionally, whether to print operations progress
        columns
            Optionally, a string or list of strings specifying the names of the components (columns) to transform.
            If specified, only these components will be transformed, and the remaining components will be kept
            untouched. For more information refer to the `BaseDataTransformer` documentation. In case the transformer
            is applied on multiple TimeSeries, it is expected that all series have the same column order.

        Examples
        --------
        >>> import numpy as np
        >>> from anomx.data.timeseries import TimeSeries
        >>> from anomx.data.processing.transformers import MissingValuesFiller
        >>> values = np.arange(start=0, stop=1, step=0.25)
        >>> values[1:3] = np.nan
        >>> series = TimeSeries.from_values(values)
        >>> transformer = MissingValuesFiller()
        >>> series_filled = transformer.transform(series)
        >>> print(series_filled.values())
        [[0.  ]
         [0.25]
         [0.5 ]
         [0.75]]
        """
        if not (isinstance(fill, str) or isinstance(fill, float)):
            raise_log(
                ValueError("`fill` should either be a string or a float."),
            )
        if isinstance(fill, str) and fill != "auto":
            raise_log(
                ValueError("invalid string for `fill`: can only be set to 'auto'."),
            )
        # Define fixed params (i.e. attributes defined before calling `super().__init__`):
        self._fill = fill
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose, columns=columns)

    @staticmethod
    def ts_transform(
        series: TimeSeries, params: Mapping[str, Any], **kwargs
    ) -> TimeSeries:
        return fill_missing_values(series, params["fixed"]["_fill"], **kwargs)
