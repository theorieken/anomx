# Copyright 2019 Unit8 SA; additional contributions by Darts contributors.
# Licensed under the Apache License, Version 2.0 (see licenses/DARTS-APACHE-2.0.txt).
# Derived from Unit8 Darts, commit dcdde2414f07f13b3bbb095428d9c6cc41492588.
# Modified for Anomx: native namespace and integration with Anomx data APIs.

import numpy as np
import pandas as pd
import pytest

from anomx.data.processing.transformers import MissingValuesFiller
from anomx.data.timeseries import TimeSeries


class TestMissingValuesFiller:
    time = pd.date_range("20130101", "20130130")
    static_covariate = pd.DataFrame({"0": [1]})

    const_series = TimeSeries.from_times_and_values(
        time, np.array([2.0] * len(time)), static_covariates=static_covariate
    )
    const_series_with_holes = TimeSeries.from_times_and_values(
        time,
        np.array([2.0] * 10 + [np.nan] * 5 + [2.0] * 10 + [np.nan] * 5),
        static_covariates=static_covariate,
    )

    lin = [float(i) for i in range(len(time))]
    lin_series = TimeSeries.from_times_and_values(time, np.array(lin))
    lin_series_with_holes = TimeSeries.from_times_and_values(
        time, np.array(lin[0:10] + [np.nan] * 5 + lin[15:24] + [np.nan] * 5 + [lin[29]])
    )

    @pytest.mark.parametrize("component_mask", [None, np.array([True])])
    def test_fill_const_series_with_const_value(self, component_mask):
        series_copy = self.const_series_with_holes.copy()
        const_transformer = MissingValuesFiller(fill=2.0)
        transformed = const_transformer.transform(
            self.const_series_with_holes, component_mask=component_mask
        )
        assert self.const_series == transformed
        assert self.const_series_with_holes == series_copy

    def test_fill_const_series_with_auto_value(self):
        auto_transformer = MissingValuesFiller()
        transformed = auto_transformer.transform(self.const_series_with_holes)
        assert self.const_series == transformed

    def test_fill_lin_series_with_auto_value(self):
        auto_transformer = MissingValuesFiller()
        transformed = auto_transformer.transform(self.lin_series_with_holes)
        assert self.lin_series == transformed

    def test_fill_static_covariates_preserved(self):
        const_transformer = MissingValuesFiller(fill=2.0)
        transformed = const_transformer.transform(self.const_series_with_holes)
        assert (
            self.const_series.static_covariates.values
            == transformed.static_covariates.values
        )

    @pytest.mark.parametrize(
        "dt_source",
        ["float32", "float64"],
    )
    def test_dtype_conversion(self, dt_source):
        series = self.const_series_with_holes.astype(dt_source)

        transformer = MissingValuesFiller(fill=2.0)
        transformed = transformer.transform(series)

        assert transformed.dtype == dt_source


class TestMissingValuesFillerInputValidation:
    def test_fill_invalid_type(self):
        with pytest.raises(ValueError, match="should either be a string or a float"):
            MissingValuesFiller(fill=None)

    def test_fill_invalid_string(self):
        with pytest.raises(ValueError, match="can only be set to 'auto'"):
            MissingValuesFiller(fill="mean")
