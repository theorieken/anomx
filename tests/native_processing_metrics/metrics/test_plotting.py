"""Numerical and Matplotlib contracts for native metric visualization."""

import numpy as np
import pytest

from anomx._series_utils.likelihood_models.base import likelihood_component_names, quantile_names
from anomx.data.timeseries import TimeSeries
from anomx.metrics import autc, plot_tolerance_curve


@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("quantiles", [None, 0.5, [0.1, 0.5, 0.9]])
def test_sampled_tolerance_plot_matches_metric(mpl_safe_plotting, components, quantiles):
    values = np.arange(1.0, 21.0).reshape(-1, 1, 1)
    values = np.repeat(values, components, axis=1)
    actual = TimeSeries.from_values(values)
    prediction = TimeSeries.from_values(
        values + np.random.default_rng(7).normal(size=(20, components, 100))
    )
    axis = plot_tolerance_curve(actual, prediction, q=quantiles)
    count = components * (len(quantiles) if isinstance(quantiles, list) else 1)
    assert len(axis.lines) == count + 1  # One additional perfect-coverage reference.
    expected = np.asarray(autc(actual, prediction, q=quantiles, component_reduction=None))
    labels = [line.get_label() for line in axis.lines[:-1]]
    for label, value in zip(labels, expected.reshape(-1)):
        assert f"AUTC = {value:.3f}" in label
    assert axis.get_xlabel() == "Tolerance (% of half-range)"


def test_explicit_axis_and_multiple_series(mpl_safe_plotting):
    import matplotlib.pyplot as plt

    actual = TimeSeries.from_values(np.arange(10.0))
    _, axis = plt.subplots()
    result = plot_tolerance_curve([actual, actual], [actual, actual], axis=axis)
    assert result is axis
    np.testing.assert_equal(axis.lines[0].get_ydata(), 100.0)
    assert "AUTC = 1.000" in axis.lines[0].get_label()


def test_named_quantile_prediction_plot(mpl_safe_plotting):
    actual = TimeSeries.from_values(np.arange(1.0, 11.0), columns=["value"])
    quantiles = [0.1, 0.5, 0.9]
    prediction = TimeSeries.from_values(
        actual.values() + np.array([-1.0, 0.0, 1.0]),
        columns=likelihood_component_names(actual.components, quantile_names(quantiles)),
    )
    axis = plot_tolerance_curve(actual, prediction, q=quantiles)
    assert len(axis.lines) == 4
    assert axis.lines[1].get_label() == "value_q0.500 (AUTC = 1.000)"
