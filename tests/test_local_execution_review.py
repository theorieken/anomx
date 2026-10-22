"""Independent execution checks for leakage, alignment, and artifact consistency."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from anomx import TimeSeries
from anomx.execution import LocalExecutor, LocalResult
from anomx.models import NaiveSeasonal


class FixedForecast:
    def __init__(self, prediction):
        self.prediction = prediction
        self.fitted = None

    def fit(self, series):
        self.fitted = series.copy()
        return self

    def predict(self, n):
        assert len(self.prediction) == n
        return self.prediction


def multivariate_series():
    return TimeSeries.from_times_and_values(
        pd.date_range("2026-10-01", periods=20, freq="h"),
        np.column_stack([np.arange(20.0), 100 + np.arange(20.0)]),
        columns=["a", "b"],
    )


@pytest.mark.parametrize("fault", ["components", "nonfinite", "stochastic"])
def test_forecast_rejects_invalid_prediction_even_when_metrics_disabled(fault):
    series = multivariate_series()
    test = series[-4:]
    values = test.values()
    columns = test.components
    if fault == "components":
        columns = list(reversed(columns))
    elif fault == "nonfinite":
        values[1, 1] = np.nan
    else:
        values = np.repeat(values[:, :, None], 3, axis=2)
    prediction = TimeSeries.from_times_and_values(test.time_index, values, columns=columns)
    with pytest.raises((ValueError, TypeError)):
        LocalExecutor(verbose=False).forecast(
            FixedForecast(prediction), series, horizon=4, metrics={},
        )


def test_scaler_fit_sees_only_training_prefix_and_predictions_return_original_scale():
    class Centering:
        def fit_transform(self, series):
            self.fitted_values = series.values()
            self.center = self.fitted_values.mean(axis=0)
            return series.with_values(series.values() - self.center)

        def inverse_transform(self, series):
            return series.with_values(series.values() + self.center)

    series = TimeSeries.from_values(np.r_[np.arange(17.0), [1000.0, 2000.0, 3000.0]])
    transformer = Centering()
    result = LocalExecutor(verbose=False).forecast(
        NaiveSeasonal(), series, horizon=3, transformer=transformer,
    )
    np.testing.assert_array_equal(transformer.fitted_values[:, 0], np.arange(17.0))
    np.testing.assert_array_equal(result.prediction.univariate_values(), [16.0, 16.0, 16.0])
    assert result.train.end_time() < result.test.start_time()


class TableModel:
    def __init__(self, prediction):
        self.output = prediction

    def fit(self, data):
        return self

    def predict(self, data):
        return self.output


@pytest.mark.parametrize("fault", ["index", "columns"])
def test_reconstruction_rejects_misaligned_labeled_output(fault):
    data = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0) + 100})
    test = data[6:]
    prediction = test.iloc[::-1] if fault == "index" else test[["b", "a"]]
    with pytest.raises(ValueError):
        LocalExecutor(verbose=False).fit_predict(TableModel(prediction), data[:6], test)


def test_array_reconstruction_inherits_held_out_coordinates_for_save_and_plot(tmp_path):
    times = pd.date_range("2026-10-01", periods=10, freq="h")
    data = pd.DataFrame({"a": np.arange(10.0)}, index=times)
    test = data[6:]
    result = LocalExecutor(verbose=False).fit_predict(
        TableModel(test.to_numpy()), data[:6], test,
    )
    assert isinstance(result.prediction, pd.DataFrame)
    pd.testing.assert_frame_equal(result.prediction, test)
    result.save(tmp_path)
    predictions = pd.read_csv(tmp_path / "predictions.csv", index_col=0, parse_dates=True)
    observations = pd.read_csv(tmp_path / "observations.csv", index_col=0, parse_dates=True)
    pd.testing.assert_frame_equal(predictions, observations)


def test_reconstruction_shape_is_required_with_custom_metrics():
    data = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0) + 100})
    with pytest.raises(ValueError):
        LocalExecutor(verbose=False).fit_predict(
            TableModel(np.zeros((4, 1))), data[:6], data[6:],
            metrics={"placeholder": lambda actual, predicted: 0.0},
        )


def test_representation_preserves_observation_count():
    data = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0) + 100})
    with pytest.raises(ValueError):
        LocalExecutor(verbose=False).fit_predict(
            TableModel(np.zeros((3, 1))), data[:6], data[6:], family="representation",
        )


def test_plot_allows_caller_line_and_opacity_settings():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    from matplotlib import pyplot as plt

    series = TimeSeries.from_values(np.arange(20.0), columns=["value"])
    result = LocalExecutor(verbose=False).forecast(NaiveSeasonal(), series, horizon=4)
    axes = result.plot(alpha=0.7, linestyle=":")
    assert all(line.get_alpha() == 0.7 for line in axes.lines)
    assert all(line.get_linestyle() == ":" for line in axes.lines)
    plt.close(axes.figure)


def test_example_does_not_form_windows_across_chronological_split(tmp_path, monkeypatch):
    location = Path(__file__).resolve().parents[1] / "tmp.py"
    spec = importlib.util.spec_from_file_location("anomx_detuning_example_review", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recorded = []
    real = module.LocalExecutor

    class RecordingExecutor(real):
        def __init__(self):
            super().__init__(verbose=False)

        def fit_predict(self, model, train, test, **kwargs):
            recorded.append((train.copy(), test.copy()))
            return super().fit_predict(model, train, test, **kwargs)

    monkeypatch.setattr(module, "LocalExecutor", RecordingExecutor)
    monkeypatch.setattr(LocalResult, "save", lambda self, directory, **kwargs: Path(directory))
    length = 400
    series = TimeSeries.from_times_and_values(
        pd.date_range("2026-10-01", periods=length, freq="10min"),
        np.sin(np.arange(length) / 7) + np.arange(length) / 100,
        columns=["detuning"],
    )
    args = SimpleNamespace(include_torch=False, output=tmp_path)
    results = module.run_experiments(series, args)
    horizon = min(144, len(series) // 5)
    boundary = series.time_index[-horizon]
    assert len(results) == 5
    assert len(recorded) == 2
    for training, testing in recorded:
        assert len(training) == length - horizon - 23
        assert len(testing) == horizon - 23
        assert training.index.max() < boundary
        assert testing.index.min() == series.time_index[-horizon + 23]
        assert not training.index.intersection(testing.index).size
        np.testing.assert_allclose(training.mean().values, 0, atol=1e-12)
