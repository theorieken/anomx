"""Executable statistical and PyTorch checks for native forecasting."""

import numpy as np
import pandas as pd
import pytest

from anomx.data.timeseries import TimeSeries
from anomx.models import (
    GenericModel,
    LinearRegressionModel,
    NaiveDrift,
    NaiveMean,
    NaiveSeasonal,
    TorchForecastingModel,
)


def series(values, start=0):
    return TimeSeries.from_times_and_values(
        pd.RangeIndex(start, start + len(values)), np.asarray(values), columns=["signal"]
    )


@pytest.mark.parametrize(
    "model, expected",
    [
        (NaiveMean(), [2.5, 2.5, 2.5]),
        (NaiveSeasonal(K=2), [3, 4, 3]),
        (NaiveDrift(), [5, 6, 7]),
    ],
)
def test_naive_baselines_future_alignment(model, expected):
    train = series([1, 2, 3, 4], start=10)
    prediction = model.fit(train).predict(3)
    np.testing.assert_allclose(prediction.univariate_values(), expected)
    assert prediction.time_index.equals(pd.RangeIndex(14, 17))
    assert prediction.components.equals(train.components)


def test_regression_fits_multivariate_datetime_series_and_preserves_metadata(tmp_path):
    times = pd.date_range("2026-10-01", periods=40, freq="5min")
    values = np.column_stack([np.arange(40), np.arange(40) * 3 + 2])
    data = TimeSeries.from_times_and_values(
        times, values, columns=["a", "b"], metadata={"unit": "Hz"}
    )
    model = LinearRegressionModel(lags=[-3, -1]).fit(data[:30])
    forecast = model.predict(10)
    np.testing.assert_allclose(forecast.values(), values[30:], atol=1e-9)
    assert forecast.time_index.equals(times[30:])
    assert forecast.metadata == {"unit": "Hz"}
    model.save(str(tmp_path / "model.pkl"))
    restored = LinearRegressionModel().load(str(tmp_path / "model.pkl"))
    np.testing.assert_allclose(restored.predict(10).values(), forecast.values())


def test_historical_forecasts_never_fit_future_observations():
    data = series([0, 0, 2, 4, 100, 200, 400, 800])
    model = NaiveMean()
    forecast = model.historical_forecasts(data, start=3, forecast_horizon=2, stride=2)
    np.testing.assert_allclose(forecast.univariate_values(), [2 / 3, 106 / 5])
    assert forecast.time_index.equals(pd.RangeIndex(4, 8, 2))
    assert model.training_series is None  # walk-forward copies do not mutate the original


def test_predict_requires_fit_and_matching_components():
    with pytest.raises(RuntimeError, match="Fit"):
        NaiveMean().predict(2)
    model = NaiveMean().fit(series([1, 2, 3]))
    with pytest.raises(ValueError, match="components"):
        model.predict(2, TimeSeries.from_values([1, 2, 3], columns=["different"]))
    with pytest.raises(ValueError, match="num_samples"):
        model.predict(2, num_samples=5)
    with pytest.raises(ValueError, match="positive integer"):
        model.predict(True)
    with pytest.raises(ValueError, match="finite"):
        NaiveMean().fit(series([1, np.nan, 3]))


@pytest.mark.parametrize("lags", [0, True, [], [1], [-1, -1], [-1.5]])
def test_regression_rejects_invalid_lags(lags):
    with pytest.raises(ValueError):
        LinearRegressionModel(lags=lags)


def test_ridge_regularizes_without_penalizing_constant_mean():
    model = LinearRegressionModel(lags=4, ridge=1e6).fit(series([7] * 20))
    np.testing.assert_allclose(model.predict(5).values(), 7, atol=1e-8)


def test_generic_estimator_preserves_dataframe_index():
    from sklearn.linear_model import LinearRegression

    x = pd.DataFrame({"a": np.arange(12)}, index=pd.Index(range(40, 52), name="sample"))
    model = GenericModel(LinearRegression()).fit(x, x * 3 + 2)
    prediction = model.predict(x)
    pd.testing.assert_frame_equal(prediction, x.astype(float) * 3 + 2, check_exact=False)
    with pytest.raises(ValueError, match="shape"):
        model.predict(np.ones((5, 2)))


def test_generic_callable_supports_representation_shapes_and_rejects_lost_alignment():
    x = TimeSeries.from_values(np.arange(12).reshape(4, 3), columns=["a", "b", "c"])
    model = GenericModel(lambda data: data.mean(axis=1), family="representation")
    embedding = model.predict(x)
    assert embedding.n_components == 1
    assert embedding.time_index.equals(x.time_index)
    np.testing.assert_allclose(embedding.values()[:, 0], [1, 4, 7, 10])
    with pytest.raises(ValueError, match="one output"):
        GenericModel(lambda data: data[:1]).predict(x)
    with pytest.raises(TypeError, match="no fit"):
        model.fit(x)


def test_generic_representation_estimator_uses_transform():
    from sklearn.decomposition import PCA

    data = TimeSeries.from_values(np.arange(30).reshape(10, 3), columns=["a", "b", "c"])
    model = GenericModel(PCA(n_components=1), family="representation").fit(data)
    features = model.predict(data)
    assert features.n_components == 1
    assert features.time_index.equals(data.time_index)
    np.testing.assert_allclose(features.values(), model.model.transform(data.values()))


def test_embedding_does_not_reuse_physical_channel_metadata_at_equal_width():
    from sklearn.decomposition import PCA

    data = TimeSeries.from_values(
        np.column_stack([np.arange(10), np.arange(10) ** 2]),
        columns=["detuning", "amplitude"],
        static_covariates=pd.DataFrame({"unit_code": [1, 2]}),
    )
    embedding = GenericModel(PCA(n_components=2), family="representation").fit(data).predict(data)
    assert embedding.components.tolist() == ["representation_0", "representation_1"]
    assert embedding.static_covariates is None


def test_generic_torch_module_actually_trains():
    torch = pytest.importorskip("torch")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4)
        module = torch.nn.Linear(1, 1)
    x = np.linspace(-1, 1, 64)[:, None]
    target = 2 * x + 0.4
    model = GenericModel(module, epochs=100, learning_rate=0.05, batch_size=32).fit(x, target)
    assert model.training_history[-1]["loss"] < model.training_history[0]["loss"] / 100
    assert np.mean((model.predict(x) - target) ** 2) < 1e-4
    assert not module.training


def test_generic_torch_classification_uses_integer_targets_and_returns_logits():
    torch = pytest.importorskip("torch")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8)
        module = torch.nn.Linear(1, 2)
    x = np.array([[-2], [-1], [1], [2]], dtype=float)
    labels = np.array([0, 0, 1, 1])
    model = GenericModel(module, family="classification", epochs=20, learning_rate=0.1).fit(
        x, labels
    )
    np.testing.assert_array_equal(model.predict(x).argmax(axis=1), labels)
    with pytest.raises(ValueError, match="integer class"):
        model.fit(x, [0, 0, 1.5, 1])


def test_generic_reconstruction_torch_and_shape_validation():
    torch = pytest.importorskip("torch")
    model = GenericModel(torch.nn.Linear(2, 2), family="reconstruction", epochs=2)
    x = np.ones((8, 2))
    assert model.fit(x).predict(x).shape == x.shape
    with pytest.raises(ValueError, match="targets"):
        GenericModel(torch.nn.Linear(2, 2)).fit(x)
    with pytest.raises(ValueError, match="shape"):
        GenericModel(torch.nn.Linear(2, 3), family="reconstruction").fit(x)


def test_torch_forecasting_trains_and_forecasts_native_timeseries():
    pytest.importorskip("torch")
    data = series(np.sin(np.arange(80) * 0.1))
    model = TorchForecastingModel(
        input_chunk_length=8,
        output_chunk_length=2,
        n_epochs=70,
        learning_rate=0.02,
        batch_size=32,
    ).fit(data[:70])
    forecast = model.predict(10)
    assert forecast.time_index.equals(data[70:].time_index)
    assert model.training_history[-1]["loss"] < model.training_history[0]["loss"] / 100
    assert np.sqrt(np.mean((forecast.values() - data[70:].values()) ** 2)) < 0.08


def test_torch_forecasting_refit_starts_fresh_and_accepts_minimal_context():
    pytest.importorskip("torch")
    data = series(np.sin(np.arange(30) * 0.1))
    model = TorchForecastingModel(input_chunk_length=4, n_epochs=2)
    first = model.fit(data).predict(2)
    model.fit(series(np.ones(30) * 100))
    second = model.fit(data).predict(2, series=data[-4:])
    np.testing.assert_allclose(first.values(), second.values())


def test_forecast_rejects_context_frequency_change():
    model = NaiveMean().fit(series([1, 2, 3, 4]))
    context = TimeSeries.from_times_and_values(
        pd.RangeIndex(4, 12, 2), [1, 2, 3, 4], columns=["signal"]
    )
    with pytest.raises(ValueError, match="frequency"):
        model.predict(2, context)


def test_native_forecasting_module_has_no_darts_dependency():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class NoDarts(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'darts':
            raise AssertionError('Native inference imported an optional model runtime: ' + fullname)
sys.meta_path.insert(0, NoDarts())
from anomx.data.timeseries import TimeSeries
from anomx.models import NaiveMean, LinearRegressionModel, GenericModel
from anomx.detection import GaussianNLLScorer, KMeansScorer, GenericDetector
assert NaiveMean().fit(TimeSeries.from_values([1, 2, 3])).predict(2).values().tolist() == [[2], [2]]
assert 'torch' not in sys.modules
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
