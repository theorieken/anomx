import json

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from anomx import TimeSeries
from anomx.data.processing.transformers import Scaler
from anomx.execution import LocalExecutor
from anomx.models import GenericModel, LinearRegressionModel, NaiveSeasonal


def test_forecast_splits_before_scaling_and_exports(tmp_path):
    series = TimeSeries.from_values(np.arange(80, dtype=float), columns=["x"])
    scaler = Scaler(StandardScaler())
    run = LocalExecutor(verbose=False).forecast(
        LinearRegressionModel(lags=4, ridge=0.01),
        series,
        horizon=10,
        transformer=scaler,
    )
    assert run.prediction.time_index.equals(series[-10:].time_index)
    assert run.metrics["mae"] < 0.01
    assert run.metadata["train_rows"] == 70
    run.save(tmp_path)
    assert json.loads((tmp_path / "run.json").read_text())["family"] == "forecasting"
    assert len(pd.read_csv(tmp_path / "predictions.csv")) == 10


def test_forecast_rejects_gaps_before_fitting():
    values = np.arange(20, dtype=float)
    values[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        LocalExecutor(verbose=False).forecast(NaiveSeasonal(), TimeSeries.from_values(values))


def test_generic_reconstruction_and_representation():
    data = pd.DataFrame({"x": np.arange(30.0), "y": np.arange(30.0) * 2})
    model = GenericModel(
        PCA(n_components=1),
        family="reconstruction",
        predict_fn=lambda fitted, x: fitted.inverse_transform(fitted.transform(x)),
    )
    result = LocalExecutor(verbose=False).fit_predict(model, data[:20], data[20:])
    assert result.metrics["rmse"] < 1e-12
    representation = GenericModel(PCA(n_components=1), family="representation")
    result = LocalExecutor(verbose=False).fit_predict(
        representation, data[:20], data[20:], family="representation"
    )
    assert result.metrics == {}
    assert np.asarray(result.prediction).shape == (10, 1)


def test_local_plot_returns_axes(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    series = TimeSeries.from_values(np.arange(40.0), columns=["x"])
    result = LocalExecutor(verbose=False).forecast(NaiveSeasonal(), series, horizon=5)
    _, axes = plt.subplots()
    assert result.plot(ax=axes) is axes
    result.save(tmp_path, plot=True)
    assert (tmp_path / "result.png").stat().st_size > 1000
    plt.close("all")
