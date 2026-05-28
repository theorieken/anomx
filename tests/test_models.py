from anomx.datasets import make_sine_anomaly_dataset
from anomx.models import NaiveSeasonalModel


def test_naive_seasonal_model_predicts_requested_horizon():
    dataset = make_sine_anomaly_dataset(periods=48)
    model = NaiveSeasonalModel(season_length=24).fit(dataset)

    forecast = model.predict(6)

    assert len(forecast.values) == 6
    assert forecast.values.columns.tolist() == ["value"]
