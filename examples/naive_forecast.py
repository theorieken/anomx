"""Fit a naive seasonal forecast model."""

from anomx.datasets import make_sine_anomaly_dataset
from anomx.models import NaiveSeasonalModel


def main() -> None:
    dataset = make_sine_anomaly_dataset()
    model = NaiveSeasonalModel(season_length=24).fit(dataset)
    forecast = model.predict(12)
    print(forecast.to_dataframe())


if __name__ == "__main__":
    main()
