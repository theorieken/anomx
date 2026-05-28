"""Run a baseline anomaly detector on a synthetic dataset."""

from anomx.datasets import make_sine_anomaly_dataset
from anomx.detectors import MovingAverageDetector


def main() -> None:
    dataset = make_sine_anomaly_dataset()
    detector = MovingAverageDetector(window=24, threshold=3.0)
    result = detector.fit_predict(dataset)
    print(result.to_dataframe().query("is_anomaly").head())


if __name__ == "__main__":
    main()
