from anomx.datasets import make_sine_anomaly_dataset
from anomx.detectors import MovingAverageDetector


def test_moving_average_detector_returns_aligned_result():
    dataset = make_sine_anomaly_dataset(periods=96, anomaly_indices=(48,))
    detector = MovingAverageDetector(window=12, threshold=2.0)

    result = detector.fit_predict(dataset)

    assert len(result.scores) == len(dataset.data)
    assert len(result.labels) == len(dataset.data)
    assert result.to_dataframe().columns.tolist() == ["score", "is_anomaly"]
