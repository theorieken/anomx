"""Native scorer, calibration, probabilistic, and custom-detector checks."""

import numpy as np
import pandas as pd
import pytest

from anomx.data.timeseries import TimeSeries
from anomx.detection import (
    CauchyNLLScorer,
    DifferenceScorer,
    ExponentialNLLScorer,
    GammaNLLScorer,
    GaussianNLLScorer,
    GenericDetector,
    GenericScorer,
    IQRDetector,
    KMeansScorer,
    LaplaceNLLScorer,
    NormScorer,
    PoissonNLLScorer,
    PyODScorer,
    QuantileDetector,
    TimeSeriesThresholdDetector,
    WassersteinScorer,
)


def series(values, start=0, columns=None):
    return TimeSeries.from_times_and_values(
        pd.RangeIndex(start, start + len(values)), np.asarray(values), columns=columns
    )


def test_residual_scores_align_time_and_require_matching_components():
    actual = series([[1, 3], [2, 5], [3, 7]], columns=["a", "b"])
    predicted = series([[1, 2], [1, 3]], start=1, columns=["a", "b"])
    diff = DifferenceScorer().score_from_prediction(actual, predicted)
    np.testing.assert_allclose(diff.values(), [[1, 3], [2, 4]])
    norm = NormScorer().score_from_prediction(actual, predicted)
    np.testing.assert_allclose(norm.univariate_values(), np.sqrt([10, 20]))
    component = NormScorer(component_wise=True).score_from_prediction(actual, predicted)
    np.testing.assert_allclose(component.values(), diff.values())
    assert norm.time_index.equals(pd.RangeIndex(1, 3))
    with pytest.raises(ValueError, match="component"):
        DifferenceScorer().score_from_prediction(actual, series([[1, 2]], columns=["c", "d"]))


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (KMeansScorer, {"k": 1}),
        (WassersteinScorer, {}),
    ],
)
def test_window_scorers_train_and_preserve_causal_alignment(cls, kwargs):
    train = series(np.zeros((12, 2)), columns=["a", "b"])
    test = series(np.concatenate([np.zeros((5, 2)), np.ones((5, 2)) * 10]), columns=["a", "b"])
    scorer = cls(window=3, component_wise=True, **kwargs).fit(train)
    scores = scorer.score(test)
    assert len(scores) == len(test) - 2
    assert scores.time_index.equals(pd.RangeIndex(2, 10))
    assert scores.n_components == 2
    assert scores.values()[-1, 0] > scores.values()[0, 0] + 5
    with pytest.raises(ValueError, match="window"):
        scorer.score(test[:2])


def test_window_aggregation_and_training_sequences_do_not_cross_boundaries():
    scorer = KMeansScorer(window=3, k=1, window_agg=True).fit(
        [series([0, 0, 0]), series([0, 0, 0])]
    )
    result = scorer.score(series([0, 0, 0, 3]))
    np.testing.assert_allclose(result.univariate_values(), [0, 1.5, 1.5, 3])
    with pytest.raises(ValueError, match="At least"):
        scorer.fit([series([1, 2]), series([3, 4])])


class DistanceEstimator:
    def fit(self, values):
        self.center = values.mean(axis=0)
        return self

    def decision_function(self, values):
        return np.linalg.norm(values - self.center, axis=1)


def test_pyod_compatible_estimator_copies_state_per_component():
    original = DistanceEstimator()
    scorer = PyODScorer(original, window=2, component_wise=True)
    train = series(np.column_stack([np.arange(10), np.arange(10) + 100]))
    result = scorer.fit_score(train)
    assert not hasattr(original, "center")
    assert scorer.models_[0] is not scorer.models_[1]
    np.testing.assert_allclose(result.values()[:, 0], result.values()[:, 1])


@pytest.mark.parametrize(
    "cls",
    [GaussianNLLScorer, LaplaceNLLScorer, CauchyNLLScorer, ExponentialNLLScorer, GammaNLLScorer],
)
def test_probabilistic_nll_scores_increase_for_surprising_observations(cls):
    rng = np.random.default_rng(8)
    samples = series(rng.uniform(0.5, 2.0, size=(6, 1, 100)))
    actual = series([1, 1, 1, 1, 1, 20])
    result = cls(window=2).score_from_prediction(actual, samples)
    assert len(result) == 5
    assert result.univariate_values()[-1] > result.univariate_values()[0]
    assert np.isfinite(result.values()).all()


def test_nll_degenerate_samples_poisson_and_domain_errors():
    samples = series(np.ones((4, 1, 10)))
    assert np.isfinite(
        GaussianNLLScorer().score_from_prediction(series([1, 1, 1, 2]), samples).values()
    ).all()
    score = PoissonNLLScorer().score_from_prediction(series([1, 1, 1, 10]), samples)
    assert score.values()[-1, 0] > score.values()[0, 0]
    with pytest.raises(ValueError, match="integer"):
        PoissonNLLScorer().score_from_prediction(series([1.5, 1, 1, 1]), samples)
    with pytest.raises(ValueError, match="at least two"):
        GaussianNLLScorer().score_from_prediction(series([1, 2]), series([1, 2]))
    with pytest.raises(TypeError, match="score_from_prediction"):
        GaussianNLLScorer().score(series([1, 2]))


def test_threshold_quantile_and_iqr_decisions_and_no_test_calibration():
    train = series([0, 1, 2, 3, 4])
    test = series([2, 3, 100, 200])
    detector = QuantileDetector(high_quantile=0.75).fit(train)
    np.testing.assert_array_equal(detector.detect(test).univariate_values(), [0, 0, 1, 1])
    np.testing.assert_allclose(detector.high_threshold, [3])
    np.testing.assert_array_equal(
        IQRDetector().fit(train).detect(test).univariate_values(), [0, 0, 1, 1]
    )
    np.testing.assert_array_equal(
        TimeSeriesThresholdDetector(low_threshold=1, high_threshold=3)
        .detect(train)
        .univariate_values(),
        [1, 0, 0, 0, 1],
    )
    with pytest.raises(RuntimeError, match="Fit"):
        QuantileDetector().detect(test)


def test_detector_bounds_and_fitted_components_are_validated():
    with pytest.raises(ValueError, match="At least"):
        TimeSeriesThresholdDetector()
    with pytest.raises(ValueError, match="cannot exceed"):
        TimeSeriesThresholdDetector(5, 2).detect(series([1, 2]))
    with pytest.raises(ValueError, match="Quantiles"):
        QuantileDetector(high_quantile=1.1).fit(series([1, 2]))
    with pytest.raises(ValueError, match="nonnegative"):
        IQRDetector(-1).fit(series([1, 2]))
    with pytest.raises(ValueError, match="components"):
        QuantileDetector().fit(series([1, 2], columns=["a"])).detect(series([1, 2], columns=["b"]))


def test_custom_scorer_and_detector_work_on_tables_and_timeseries():
    data = pd.DataFrame({"value": [1, 2, 8]}, index=[10, 20, 30])
    scored = GenericScorer(lambda values: values[:, 0] ** 2).score(data)
    assert scored["score"].tolist() == [1, 4, 64]
    assert scored.index.equals(data.index)
    custom = GenericDetector(lambda values: values[:, -1] > 10)
    labeled = custom.detect(scored)
    assert labeled["is_anomaly"].tolist() == [False, False, True]
    labels = GenericDetector(
        lambda values: np.where(values[:, 0] > 5, -1, 1), anomaly_label=-1
    ).detect(series([1, 2, 8]))
    assert labels.univariate_values().tolist() == [0, 0, 1]


def test_generic_scorer_direction_and_detector_fit_callbacks():
    scorer = GenericScorer(lambda data: -data[:, 0], higher_is_anomaly=False)
    np.testing.assert_allclose(scorer.score(np.array([1, 2, 3]))[:, 0], [1, 2, 3])
    state = {}
    detector = GenericDetector(
        state,
        fit_fn=lambda model, data: model.update(threshold=float(data.mean())),
        detect_fn=lambda model, data: data[:, 0] > model["threshold"],
    ).fit(np.array([1, 2, 3]))
    np.testing.assert_array_equal(detector.detect(np.array([1, 3]))[:, 0], [0, 1])
    with pytest.raises(ValueError, match="labels must"):
        GenericDetector(lambda values: np.full(len(values), 4)).detect(np.array([1, 2]))
    with pytest.raises(ValueError, match="one output"):
        GenericDetector(lambda values: np.ones(1)).detect(np.array([1, 2]))
    with pytest.raises(ValueError, match="finite"):
        GenericScorer(lambda values: np.full(len(values), np.nan)).score(np.array([1, 2]))


def test_generic_torch_scoring_and_detection_modules_work_without_numpy_callbacks():
    torch = pytest.importorskip("torch")

    class Magnitude(torch.nn.Module):
        def forward(self, inputs):
            return inputs.abs().sum(dim=1)

    class Positive(torch.nn.Module):
        def forward(self, inputs):
            return inputs[:, 0] > 0

    values = series([-2, 0, 3])
    np.testing.assert_allclose(GenericScorer(Magnitude()).score(values).values()[:, 0], [2, 0, 3])
    np.testing.assert_allclose(GenericDetector(Positive()).detect(values).values()[:, 0], [0, 0, 1])
