import pandas as pd
import pytest

from anomx.data import Dataset, TimeSeriesDataset
from anomx.data.processing import ProcessingPipeline, Resample, Transform


class Center(Transform):
    def __init__(self):
        self.fit_calls = 0

    def fit(self, frame):
        self.fit_calls += 1
        self.mean = frame.mean()
        return self

    def transform(self, frame):
        result = frame - self.mean
        result.attrs = frame.attrs.copy()
        return result


def test_fitting_uses_training_data_and_preserves_inputs_and_metadata():
    train = pd.DataFrame({"x": [1.0, 3.0]}, index=[10, 11])
    train.attrs["source_version"] = "v1"
    expected = train.copy()
    center, second = Center(), Center()
    pipeline = ProcessingPipeline([center, second])
    transformed = pipeline.fit_transform(train)
    assert center.mean.x == 2
    assert second.mean.x == 0
    pd.testing.assert_frame_equal(train, expected)
    assert transformed.x.tolist() == [-1, 1]
    assert transformed.index.tolist() == [10, 11]
    assert transformed.attrs == train.attrs
    assert pipeline.transform(pd.DataFrame({"x": [100.0]})).x.tolist() == [98]
    assert (center.fit_calls, second.fit_calls) == (1, 1)


def test_dataset_batches_reuse_fitted_steps_without_loading_all_rows():
    center = Center()
    pipeline = ProcessingPipeline([center]).fit(pd.DataFrame({"x": [1.0, 3.0]}))
    dataset = Dataset.from_dataframe(pd.DataFrame({"x": [5.0, 6.0, 7.0]}))
    result = list(pipeline.transform_batches(dataset.iter_batches(2)))
    assert [frame.x.tolist() for frame in result] == [[3, 4], [5]]
    assert center.fit_calls == 1


def test_batch_pipeline_delegates_temporal_state_to_each_transform():
    class Difference(Transform):
        def transform(self, frame):
            return frame.diff()

        def transform_batches(self, batches):
            previous = None
            for frame in batches:
                combined = pd.concat([previous, frame]) if previous is not None else frame
                yield combined.diff().iloc[-len(frame) :]
                previous = frame.tail(1)

    frame = pd.DataFrame({"x": [1.0, 2.0, 5.0, 9.0]})
    pipeline = ProcessingPipeline([Difference()])
    batches = iter([frame.iloc[:2], frame.iloc[2:]])
    pd.testing.assert_frame_equal(pd.concat(pipeline.transform_batches(batches)), frame.diff())


def test_resampling_is_shared_and_never_aggregates_partial_bins():
    frame = pd.DataFrame(
        {"x": [1.0, 3.0, 5.0, 7.0]}, index=pd.date_range("2026", periods=4, freq="s")
    )
    frame.attrs["source_version"] = "v1"
    resample = Resample("2s")
    result = resample.transform(frame)
    assert result.x.tolist() == [2, 6]
    assert result.attrs == frame.attrs
    dataset = TimeSeriesDataset.from_dataframe(
        frame, target_columns="x", metadata={"version": "v1"}
    )
    pd.testing.assert_frame_equal(dataset.resample("2s").data, result)
    assert dataset.resample("2s").metadata == dataset.metadata
    with pytest.raises(NotImplementedError, match="complete selection"):
        list(
            ProcessingPipeline([resample]).transform_batches(iter([frame.iloc[:1], frame.iloc[1:]]))
        )


def test_empty_pipeline_returns_independent_frames():
    frame = pd.DataFrame({"x": [1.0]})
    copy = ProcessingPipeline([]).fit_transform(frame)
    copy.loc[0, "x"] = 9
    assert frame.x.tolist() == [1]
