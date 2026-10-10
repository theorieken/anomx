import pandas as pd
import pytest

from anomx import Dataset, WorkContext
from anomx.agent.skills import load_builtin_skills, parse_skill_markdown, skill_to_markdown


def test_dataset_batches_pin_version_and_keep_timestamp_index():
    calls = []

    def read(**arguments):
        calls.append(arguments)
        offset = 1 if arguments["cursor"] else 0
        return {
            "version": "v1",
            "columns": ["time", "x"],
            "records": [{"time": f"2026-10-09T00:00:0{offset}Z", "x": offset}],
            "next_cursor": "next" if offset == 0 else None,
        }

    work = WorkContext(
        callbacks={
            "describe_dataset": lambda **_: {
                "version": "v1",
                "kind": "time_series",
                "mode": "fixed",
                "time_column": "time",
            },
            "read_dataset": read,
        }
    )
    with work.activate():
        frame = Dataset("data_dataset-example").to_dataframe()
    assert list(frame.x) == [0, 1]
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert str(frame.index.tz) == "UTC"
    assert [call["version"] for call in calls] == ["v1", "v1"]


@pytest.mark.parametrize(
    "response",
    [
        {"version": "other", "columns": [], "records": []},
        {"version": "v1", "columns": ["x"], "records": [{"x": 1}], "next_cursor": "repeat"},
    ],
)
def test_dataset_rejects_changed_versions_and_looping_cursors(response):
    with (
        WorkContext(
            callbacks={
                "describe_dataset": lambda **_: {
                    "version": "v1",
                    "kind": "sequence",
                    "mode": "automatic",
                },
                "read_dataset": lambda **_: response,
            }
        ).activate(),
        pytest.raises(ValueError),
    ):
        Dataset("data_dataset-example").to_dataframe()


def test_local_dataset_does_not_mutate_input_and_needs_no_darts():
    frame = pd.DataFrame({"x": [1, 2, 3]})
    data = Dataset.from_dataframe(frame)
    frame.loc[0, "x"] = 99
    assert data.values().tolist() == [[1], [2], [3]]
    assert [len(batch) for batch in data.iter_batches(2)] == [2, 1]
    with pytest.raises(ValueError):
        Dataset.from_channels("not-a-list")


@pytest.mark.parametrize(
    "unit, timestamp", [("s", 1), ("ms", 1000), ("us", 1000000), ("ns", 1000000000)]
)
def test_numeric_timestamps_use_the_resolved_unit(unit, timestamp):
    with WorkContext(
        callbacks={
            "describe_dataset": lambda **_: {
                "version": "v1",
                "kind": "time_series",
                "mode": "fixed",
                "time_column": "time",
                "time_unit": unit,
            },
            "read_dataset": lambda **_: {
                "version": "v1",
                "columns": ["time", "x"],
                "records": [{"time": timestamp, "x": 7}],
            },
        }
    ).activate():
        frame = Dataset("data_dataset-example").to_dataframe()
    assert frame.index[0] == pd.Timestamp("1970-01-01T00:00:01Z")
    assert list(frame.columns) == ["x"]


def test_dataset_skill_keeps_platform_requirement_when_serialized():
    skill = next(skill for skill in load_builtin_skills() if skill.command == "create-dataset")
    assert skill.requires_platform
    parsed = parse_skill_markdown(
        skill_to_markdown(skill), default_command=skill.command, source="platform", path=None
    )
    assert parsed.requires_platform
