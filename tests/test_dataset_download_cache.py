"""Durability, recovery, freshness, and concurrency for resumable downloads."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from anomx.data import download as module
from anomx.data.download import DatasetDownload, DownloadConflict


def definition(**updates):
    return {
        "version": "selection-v1",
        "cache_token": "content-v1",
        "kind": "time_series",
        "time_column": "timestamp",
        "time_unit": None,
        **updates,
    }


def page(values, *, cursor=None, version="selection-v1"):
    return {
        "version": version,
        "columns": ["timestamp", "value"],
        "records": [
            {"timestamp": f"2026-10-01T00:00:{value:02d}.123456789Z", "value": value}
            for value in values
        ],
        "next_cursor": cursor,
    }


def test_interrupted_download_reopens_at_cursor_and_extends_to_complete(tmp_path):
    first = page([0, 1], cursor="opaque-next")
    with DatasetDownload(tmp_path, definition=definition()) as journal:
        assert (journal.row_count, journal.page_count, journal.next_cursor) == (0, 0, None)
        assert not journal.complete and not journal.reused
        journal.append(first, request_cursor=None)
        assert list(journal.iter_pages()) == [first]

    with DatasetDownload(tmp_path, definition=definition()) as resumed:
        assert resumed.reused and not resumed.complete
        assert resumed.next_cursor == "opaque-next"
        assert resumed.row_count == 2
        # A different requested batch size does not change journal identity.
        resumed.append(page([2, 3, 4]), request_cursor="opaque-next")
        assert resumed.complete
        assert (resumed.row_count, resumed.page_count, resumed.next_cursor) == (5, 2, None)

    with DatasetDownload(tmp_path, definition=definition()) as complete:
        assert complete.complete and complete.reused
        values = [row["value"] for part in complete.iter_pages() for row in part["records"]]
        assert values == list(range(5))


def test_empty_cursor_advancing_pages_are_durable(tmp_path):
    with DatasetDownload(tmp_path, definition=definition()) as journal:
        journal.append(page([], cursor="after-empty"), request_cursor=None)
        assert journal.page_count == 1
        assert journal.row_count == 0
        assert journal.next_cursor == "after-empty"
    with DatasetDownload(tmp_path, definition=definition()) as journal:
        assert len(list(journal.iter_pages())) == 1
        journal.append(page([1]), request_cursor="after-empty")
        assert journal.complete and journal.row_count == 1


def test_empty_terminal_page_completes_empty_dataset(tmp_path):
    with DatasetDownload(tmp_path, definition=definition()) as journal:
        journal.append(page([]), request_cursor=None)
    with DatasetDownload(tmp_path, definition=definition()) as journal:
        assert journal.complete and journal.reused
        assert journal.page_count == 1 and journal.row_count == 0


@pytest.mark.parametrize(
    "change",
    [
        {"version": "selection-v2"},
        {"cache_token": "content-v2"},
        {"cache_token": None},
        {"time_unit": "ms"},
        {"time_column": "other-time"},
        {"kind": "sequence"},
    ],
)
def test_changed_source_or_interpretation_starts_a_new_journal(tmp_path, change):
    with DatasetDownload(tmp_path, definition=definition()) as journal:
        journal.append(page([0], cursor="next"), request_cursor=None)
    with DatasetDownload(tmp_path, definition=definition(**change)) as journal:
        assert journal.page_count == journal.row_count == 0
        assert not journal.reused and not journal.complete
        assert journal.next_cursor is None
        assert list(journal.iter_pages()) == []
        assert not list(tmp_path.glob("page-*.json"))


def test_resolved_at_metadata_does_not_invalidate_same_source(tmp_path):
    first = definition(metadata={"resolved_at": "2026-10-01"})
    second = definition(metadata={"resolved_at": "2026-10-02"})
    with DatasetDownload(tmp_path, definition=first) as journal:
        journal.append(page([0], cursor="next"), request_cursor=None)
    with DatasetDownload(tmp_path, definition=second) as journal:
        assert journal.reused and journal.row_count == 1


def test_completed_without_token_needs_ttl_or_explicit_snapshot_mode(tmp_path):
    selection = definition(cache_token=None)
    with DatasetDownload(tmp_path, definition=selection) as journal:
        journal.append(page([0]), request_cursor=None)
    with DatasetDownload(tmp_path, definition=selection) as journal:
        assert not journal.complete and not journal.reused and journal.row_count == 0
        journal.append(page([1]), request_cursor=None)
    with DatasetDownload(tmp_path, definition=selection, max_age=None) as journal:
        assert journal.complete and journal.reused
        assert list(journal.iter_pages())[0]["records"][0]["value"] == 1


def test_ttl_measures_oldest_page_and_partial_resume_does_not_refresh_it(tmp_path):
    selection = definition(cache_token=None)
    with (
        patch("anomx.data.download.time.time", return_value=1000),
        DatasetDownload(tmp_path, definition=selection, max_age=60) as journal,
    ):
        journal.append(page([0], cursor="next"), request_cursor=None)
    with (
        patch("anomx.data.download.time.time", return_value=1200),
        DatasetDownload(tmp_path, definition=selection, max_age=60) as journal,
    ):
        assert journal.reused and journal.next_cursor == "next"
        journal.append(page([1]), request_cursor="next")
        assert journal.complete
    with (
        patch("anomx.data.download.time.time", return_value=1201),
        DatasetDownload(tmp_path, definition=selection, max_age=60) as journal,
    ):
        assert not journal.reused
        assert journal.row_count == 0


@pytest.mark.parametrize("elapsed,valid", [(59, True), (60, True), (61, False), (-1, False)])
def test_completed_ttl_boundary(tmp_path, elapsed, valid):
    selection = definition(cache_token=None)
    with (
        patch("anomx.data.download.time.time", return_value=1000),
        DatasetDownload(tmp_path, definition=selection, max_age=60) as journal,
    ):
        journal.append(page([0]), request_cursor=None)
    with (
        patch("anomx.data.download.time.time", return_value=1000 + elapsed),
        DatasetDownload(tmp_path, definition=selection, max_age=60) as journal,
    ):
        assert journal.reused is valid
        assert journal.complete is valid


def test_snapshot_reuse_can_be_resumed_long_after_creation(tmp_path):
    selection = definition(cache_token=None)
    with (
        patch("anomx.data.download.time.time", return_value=1000),
        DatasetDownload(tmp_path, definition=selection, max_age=None) as journal,
    ):
        journal.append(page([0]), request_cursor=None)
    with (
        patch("anomx.data.download.time.time", return_value=10**10),
        DatasetDownload(tmp_path, definition=selection, max_age=None) as journal,
    ):
        assert journal.complete and journal.reused


def three_pages(directory, *, source=None, max_age=0):
    with DatasetDownload(directory, definition=source or definition(), max_age=max_age) as journal:
        journal.append(page([0], cursor="c1"), request_cursor=None)
        journal.append(page([1], cursor="c2"), request_cursor="c1")
        journal.append(page([2]), request_cursor="c2")


@pytest.mark.parametrize("damage", ["json", "checksum", "missing"])
def test_page_damage_recovers_only_valid_contiguous_prefix(tmp_path, damage):
    three_pages(tmp_path)
    middle = tmp_path / "page-000000000001.json"
    if damage == "json":
        middle.write_text("incomplete JSON")
    elif damage == "checksum":
        envelope = json.loads(middle.read_text())
        envelope["response"]["records"][0]["value"] = 999
        middle.write_text(json.dumps(envelope))
    else:
        middle.unlink()
    with DatasetDownload(tmp_path, definition=definition()) as journal:
        assert journal.reused and not journal.complete
        assert journal.row_count == journal.page_count == 1
        assert journal.next_cursor == "c1"
        assert list(journal.iter_pages()) == [page([0], cursor="c1")]
        assert len(list(tmp_path.glob("page-*.json"))) == 1
        journal.append(page([1, 2]), request_cursor="c1")
        assert journal.complete and journal.row_count == 3


@pytest.mark.parametrize("damage", ["missing", "json", "checksum"])
def test_manifest_damage_recovers_checksums_and_completion_from_pages(tmp_path, damage):
    three_pages(tmp_path)
    manifest = tmp_path / "manifest.json"
    if damage == "missing":
        manifest.unlink()
    elif damage == "json":
        manifest.write_text("[")
    else:
        entry = json.loads(manifest.read_text())
        entry["row_count"] = 9999
        manifest.write_text(json.dumps(entry))
    with DatasetDownload(tmp_path, definition=definition()) as journal:
        assert journal.complete and journal.row_count == 3
        assert json.loads(manifest.read_text())["row_count"] == 3


def test_corrupt_completed_page_does_not_launder_expired_ttl(tmp_path):
    source = definition(cache_token=None)
    with patch("anomx.data.download.time.time", return_value=1000):
        three_pages(tmp_path, source=source, max_age=60)
    (tmp_path / "page-000000000001.json").write_text("broken")
    with (
        patch("anomx.data.download.time.time", return_value=1100),
        DatasetDownload(tmp_path, definition=source, max_age=60) as journal,
    ):
        assert not journal.reused and journal.row_count == 0


def test_page_committed_before_manifest_failure_is_recovered(tmp_path, monkeypatch):
    atomic = module._atomic_json
    with DatasetDownload(tmp_path, definition=definition()) as journal:

        def fail_manifest(path, value):
            if path.name == "manifest.json":
                raise OSError("simulated interruption after page commit")
            return atomic(path, value)

        with monkeypatch.context() as scoped:
            scoped.setattr(module, "_atomic_json", fail_manifest)
            with pytest.raises(OSError, match="interruption"):
                journal.append(page([0], cursor="next"), request_cursor=None)
    with DatasetDownload(tmp_path, definition=definition()) as journal:
        assert journal.row_count == 1 and journal.next_cursor == "next"
        assert journal.reused


def test_failed_page_write_does_not_advance_cursor_or_leave_temporary_files(tmp_path):
    with DatasetDownload(tmp_path, definition=definition()) as journal:
        with (
            patch("anomx.data.download.os.replace", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            journal.append(page([0], cursor="next"), request_cursor=None)
        assert journal.row_count == journal.page_count == 0
        assert journal.next_cursor is None
        assert not list(tmp_path.glob("page-*.json"))
        assert not list(tmp_path.glob(".download-*"))


def test_refresh_and_explicit_invalidation_clear_only_owned_journal_files(tmp_path):
    three_pages(tmp_path)
    unrelated = tmp_path / "other.txt"
    unrelated.write_text("keep")
    with DatasetDownload(tmp_path, definition=definition(), refresh=True) as journal:
        assert journal.page_count == 0 and not journal.reused
        journal.append(page([0], cursor="next"), request_cursor=None)
        journal.invalidate()
        assert journal.next_cursor is None and not journal.complete
        assert journal.row_count == journal.page_count == 0
        assert unrelated.read_text() == "keep"


def test_clear_requires_no_source_definition_and_keeps_other_files(tmp_path):
    three_pages(tmp_path)
    unrelated = tmp_path / "other.txt"
    unrelated.write_text("keep")
    # Clearing local observations needs neither metadata recovery nor a remote
    # authorization call, even if the metadata itself is unreadable or corrupt.
    (tmp_path / "manifest.json").write_text("not JSON")
    with patch.object(DatasetDownload, "_recover", side_effect=AssertionError("offline")):
        DatasetDownload.clear(tmp_path)
    DatasetDownload.clear(tmp_path)
    assert not list(tmp_path.glob("page-*.json"))
    assert unrelated.read_text() == "keep"
    with DatasetDownload(tmp_path, definition=definition()) as journal:
        assert journal.row_count == journal.page_count == 0
        assert not journal.reused


def test_clear_absent_download_does_not_create_directory(tmp_path):
    directory = tmp_path / "absent.parts"
    DatasetDownload.clear(directory)
    assert not directory.exists()


def test_clear_conflicts_with_active_writer_without_deleting_pages(tmp_path):
    with DatasetDownload(tmp_path, definition=definition()) as journal:
        journal.append(page([0], cursor="next"), request_cursor=None)
        with pytest.raises(DownloadConflict):
            DatasetDownload.clear(tmp_path)
        assert list(journal.iter_pages()) == [page([0], cursor="next")]
    DatasetDownload.clear(tmp_path)
    assert not list(tmp_path.glob("page-*.json"))


def test_append_rejects_stale_cursor_repeated_cursor_version_and_schema(tmp_path):
    with DatasetDownload(tmp_path, definition=definition()) as journal:
        journal.append(page([0], cursor="next"), request_cursor=None)
        with pytest.raises(DownloadConflict, match="obsolete"):
            journal.append(page([1]), request_cursor=None)
        with pytest.raises(ValueError, match="advance"):
            journal.append(page([1], cursor="next"), request_cursor="next")
        with pytest.raises(ValueError, match="version"):
            journal.append(page([1], version="wrong"), request_cursor="next")
        altered = page([1])
        altered["columns"] = ["time", "value"]
        with pytest.raises(ValueError, match="schema"):
            journal.append(altered, request_cursor="next")
        assert journal.row_count == 1
        journal.append(page([1]), request_cursor="next")
        with pytest.raises(DownloadConflict, match="complete"):
            journal.append(page([2]), request_cursor=None)


def test_only_one_file_per_page_and_constant_manifest_fields(tmp_path):
    with DatasetDownload(tmp_path, definition=definition()) as journal:
        fields = set(json.loads((tmp_path / "manifest.json").read_text()))
        for index in range(20):
            journal.append(
                page([index], cursor=f"c{index + 1}"),
                request_cursor=journal.next_cursor,
            )
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert set(manifest) == fields
        assert manifest["page_count"] == manifest["row_count"] == 20
        assert len(list(tmp_path.iterdir())) == 22  # 20 pages, manifest, lock.
        assert len((tmp_path / "manifest.json").read_bytes()) < 1000


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission and flock behavior")
def test_private_file_permissions_and_lock_conflict(tmp_path):
    directory = tmp_path / "private.parts"
    with DatasetDownload(directory, definition=definition()) as journal:
        journal.append(page([0], cursor="next"), request_cursor=None)
        assert directory.stat().st_mode & 0o077 == 0
        assert all(path.stat().st_mode & 0o077 == 0 for path in directory.iterdir())
        with (
            pytest.raises(DownloadConflict),
            DatasetDownload(directory, definition=definition()),
        ):
            pass
        journal.append(page([1]), request_cursor="next")
    with DatasetDownload(directory, definition=definition()) as journal:
        assert journal.row_count == 2


def test_process_exit_releases_lock_and_preserves_committed_page(tmp_path):
    code = """
import os, sys
from anomx.data.download import DatasetDownload
with DatasetDownload(sys.argv[1], definition={"version":"v1"}) as journal:
    response = {"version":"v1", "columns":["x"], "records":[{"x":1}], "next_cursor":"next"}
    journal.append(response, request_cursor=None)
    os._exit(0)
"""
    source = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        env={**os.environ, "PYTHONPATH": str(source)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    with DatasetDownload(tmp_path, definition={"version": "v1"}) as journal:
        assert journal.reused and journal.row_count == 1 and journal.next_cursor == "next"


def test_credential_and_reference_scope_is_supplied_by_distinct_directory(tmp_path):
    first, second = tmp_path / "auth-a-reference-a.parts", tmp_path / "auth-b-reference-a.parts"
    with DatasetDownload(first, definition=definition()) as journal:
        journal.append(page([0], cursor="next"), request_cursor=None)
    with DatasetDownload(second, definition=definition()) as journal:
        assert not journal.reused and journal.row_count == 0
    with DatasetDownload(first, definition=definition()) as journal:
        assert journal.reused and journal.row_count == 1


@pytest.mark.parametrize("age", [-1, float("inf"), float("nan"), True])
def test_invalid_ttl_rejected(age, tmp_path):
    with pytest.raises(ValueError, match="max_age"):
        DatasetDownload(tmp_path, definition=definition(), max_age=age)


def test_input_payload_and_definition_are_not_mutated(tmp_path):
    source = definition()
    response = page([0], cursor="next")
    original_source, original_response = deepcopy(source), deepcopy(response)
    with DatasetDownload(tmp_path, definition=source) as journal:
        journal.append(response, request_cursor=None)
        loaded = list(journal.iter_pages())[0]
        loaded["records"][0]["value"] = 99
        assert list(journal.iter_pages())[0] == original_response
    assert source == original_source and response == original_response


def test_interrupted_invalidation_cannot_resurrect_previous_snapshot(tmp_path, monkeypatch):
    three_pages(tmp_path)
    with (
        DatasetDownload(tmp_path, definition=definition()) as journal,
        monkeypatch.context() as scoped,
    ):
        scoped.setattr(journal, "_discard_suffix", lambda _: (_ for _ in ()).throw(OSError("stop")))
        with pytest.raises(OSError, match="stop"):
            journal.invalidate()
    with DatasetDownload(tmp_path, definition=definition()) as journal:
        assert journal.row_count == 0 and not journal.reused
        assert not journal.complete and journal.next_cursor is None
