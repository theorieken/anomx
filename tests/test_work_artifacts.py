"""Portable artifact behavior without an application or scientific runtime."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from anomx import WorkContext
from anomx.execution.work.artifacts import artifact_metadata


class WorkArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "scores.csv"
        self.source.write_bytes(b"score\n0.125\n")

    def test_host_receives_verified_descriptor_without_file_bytes(self):
        accepted = {"file": "core_file-example"}
        callback = Mock(return_value=accepted)
        work = WorkContext({"artifact": callback})
        result = work.artifact(self.source, name="Test scores.csv", metadata={"split": "test"})
        self.assertIs(result, accepted)
        callback.assert_called_once_with(
            path=str(self.source.resolve()),
            name="Test-scores.csv",
            sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(),
            size_bytes=self.source.stat().st_size,
            metadata={"split": "test"},
        )
        self.assertEqual(work.events[-1]["kind"], "artifact")

    def test_local_copy_survives_source_removal_and_retries(self):
        work = WorkContext(artifact_directory=self.root / "retained")
        first = work.artifact(self.source, metadata={"channels": ["phase"]})
        second = work.artifact(self.source)
        self.assertEqual(first["path"], second["path"])
        self.source.unlink()
        self.assertEqual(Path(first["path"]).read_bytes(), b"score\n0.125\n")
        self.assertTrue(Path(first["path"]).is_relative_to(self.root / "retained"))
        self.assertNotIn("file", first)

    def test_local_different_contents_keep_distinct_copies(self):
        work = WorkContext(artifact_directory=self.root / "retained")
        first = work.artifact(self.source)
        self.source.write_bytes(b"new values")
        second = work.artifact(self.source)
        self.assertNotEqual(first["path"], second["path"])
        self.assertEqual(Path(first["path"]).read_bytes(), b"score\n0.125\n")

    def test_local_corruption_is_reported(self):
        work = WorkContext(artifact_directory=self.root / "retained")
        first = work.artifact(self.source)
        Path(first["path"]).write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "checksum"):
            work.artifact(self.source)

    def test_metadata_rejects_nonfinite_nonjson_and_nonstring_keys(self):
        callback = Mock()
        work = WorkContext({"artifact": callback})
        invalid = [
            [],
            "text",
            {"v": float("nan")},
            {"v": float("inf")},
            {"v": object()},
            {1: "coerced"},
            {"v": [{True: "coerced"}]},
        ]
        circular = {}
        circular["v"] = circular
        invalid.append(circular)
        for value in invalid:
            with self.subTest(value=type(value).__name__), self.assertRaises(ValueError):
                work.artifact(self.source, metadata=value)
        callback.assert_not_called()
        self.assertEqual(work.events, [])

    def test_metadata_is_independent_and_bounded(self):
        original = {"nested": [{"number": 1.0}]}
        copied = artifact_metadata(original)
        original["nested"][0]["number"] = 2
        self.assertEqual(copied["nested"][0]["number"], 1.0)
        with self.assertRaisesRegex(ValueError, "64 KiB"):
            artifact_metadata({"large": "x" * 65536})

    def test_names_cannot_be_paths(self):
        callback = Mock()
        for name in [
            "../scores.csv",
            "folder/scores.csv",
            "folder\\scores.csv",
            "..",
            "",
            "x" * 256,
            42,
        ]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                WorkContext({"artifact": callback}).artifact(self.source, name=name)
        callback.assert_not_called()

    def test_size_and_regular_file_checks_precede_host_write(self):
        callback = Mock()
        work = WorkContext({"artifact": callback})
        with (
            patch("anomx.execution.work.artifacts.MAX_ARTIFACT_BYTES", 3),
            self.assertRaisesRegex(ValueError, "64 MiB"),
        ):
            work.artifact(self.source)
        with self.assertRaisesRegex(ValueError, "regular file"):
            work.artifact(self.root)
        with self.assertRaises(FileNotFoundError):
            work.artifact(self.root / "missing")
        callback.assert_not_called()

    def test_local_storage_rejects_symlink_escape(self):
        directory = self.root / "retained"
        directory.mkdir()
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        outside = self.root / "outside"
        outside.mkdir()
        (directory / digest).symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "inside"):
            WorkContext(artifact_directory=directory).artifact(self.source)
        self.assertFalse((outside / self.source.name).exists())


if __name__ == "__main__":
    unittest.main()
