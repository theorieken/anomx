"""Native-reader boundaries exercised through real isolated child processes."""

import os
import signal
import subprocess
import sys
import time
from contextlib import closing
from email.utils import formatdate

import pytest

from anomx.data.sources.daq import DaqReadError, iter_daq_records
from anomx.data.sources.mounted import DatasetSourceChanged, MountedSource

_FAKE = """
import os, time
class Scalar:
    shape = (1, 2)
    def __init__(self, value): self.value = value
    def __getitem__(self, index): return self.value if index == (0, 1) else 0.0
count = 0
channel = None
def connect(**kwargs):
    global channel
    assert kwargs["local"] is True
    assert os.path.isabs(kwargs["files"][0])
    channel = kwargs["chans"][0]
    print("native library diagnostic on stdout", flush=True)
    if os.environ.get("PID_FILE"):
        with open(os.environ["PID_FILE"], "w") as stream: stream.write(str(os.getpid()))
    if os.environ.get("MUTATE"):
        with open(kwargs["files"][0], "ab") as stream: stream.write(b"x")
def disconnect(): pass
def getdata():
    global count
    if os.environ.get("STALL") and count >= 1: time.sleep(30)
    if count >= 5: return None
    n = count
    count += 1
    result = []
    for index in [7, 31]:
        if os.environ.get("MISSING") and index == 31: continue
        stamp = 1234.25 + n / 10
        if os.environ.get("MISALIGNED") and index == 31: stamp += 1
        name = "field" + str(index)
        if os.environ.get("LONG_NAME"): name = "x" * 5000
        result.append({
            "type": "OTHER" if os.environ.get("OTHER_TYPE") else "A_TS_GSPECTRUM",
            "data": Scalar(float("nan") if n == 2 and index == 7 else n + index / 100),
            "timestamp": stamp, "macropulse": 100 + n,
            "miscellaneous": {"index": index, "daqname": channel, "comment": name},
        })
    return [result]
"""


@pytest.fixture
def native(tmp_path):
    path = tmp_path / "source.raw"
    path.write_bytes(b"immutable raw fixture")
    stat = path.stat()
    source = MountedSource(
        tmp_path,
        {
            "path": "/source.raw",
            "size_bytes": stat.st_size,
            "modified_at": formatdate(stat.st_mtime, usegmt=True),
        },
        "pinned-version",
    )
    module = tmp_path / "native_lib"
    module.mkdir()
    (module / "pydaq.py").write_text(_FAKE)
    return source, {"PYTHONPATH": str(module)}, tmp_path


def read(native, **kwargs):
    source, environment, _ = native
    return iter_daq_records(
        source,
        channel="TEST/DAQ/CHANNEL",
        fields={"input": 7, "output": 31},
        experiment="example",
        linac="XFEL",
        python_executable=sys.executable,
        environment=environment,
        **kwargs,
    )


def test_batched_native_reader_preserves_fields_times_order_and_nonfinite(native):
    source, _, _ = native
    before = source.sha256()
    with closing(read(native, batch_size=2)) as records:
        rows = list(records)
    assert [row.event_id for row in rows] == list(range(100, 105))
    assert [row.record_index for row in rows] == list(range(5))
    assert rows[1].timestamp == 1234.35
    assert rows[1].values == {"input": 1.07, "output": 1.31}
    assert rows[2].values["input"] is None
    assert rows[2].invalid_fields == {"input": "nan"}
    assert rows[2].field_metadata["input"]["name"] == "field7"
    assert rows[0].source_path == "/source.raw"
    assert rows[0].dataset_version == "pinned-version"
    assert rows[0].channel == "TEST/DAQ/CHANNEL"
    assert source.sha256() == before


def test_explicit_limit_and_independent_connections(native):
    assert len(list(read(native, limit=3))) == 3
    assert list(read(native, limit=0)) == []
    assert list(read(native, limit=1))[0].event_id == 100


@pytest.mark.parametrize(
    ("variable", "message"),
    [
        ("MISSING", "absent"),
        ("MISALIGNED", "inconsistent"),
        ("OTHER_TYPE", "A_TS_GSPECTRUM"),
        ("LONG_NAME", "oversized"),
    ],
)
def test_native_schema_errors_fail_closed(native, variable, message):
    native[1][variable] = "1"
    with pytest.raises(DaqReadError, match=message):
        list(read(native))


def test_missing_optional_dependency_has_actionable_error(native):
    module = native[2] / "native_lib" / "pydaq.py"
    module.write_text('raise ImportError("test missing native ABI")')
    with pytest.raises(DaqReadError, match="Optional pydaq is unavailable"):
        list(read(native))


def test_stalled_native_reader_times_out_and_is_reaped(native):
    pid_file = native[2] / "pid"
    native[1].update(STALL="1", PID_FILE=str(pid_file))
    start = time.monotonic()
    with closing(read(native, batch_size=1, timeout_seconds=0.5)) as records:
        assert next(records).event_id == 100
        with pytest.raises(DaqReadError, match="timed out"):
            next(records)
    assert time.monotonic() - start < 5
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


@pytest.mark.skipif(os.name != "posix", reason="Process-group cleanup for stalled subprocess test")
def test_large_request_startup_stall_obeys_progress_timeout(native):
    script = r"""
import sys
from pathlib import Path
from email.utils import formatdate
import anomx.data.sources.daq as daq
from anomx.data.sources.mounted import MountedSource
path = Path(sys.argv[1]).resolve()
stat = path.stat()
source = MountedSource(path.parent, {
    "path": "/" + path.name, "size_bytes": stat.st_size,
    "modified_at": formatdate(stat.st_mtime, usegmt=True),
}, "fixture")
daq._CHILD = "import time; time.sleep(30)"
# Valid Unicode arguments expand beyond pipe capacity under JSON escaping.
fields = {("\U0001f9ea" * 120) + str(i): i for i in range(64)}
try:
    list(daq.iter_daq_records(
        source, fields=fields, channel="\U0001f9ea" * 1024,
        experiment="\U0001f9ea" * 1024, linac="\U0001f9ea" * 1024,
        timeout_seconds=0.1,
    ))
except daq.DaqReadError as error:
    assert "timed out" in str(error), str(error)
else:
    raise AssertionError("A stalled interpreter did not time out")
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(native[0].path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        _, errors = process.communicate(timeout=5)
        assert process.returncode == 0, errors
    finally:
        # Kill both the harness and its native child if the regression returns.
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()


def test_early_close_reaps_child_and_releases_source_guard(native):
    pid_file = native[2] / "pid"
    native[1].update(STALL="1", PID_FILE=str(pid_file))
    with closing(read(native, batch_size=1)) as records:
        assert next(records).event_id == 100
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
    with native[0].open() as stream:
        assert stream.read() == b"immutable raw fixture"


def test_mounted_source_changes_are_detected(native):
    native[1]["MUTATE"] = "1"
    with pytest.raises(DatasetSourceChanged):
        list(read(native))


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"limit": -1}, {"timeout_seconds": 0}])
def test_invalid_bounds_are_rejected(native, kwargs):
    with pytest.raises(ValueError):
        list(read(native, **kwargs))


def test_protocol_message_bound(native, monkeypatch):
    import anomx.data.sources.daq as module

    monkeypatch.setattr(module, "_CHILD", 'import sys; sys.stdout.write("x" * (9*1024*1024))')
    with pytest.raises(DaqReadError, match="size bound"):
        list(read(native, timeout_seconds=2))
