"""Native functionality must remain independent of the optional Darts package."""

import os
import subprocess
import sys
from pathlib import Path


def test_native_series_processing_and_metrics_without_darts():
    source = Path(__file__).resolve().parents[2] / "src"
    program = '''
import importlib.abc
import sys

class RejectOptionalDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"darts", "torch", "holidays", "matplotlib", "plotly", "xarray"}:
            raise AssertionError(f"Unexpected optional engine import: {fullname}")

sys.meta_path.insert(0, RejectOptionalDependencies())
import numpy as np
import pandas as pd
from anomx.data import TimeSeries, concatenate
from anomx.data.processing.transformers import Scaler
from anomx.metrics import rmse

values = np.arange(24, dtype=float).reshape(12, 2)
series = TimeSeries.from_times_and_values(
    pd.date_range("2026-10-01", periods=12, freq="h"),
    values,
    columns=["a", "b"],
    metadata={"source": "native boundary test"},
)
train, test = series.split_before(0.75)
assert concatenate([train, test]) == series
scaler = Scaler()
scaler.fit(train)
restored = scaler.inverse_transform(scaler.transform(test))
assert rmse(test, restored) < 1e-12
assert restored.metadata == series.metadata
assert type(restored).__module__ == "anomx.data.timeseries"
assert not any(name.split(".")[0] in {"darts", "torch"} for name in sys.modules)
'''
    result = subprocess.run(
        [sys.executable, "-c", program],
        env={**os.environ, "PYTHONPATH": str(source)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
