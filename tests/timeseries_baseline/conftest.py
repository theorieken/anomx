"""Optional backends and isolated plotting for the native TimeSeries baseline."""

import importlib.util
from unittest.mock import patch

import pandas as pd
import pytest

PANDAS_30_OR_GREATER = int(pd.__version__.split(".")[0]) >= 3
IPYTHON_AVAILABLE = importlib.util.find_spec("IPython") is not None
POLARS_AVAILABLE = importlib.util.find_spec("polars") is not None
PLOTLY_AVAILABLE = importlib.util.find_spec("plotly") is not None


@pytest.fixture(scope="module")
def tmpdir_module(tmp_path_factory):
    return tmp_path_factory.mktemp("anomx-series")


@pytest.fixture
def mpl_safe_plotting():
    import matplotlib.pyplot as plt

    with patch("matplotlib.pyplot.show") as patched_show:
        yield patched_show
    plt.close("all")
