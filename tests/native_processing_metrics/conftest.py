"""Headless plotting for native processing regression tests."""

from unittest.mock import patch

import pytest


@pytest.fixture
def mpl_safe_plotting():
    """Exercise plotting while preventing GUI windows and releasing figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with patch("matplotlib.pyplot.show") as patched_show:
        yield patched_show
    plt.close("all")
