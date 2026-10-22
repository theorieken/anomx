# Copyright 2019 Unit8 SA; additional contributions by Darts contributors.
# Licensed under the Apache License, Version 2.0 (see licenses/DARTS-APACHE-2.0.txt).
# Derived from Unit8 Darts, commit dcdde2414f07f13b3bbb095428d9c6cc41492588.
# Modified for Anomx: native namespace and integration with Anomx data APIs.

"""Native metric visualizations with optional Matplotlib dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from anomx._series_typing import TimeSeriesLike
from anomx._series_utils.likelihood_models.base import likelihood_component_names, quantile_names
from anomx._series_utils.ts_utils import get_single_series
from anomx.metrics.metrics import _NP_TRAPEZOID_FN, _tolerance_coverages
from anomx.metrics.utils import _get_tolerance_levels

if TYPE_CHECKING:
    from matplotlib.axes import Axes

def plot_tolerance_curve(
    actual_series: TimeSeriesLike,
    pred_series: TimeSeriesLike,
    intersect: bool = True,
    min_tolerance: float = 0.0,
    max_tolerance: float = 1.0,
    step: float = 0.01,
    q: float | list[float] | None = None,
    n_jobs: int = 1,
    verbose: bool = False,
    fig_size: tuple[int, int] | None = None,
    axis: Axes | None = None,
) -> Axes:
    """
    Plots the Tolerance Curve for evaluating forecast alignment.

    The tolerance curve shows, for each tolerance level (as a percentage of the actual series range),
    what fraction of prediction points fall within that tolerance. The scale is
    half the observed range, matching the AUTC metric.

    For multivariate series, one curve is plotted per component.

    The Area Under Tolerance Curve (AUTC) can be computed using :func:`~anomx.metrics.metrics.autc`.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    min_tolerance
        The minimum tolerance level as a fraction of the series half-range. Default is 0.0 (0%).
    max_tolerance
        The maximum tolerance level as a fraction of the series half-range. Default is 1.0 (100%).
    step
        The step size between tolerance levels. Default is 0.01 (1%).
        For example, with defaults, tolerances are [0.0, 0.01, 0.02, ..., 1.0].
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.
    fig_size
        The size of the figure to be displayed.
    axis
        Optionally, an axis object to plot on.

    Returns
    -------
    matplotlib.axes.Axes
        The populated axes, suitable for further styling or saving its figure.

    See Also
    --------
    :func:`~anomx.metrics.metrics.autc` : Compute the Area Under Tolerance Curve as a single score.
    """
    import matplotlib.pyplot as plt

    # get coverage for different tolerance levels
    coverages = _tolerance_coverages(
        actual_series=actual_series,
        pred_series=pred_series,
        intersect=intersect,
        min_tolerance=min_tolerance,
        max_tolerance=max_tolerance,
        step=step,
        q=q,
        component_reduction=None,
        series_reduction=np.nanmean,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    tolerances = _get_tolerance_levels(
        min_tolerance=min_tolerance,
        max_tolerance=max_tolerance,
        step=step,
    )

    # -> (n coverages, n components * n quantiles)
    if len(coverages.shape) == 1:
        coverages = coverages.reshape(-1, 1)

    series_ = get_single_series(actual_series)
    if q is None or isinstance(q, float):
        coverage_labels = series_.components.tolist()
    else:
        coverage_labels = likelihood_component_names(series_.components, quantile_names(q))

    if axis is None:
        _, axis = plt.subplots(figsize=fig_size)

    tolerances_perc = tolerances * 100.0
    for i, comp_name in enumerate(coverage_labels):
        coverages_comp = coverages[:, i]
        coverages_perc = coverages_comp * 100.0

        auc = _NP_TRAPEZOID_FN(coverages_comp, tolerances)
        label = f"{comp_name} (AUTC = {auc:.3f})"
        axis.step(
            tolerances_perc,
            coverages_perc,
            where="post",
            linewidth=2,
            label=label,
        )
        if len(coverage_labels) == 1:
            axis.fill_between(
                tolerances_perc,
                coverages_perc,
                alpha=0.25,
                color=axis.get_lines()[-1].get_color(),
                step="post",
            )

    # add reference line for perfect predictions
    axis.axhline(y=100, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    axis.set_xlim(0, 100)
    axis.set_ylim(0, 105)
    axis.set_xlabel("Tolerance (% of half-range)")
    axis.set_ylabel("Coverage (%)")
    axis.set_title("Tolerance Curve")
    axis.grid(True, alpha=0.3)
    axis.legend()

    return axis
