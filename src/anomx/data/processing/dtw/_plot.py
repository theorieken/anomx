# Copyright 2019 Unit8 SA; additional contributions by Darts contributors.
# Licensed under the Apache License, Version 2.0 (see licenses/DARTS-APACHE-2.0.txt).
# Derived from Unit8 Darts, commit dcdde2414f07f13b3bbb095428d9c6cc41492588.
# Modified for Anomx: native namespace and integration with Anomx data APIs.

import numpy as np


def plot(
    self,
    new_plot: bool = False,
    show_series: bool = False,
    show_cost: bool = True,
    cost_cmap: str = "summer",
    args_path: dict | None = None,
    args_cost: dict | None = None,
    args_series1: dict | None = None,
    args_series2: dict | None = None,
):
    """
    Plot the warp path.

    Parameters
    ----------
    new_plot
        Boolean value indicating whether to spawn a new figure.
    show_series
        Boolean value indicating whether to additionally plot the two time series.
        Series1 will be plotted below the cost matrix and series2 to the left of the cost matrix.
    show_cost
        Boolean value indicating whether to additionally plot the cost matrix.
        Cost Matrix will be displayed as a heatmap.
        Useful for visualizing the effect of the window function.
    cost_cmap
        Colormap style for cost matrix plot
    args_path
        Some keyword arguments to pass to `plot()` function for warp path
    args_cost
        Some keyword arguments to pass to `imshow` function for cost matrix
    args_series1
        Some keyword arguments to pass to `plot()` method for series1
    args_series2
        Keyword arguments for the horizontal Matplotlib plot of series2.
    """
    from matplotlib import pyplot as plt

    args_path = dict(args_path or {})
    args_cost = dict(args_cost or {})
    args_series1 = dict(args_series1 or {})
    args_series2 = dict(args_series2 or {})

    if new_plot:
        fig = plt.figure()
    else:
        fig = plt.gcf()

    if show_series:
        gs = fig.add_gridspec(2, 2, width_ratios=[0.4, 0.6], height_ratios=[0.6, 0.4])

        warp = fig.add_subplot(gs[0, 1])
        left = fig.add_subplot(gs[0, 0])
        bottom = fig.add_subplot(gs[1, 1])

    else:
        warp = plt.gca()

    warp.title.set_text("Cost Matrix")
    warp.set_xlim([0, self.n])
    warp.set_ylim([0, self.m])
    warp.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    if show_cost:
        cost_matrix = self.cost.to_dense()
        cost_matrix = np.transpose(cost_matrix[1:, 1:])
        warp.imshow(
            cost_matrix,
            cmap=cost_cmap,
            interpolation="none",
            origin="lower",
            extent=[0, self.n, 0, self.m],
            **args_cost,
        )

    show_path = True
    if show_path:
        path = self.path()

        i_coords = path[:, 0] + 0.5
        j_coords = path[:, 1] + 0.5
        warp.plot(i_coords, j_coords, **args_path)

    if show_series:
        self.series1.plot(ax=bottom, **args_series1)
        args_series2.setdefault("label", list(self.series2.components))
        left.plot(self.series2.values(copy=False), self.series2.time_index, **args_series2)

        bottom.set_xlim([self.series1.start_time(), self.series1.end_time()])
        left.set_ylim([self.series2.start_time(), self.series2.end_time()])

        bottom.legend()
        bottom.set_title("")
        left.legend()
        left.set_title("")

    warp.set_aspect("auto")


def plot_alignment(
    self,
    new_plot: bool = False,
    series1_y_offset: float = 0,
    series2_y_offset: float = 0,
    components: tuple[str | int, str | int] = (0, 0),
    args_line: dict | None = None,
    args_series1: dict | None = None,
    args_series2: dict | None = None,
):
    """
    Plots the uni-variate component of each series,
    with lines between them indicating the alignment selected by the DTW algorithm.

    Parameters
    ----------
    new_plot
        whether to spawn a new Figure
    series2_y_offset
        offset series1 vertically for ease of viewing.
    series1_y_offset
        offset series2 vertically for ease of viewing.
    components
        Tuple of component index for series1 and component index for series2.
    args_line
        Some keywords arguments to pass to `plot()` method for line
    args_series1
        Some keyword arguments to pass to `plot()` method for series1
    args_series2
        Some keyword arguments to pass to `plot()` method for series2
    """
    from matplotlib import pyplot as plt

    args_line = dict(args_line or {})
    args_series1 = dict(args_series1 or {})
    args_series2 = dict(args_series2 or {})
    if new_plot:
        _, axis = plt.subplots()
    else:
        axis = plt.gca()

    series1 = self.series1
    series2 = self.series2

    (component1, component2) = components

    if not series1.is_univariate:
        series1 = series1.univariate_component(component1)
    if not series2.is_univariate:
        series2 = series2.univariate_component(component2)

    series1 += series1_y_offset
    series2 += series2_y_offset

    path = self.path()
    n = len(path)

    x_coords1 = series1.time_index.to_numpy()[path[:, 0]]
    x_coords2 = series2.time_index.to_numpy()[path[:, 1]]

    y_coords1 = series1.univariate_values()[path[:, 0]]
    y_coords2 = series2.univariate_values()[path[:, 1]]

    if series1.has_datetime_index:
        x_dtype = x_coords1.dtype
        x_nan = np.datetime64("NaT")
    else:
        x_dtype = np.float64
        x_nan = np.nan

    x_coords = np.empty(n * 3, dtype=x_dtype)
    y_coords = np.empty(n * 3, dtype=np.float64)

    x_coords[0::3] = x_coords1
    x_coords[1::3] = x_coords2
    x_coords[2::3] = x_nan

    y_coords[0::3] = y_coords1
    y_coords[1::3] = y_coords2
    y_coords[2::3] = np.nan

    axis.plot(x_coords, y_coords, **args_line)

    series1.plot(ax=axis, **args_series1)
    series2.plot(ax=axis, **args_series2)
