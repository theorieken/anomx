"""Local detuning experiments: native forecasts, reconstruction, and representations.

Run ``python tmp.py`` with the client connection configured below.
Use ``--include-torch`` for a small neural forecast, ``--show`` for Matplotlib,
and ``--refresh`` to bypass cached observations. No Darts installation is needed.
The source channel remains irregular; this example explicitly averages into
one-second bins and selects the longest complete interval rather than filling
unobserved periods. Change CONFIGURATIONS to explore other native models.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from getpass import getpass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from anomx import Dataset, GenericModel, LocalExecutor, TimeSeries
from anomx.data.processing.transformers import Scaler
from anomx.detection import NormScorer, QuantileDetector
from anomx.integrations.platform import PlatformClient
from anomx.models import LinearRegressionModel, NaiveSeasonal, TorchForecastingModel

DATASET_REFERENCE = "data_dataset-486091c4-9de9-46d7-a1ee-74c1a4de7cd9"
CONFIGURATIONS: dict[str, Callable[[], Any]] = {
    "persistence": lambda: NaiveSeasonal(K=1),
    "seasonal-60": lambda: NaiveSeasonal(K=60),  # chosen baseline period, in observations
    "autoregression": lambda: LinearRegressionModel(lags=24, ridge=1.0),
}


def load_detuning(args: argparse.Namespace) -> pd.DataFrame:
    """Read a platform dataset, or an explicitly selected local snapshot."""
    if args.snapshot:
        print(f"anomx · explicit local snapshot: {args.snapshot} (no freshness check)")
        dataset = Dataset.from_dataframe(pd.read_parquet(args.snapshot))
        if args.start or args.end:
            dataset = dataset.select(start=args.start, end=args.end)
        if args.server_downsample:
            dataset = dataset.downsample(args.frequency)
        frame = dataset.to_dataframe(max_rows=args.max_rows)
        frame.attrs["anomx_selection"] = {
            **frame.attrs.get("anomx_selection", {}),
            "snapshot": str(args.snapshot),
        }
        return frame
    client = PlatformClient(
        url=os.environ.get("ANOMX_URL", "https://anomalies.msktools.desy.de"),
        token=os.environ.get("ANOMX_TOKEN") or getpass("Anomx API token: "),
        timeout=60,
    )
    dataset = Dataset(
        anomx_reference=args.dataset,
        client=client,
        cache=True,
        cache_ttl=args.cache_ttl,
    )
    if args.start or args.end:
        dataset = dataset.select(start=args.start, end=args.end)
    if args.server_downsample:
        dataset = dataset.downsample(args.frequency, aggregation="mean")
    policy = (
        "reuse saved snapshot; --refresh checks out new observations"
        if args.cache_ttl is None
        else f"complete snapshots expire after {args.cache_ttl:g}s without a content validator"
    )
    print(f"anomx · cache: {policy}; interrupted downloads resume automatically")
    frame = dataset.to_dataframe(refresh=args.refresh, max_rows=args.max_rows)
    print(
        f"anomx · {dataset.name}: {len(frame):,} observations · "
        f"cache={dataset.cache_info['status']} · "
        f"reused={dataset.cache_info.get('cached_rows', 0):,} · "
        f"downloaded={dataset.cache_info.get('downloaded_rows', 0):,}"
    )
    frame.attrs["dataset_reference"] = dataset.reference
    frame.attrs["dataset_version"] = dataset.describe()["version"]
    return frame


def prepare_series(frame: pd.DataFrame, frequency: str = "1s") -> TimeSeries:
    """Create one complete regular interval and retain its source provenance."""
    numeric = frame.select_dtypes(include="number")
    if numeric.shape[1] != 1 or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("This example expects one numeric detuning channel with timestamps.")
    regular = numeric.sort_index().resample(frequency, origin="epoch").mean()
    complete = regular.notna().all(axis=1)
    groups = complete.ne(complete.shift()).cumsum()
    intervals = [part for _, part in regular[complete].groupby(groups[complete])]
    if not intervals:
        raise ValueError("There are no complete detuning bins in this selection.")
    selected = max(intervals, key=len).copy()
    if len(selected) < 300:
        raise ValueError("Choose a source interval with at least 300 complete one-second bins.")
    print(
        f"anomx · {frequency} mean: {len(selected):,}/{len(regular):,} bins "
        "in longest complete interval; no gap filling"
    )
    selected.columns = ["detuning"]
    return Dataset.from_dataframe(
        selected,
        name="Detuning",
        metadata={
            "aggregation": f"{frequency} mean",
            "selection": frame.attrs.get("anomx_selection"),
            "gap_policy": "longest complete interval",
            "dataset_reference": frame.attrs.get("dataset_reference", DATASET_REFERENCE),
            "dataset_version": frame.attrs.get("dataset_version"),
            "source_rows": len(frame),
            "source_start": str(frame.index.min()),
            "source_end": str(frame.index.max()),
        },
    ).to_timeseries(progress=False)


def run_experiments(series: TimeSeries, args: argparse.Namespace) -> list[Any]:
    """Compare three forecasts, then fit window reconstruction and representation."""
    executor = LocalExecutor()
    horizon = min(300, len(series) // 5)
    models = dict(CONFIGURATIONS)
    if args.include_torch:
        models["torch-forecast"] = lambda: TorchForecastingModel(
            input_chunk_length=24,
            n_epochs=20,
            random_state=42,
            verbose=True,
        )
    results = []
    for name, make_model in models.items():
        result = executor.forecast(
            make_model(),
            series,
            horizon=horizon,
            name=name,
            transformer=Scaler(StandardScaler()),
        )
        result.save(args.output / name, plot=True)
        results.append(result)

    # 24-point windows: split the raw series first, so no window
    # shares observations across the training/test boundary.
    train, test = series[:-horizon], series[-horizon:]
    width = 24
    train_windows = np.lib.stride_tricks.sliding_window_view(train.univariate_values(), width)
    test_windows = np.lib.stride_tricks.sliding_window_view(test.univariate_values(), width)
    scaler = StandardScaler().fit(train_windows)
    train_windows, test_windows = scaler.transform(train_windows), scaler.transform(test_windows)
    columns = [f"lag_{i}" for i in range(width)]
    training = pd.DataFrame(train_windows, columns=columns, index=train.time_index[width - 1 :])
    testing = pd.DataFrame(test_windows, columns=columns, index=test.time_index[width - 1 :])
    reconstruction = GenericModel(
        PCA(n_components=3),
        family="reconstruction",
        predict_fn=lambda model, x: model.inverse_transform(model.transform(x)),
    )
    result = executor.fit_predict(reconstruction, training, testing, name="pca-reconstruction")
    result.metadata["scale"] = "standardized windows; scaler fitted on training only"
    result.save(args.output / result.name, plot=True)
    results.append(result)

    # Calibrate on training reconstruction errors only. Labels here are model
    # threshold exceedances, not independently verified physical anomalies.
    scorer = NormScorer(ord=2)
    training_series = TimeSeries.from_dataframe(training)
    training_prediction = TimeSeries.from_dataframe(reconstruction.predict(training))
    calibration = scorer.score_from_prediction(training_series, training_prediction)
    detector = QuantileDetector(high_quantile=0.99).fit(calibration)
    scores = scorer.score_from_prediction(
        TimeSeries.from_dataframe(testing), TimeSeries.from_dataframe(result.prediction)
    )
    labels = detector.detect(scores)
    scores.to_dataframe().to_csv(args.output / "reconstruction-scores.csv")
    labels.to_dataframe().to_csv(args.output / "threshold-exceedances.csv")
    print(f"anomx · PCA threshold exceedances: {int(labels.values().sum())}/{len(labels)} windows")

    representation = GenericModel(PCA(n_components=2), family="representation")
    result = executor.fit_predict(
        representation, training, testing, family="representation", name="pca-representation"
    )
    result.save(args.output / result.name, plot=True)
    results.append(result)
    return results


def main() -> None:
    """Load, train locally, write receipts/plots, and optionally open the figures."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_REFERENCE)
    parser.add_argument(
        "--snapshot", type=Path, help="Explicit offline parquet snapshot instead of the API"
    )
    parser.add_argument(
        "--cache-ttl",
        type=float,
        default=None,
        help="Expire complete snapshots after N seconds; default reuses until --refresh",
    )
    parser.add_argument("--refresh", action="store_true", help="Restart the selected download")
    parser.add_argument("--start", help="Inclusive ISO timestamp; defaults to dataset start")
    parser.add_argument("--end", help="Exclusive ISO timestamp; defaults to dataset end")
    parser.add_argument(
        "--server-downsample",
        action="store_true",
        help="Request --frequency means from the API (requires updated platform)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional preview limit; default loads all rows in the selected dataset/window",
    )
    parser.add_argument(
        "--all-rows", action="store_true", help="Download the complete dataset instead of a prefix"
    )
    parser.add_argument(
        "--frequency", default="1s", help="Explicit resampling interval, default 1s"
    )
    parser.add_argument("--include-torch", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "artifacts" / "detuning-local"
    )
    args = parser.parse_args()
    if args.all_rows:
        args.max_rows = None
    selected_rows = (
        "all selected rows (no cap)" if args.max_rows is None else f"first {args.max_rows:,} rows"
    )
    location = "server" if args.server_downsample else "local"
    print(f"anomx · selection: {selected_rows} · {location} resampling={args.frequency}")
    if args.start or args.end:
        print(
            f"anomx · time window: [{args.start or 'dataset start'}, {args.end or 'dataset end'})"
        )
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    results = run_experiments(prepare_series(load_detuning(args), args.frequency), args)
    summary = [
        {"name": result.name, "family": result.family, **result.metrics} for result in results
    ]
    print(pd.DataFrame(summary).fillna("—").to_string(index=False))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"anomx · results saved to {args.output.resolve()}")
    if args.show:
        from matplotlib import pyplot as plt

        for result in results:
            result.plot()
        plt.show()


if __name__ == "__main__":
    main()
