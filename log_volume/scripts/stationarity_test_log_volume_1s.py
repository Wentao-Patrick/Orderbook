from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "sanofi_book_snapshots_1s.parquet"
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent / "results"
TIME_COL_CANDIDATES = [
    "timestamp",
    "time",
    "date",
    "datetime",
    "trade_time_paris",
    "ts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stationarity diagnostics for 1-second log-volume level-1 snapshots."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to sanofi_book_snapshots_1s.parquet.",
    )
    parser.add_argument(
        "--sides",
        nargs="+",
        choices=["bid", "ask"],
        default=["bid", "ask"],
        help="Book sides to analyze.",
    )
    parser.add_argument(
        "--tz",
        type=str,
        default="Europe/Paris",
        help="Timezone used for plotting.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level used in the summary verdict.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=900,
        help="Rolling window length in observations for mean/std visualization.",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "log_volume_1s_stationarity_overview.png",
        help="PNG path for the overview figure.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "log_volume_1s_stationarity_tests.csv",
        help="CSV path for test results.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "log_volume_1s_stationarity_summary.md",
        help="Markdown path for the summary report.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_snapshots(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Unable to read parquet input. Install a parquet engine such as "
            "`pyarrow` or `fastparquet`."
        ) from exc


def detect_time_col(df: pd.DataFrame) -> str:
    for col in TIME_COL_CANDIDATES:
        if col in df.columns:
            return col
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            return col
    if isinstance(df.index, pd.DatetimeIndex):
        return "__index__"
    raise ValueError("No datetime column or DatetimeIndex found in the snapshots data.")


def to_datetime_index(df: pd.DataFrame, time_col: str, tz: str) -> pd.DataFrame:
    if time_col == "__index__":
        indexed = df.copy()
        idx = pd.DatetimeIndex(indexed.index)
    else:
        indexed = df.copy()
        idx = pd.to_datetime(indexed[time_col], errors="coerce")
        indexed = indexed.drop(columns=[time_col])

    indexed.index = idx
    indexed = indexed[~indexed.index.isna()].sort_index()
    indexed.index.name = "time"
    if indexed.index.tz is None:
        indexed.index = indexed.index.tz_localize(tz)
    else:
        indexed.index = indexed.index.tz_convert(tz)
    return indexed


def build_log_volume_series(df: pd.DataFrame, side: str) -> pd.Series:
    col = f"{side}volume1"
    if col not in df.columns:
        raise ValueError(f"Missing required column `{col}` in input data.")

    volume = pd.to_numeric(df[col], errors="coerce")
    positive = volume.where(volume > 0)
    log_series = np.log(positive)
    log_series.name = f"log_{side}volume1"
    return log_series.replace([np.inf, -np.inf], np.nan).dropna()


def run_adf(series: pd.Series, regression: str) -> dict[str, object]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stat, pvalue, usedlag, nobs, crit, icbest = adfuller(
            series.to_numpy(),
            regression=regression,
            autolag="AIC",
        )
    return {
        "test": "ADF",
        "regression": regression,
        "statistic": float(stat),
        "p_value": float(pvalue),
        "used_lags": int(usedlag),
        "nobs": int(nobs),
        "icbest": float(icbest),
        "critical_1pct": float(crit["1%"]),
        "critical_5pct": float(crit["5%"]),
        "critical_10pct": float(crit["10%"]),
        "warning": " | ".join(str(item.message) for item in caught),
        "null_hypothesis": "unit root (non-stationary)",
    }


def run_kpss(series: pd.Series, regression: str) -> dict[str, object]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stat, pvalue, usedlags, crit = kpss(
            series.to_numpy(),
            regression=regression,
            nlags="auto",
        )
    return {
        "test": "KPSS",
        "regression": regression,
        "statistic": float(stat),
        "p_value": float(pvalue),
        "used_lags": int(usedlags),
        "nobs": int(series.shape[0]),
        "icbest": np.nan,
        "critical_1pct": float(crit["1%"]),
        "critical_5pct": float(crit["5%"]),
        "critical_10pct": float(crit["10%"]),
        "warning": " | ".join(str(item.message) for item in caught),
        "null_hypothesis": (
            "stationary around a constant"
            if regression == "c"
            else "trend-stationary"
        ),
    }


def classify_stationarity(adf_p: float, kpss_p: float, alpha: float) -> str:
    adf_reject = adf_p < alpha
    kpss_reject = kpss_p < alpha
    if adf_reject and not kpss_reject:
        return "evidence supports stationarity"
    if not adf_reject and kpss_reject:
        return "evidence supports non-stationarity"
    if adf_reject and kpss_reject:
        return "mixed: likely structural change or time-varying moments"
    return "inconclusive"


def analyze_series(series: pd.Series, side: str, alpha: float) -> tuple[pd.DataFrame, list[str]]:
    records: list[dict[str, object]] = []
    summary_lines: list[str] = []
    transforms = {
        "level": series,
        "diff1": series.diff().dropna(),
    }

    for transform_name, current in transforms.items():
        if current.empty:
            continue

        adf_c = run_adf(current, regression="c")
        adf_ct = run_adf(current, regression="ct")
        kpss_c = run_kpss(current, regression="c")
        kpss_ct = run_kpss(current, regression="ct")

        verdict_level = classify_stationarity(adf_c["p_value"], kpss_c["p_value"], alpha)
        verdict_trend = classify_stationarity(adf_ct["p_value"], kpss_ct["p_value"], alpha)

        for result in (adf_c, adf_ct, kpss_c, kpss_ct):
            reject_null = bool(result["p_value"] < alpha)
            records.append(
                {
                    "side": side,
                    "transform": transform_name,
                    "series_name": current.name,
                    "sample_size": int(current.shape[0]),
                    "mean": float(current.mean()),
                    "std": float(current.std(ddof=1)),
                    "alpha": alpha,
                    "reject_null": reject_null,
                    **result,
                    "verdict_constant": verdict_level,
                    "verdict_trend": verdict_trend,
                }
            )

        summary_lines.append(
            (
                f"- `{side}` `{transform_name}`: level-stationary view -> {verdict_level} "
                f"(ADF(c) p={adf_c['p_value']:.4g}, KPSS(c) p={kpss_c['p_value']:.4g}); "
                f"trend-stationary view -> {verdict_trend} "
                f"(ADF(ct) p={adf_ct['p_value']:.4g}, KPSS(ct) p={kpss_ct['p_value']:.4g})."
            )
        )

    return pd.DataFrame.from_records(records), summary_lines


def plot_overview(
    series_map: dict[str, pd.Series],
    rolling_window: int,
    output_path: Path,
) -> None:
    ensure_parent(output_path)
    nrows = len(series_map)
    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(18, 5 * nrows), squeeze=False)

    for row_idx, (side, series) in enumerate(series_map.items()):
        rolling = series.rolling(window=rolling_window, min_periods=max(30, rolling_window // 4))
        rolling_mean = rolling.mean()
        rolling_std = rolling.std()
        date_fmt = mdates.DateFormatter("%H:%M", tz=series.index.tz)

        ax_level = axes[row_idx, 0]
        ax_level.plot(series.index, series.to_numpy(), lw=0.7, color="tab:blue")
        ax_level.axhline(series.mean(), color="black", ls="--", lw=1.0, label="full-sample mean")
        ax_level.set_title(f"{side.upper()} log-volume1")
        ax_level.set_ylabel(series.name)
        ax_level.legend(loc="upper right")
        ax_level.xaxis.set_major_formatter(date_fmt)

        ax_mean = axes[row_idx, 1]
        ax_mean.plot(rolling_mean.index, rolling_mean.to_numpy(), lw=1.0, color="tab:orange")
        ax_mean.axhline(series.mean(), color="black", ls="--", lw=1.0, label="full-sample mean")
        ax_mean.set_title(f"{side.upper()} rolling mean ({rolling_window} obs)")
        ax_mean.legend(loc="upper right")
        ax_mean.xaxis.set_major_formatter(date_fmt)

        ax_std = axes[row_idx, 2]
        ax_std.plot(rolling_std.index, rolling_std.to_numpy(), lw=1.0, color="tab:green")
        ax_std.axhline(series.std(ddof=1), color="black", ls="--", lw=1.0, label="full-sample std")
        ax_std.set_title(f"{side.upper()} rolling std ({rolling_window} obs)")
        ax_std.legend(loc="upper right")
        ax_std.xaxis.set_major_formatter(date_fmt)

        for ax in (ax_level, ax_mean, ax_std):
            ax.grid(alpha=0.2)
            ax.tick_params(axis="x", rotation=30)

    fig.suptitle("1-second log-volume stationarity diagnostics", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    summary_output: Path,
    input_path: Path,
    figure_output: Path,
    csv_output: Path,
    rolling_window: int,
    series_map: dict[str, pd.Series],
    summary_lines: list[str],
) -> None:
    ensure_parent(summary_output)
    header = [
        "# Log-volume 1s stationarity report",
        "",
        f"- Input: `{input_path}`",
        f"- Figure: `{figure_output}`",
        f"- CSV: `{csv_output}`",
        f"- Rolling window for visualization: `{rolling_window}` observations",
        "",
        "## Sample information",
    ]

    sample_lines = []
    for side, series in series_map.items():
        sample_lines.append(
            (
                f"- `{side}`: {len(series)} observations from "
                f"{series.index.min()} to {series.index.max()}, "
                f"mean={series.mean():.6f}, std={series.std(ddof=1):.6f}."
            )
        )

    lines = header + sample_lines + ["", "## Test summary"] + summary_lines + [""]
    summary_output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    df_raw = read_snapshots(args.input)
    time_col = detect_time_col(df_raw)
    df = to_datetime_index(df_raw, time_col=time_col, tz=args.tz)

    series_map: dict[str, pd.Series] = {}
    results_frames: list[pd.DataFrame] = []
    summary_lines: list[str] = []

    for side in args.sides:
        series = build_log_volume_series(df, side)
        series_map[side] = series
        result_df, lines = analyze_series(series, side=side, alpha=args.alpha)
        results_frames.append(result_df)
        summary_lines.extend(lines)

    all_results = pd.concat(results_frames, ignore_index=True)

    ensure_parent(args.csv_output)
    all_results.to_csv(args.csv_output, index=False)
    plot_overview(series_map, rolling_window=args.rolling_window, output_path=args.figure_output)
    write_summary(
        summary_output=args.summary_output,
        input_path=args.input,
        figure_output=args.figure_output,
        csv_output=args.csv_output,
        rolling_window=args.rolling_window,
        series_map=series_map,
        summary_lines=summary_lines,
    )

    print(f"Saved tests to: {args.csv_output}")
    print(f"Saved figure to: {args.figure_output}")
    print(f"Saved summary to: {args.summary_output}")


if __name__ == "__main__":
    main()
