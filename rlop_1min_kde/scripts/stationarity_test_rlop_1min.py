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
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent / "results"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "causal_zovko" / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stationarity diagnostics for 1-minute mean RLOP series."
    )
    parser.add_argument(
        "--bid-input",
        type=Path,
        default=DEFAULT_INPUT_DIR / "rlop_bid_1min.csv",
        help="Path to rlop_bid_1min.csv.",
    )
    parser.add_argument(
        "--ask-input",
        type=Path,
        default=DEFAULT_INPUT_DIR / "rlop_ask_1min.csv",
        help="Path to rlop_ask_1min.csv.",
    )
    parser.add_argument(
        "--sides",
        nargs="+",
        choices=["bid", "ask"],
        default=["bid", "ask"],
        help="Book sides to analyze.",
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
        default=15,
        help="Rolling window length in observations for mean/std visualization.",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "rlop_1min_stationarity_overview.png",
        help="PNG path for the overview figure.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "rlop_1min_stationarity_tests.csv",
        help="CSV path for test results.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "rlop_1min_stationarity_summary.md",
        help="Markdown path for the summary report.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_rlop_series(path: Path, side: str) -> pd.Series:
    df = pd.read_csv(path)
    if "time_paris" not in df.columns or "delta_mean" not in df.columns:
        raise ValueError(f"`{path}` must contain `time_paris` and `delta_mean` columns.")

    index = pd.to_datetime(df["time_paris"], errors="coerce")
    values = pd.to_numeric(df["delta_mean"], errors="coerce")

    series = pd.Series(values.to_numpy(), index=index, name=f"rlop_{side}_mean_1min")
    series = series[~series.index.isna()].sort_index()
    return series.dropna()


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
            records.append(
                {
                    "side": side,
                    "transform": transform_name,
                    "series_name": current.name,
                    "sample_size": int(current.shape[0]),
                    "mean": float(current.mean()),
                    "std": float(current.std(ddof=1)),
                    "alpha": alpha,
                    "reject_null": bool(result["p_value"] < alpha),
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
    fig, axes = plt.subplots(nrows=len(series_map), ncols=3, figsize=(18, 5 * len(series_map)), squeeze=False)

    for row_idx, (side, series) in enumerate(series_map.items()):
        rolling = series.rolling(window=rolling_window, min_periods=max(5, rolling_window // 3))
        rolling_mean = rolling.mean()
        rolling_std = rolling.std()
        date_fmt = mdates.DateFormatter("%H:%M", tz=series.index.tz)

        ax_level = axes[row_idx, 0]
        ax_level.plot(series.index, series.to_numpy(), lw=0.9, color="tab:blue")
        ax_level.axhline(series.mean(), color="black", ls="--", lw=1.0, label="full-sample mean")
        ax_level.set_title(f"{side.upper()} RLOP mean (1min)")
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

    fig.suptitle("1-minute mean RLOP stationarity diagnostics", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    summary_output: Path,
    input_paths: dict[str, Path],
    figure_output: Path,
    csv_output: Path,
    rolling_window: int,
    series_map: dict[str, pd.Series],
    summary_lines: list[str],
) -> None:
    ensure_parent(summary_output)
    header = [
        "# RLOP 1min stationarity report",
        "",
        f"- Bid input: `{input_paths['bid']}`",
        f"- Ask input: `{input_paths['ask']}`",
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

    input_paths = {
        "bid": args.bid_input,
        "ask": args.ask_input,
    }

    series_map: dict[str, pd.Series] = {}
    result_frames: list[pd.DataFrame] = []
    summary_lines: list[str] = []

    for side in args.sides:
        series = load_rlop_series(input_paths[side], side=side)
        series_map[side] = series
        result_df, lines = analyze_series(series, side=side, alpha=args.alpha)
        result_frames.append(result_df)
        summary_lines.extend(lines)

    all_results = pd.concat(result_frames, ignore_index=True)

    ensure_parent(args.csv_output)
    all_results.to_csv(args.csv_output, index=False)
    plot_overview(series_map, rolling_window=args.rolling_window, output_path=args.figure_output)
    write_summary(
        summary_output=args.summary_output,
        input_paths=input_paths,
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
