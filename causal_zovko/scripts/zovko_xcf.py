"""
Compute Zovko-style cross-correlation between volatility and relative limit price.

Default input/output locations (relative to EA_recherche root):
- input directory: causal_zovko/data
- outputs:
    - causal_zovko/figures/zovko_xcf_{side}_{bucket}.png
    - causal_zovko/results/xcorr_summary_{side}_{bucket}.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
CAUSAL_ZOVKO_DIR = SCRIPT_DIR.parent


def xcf(series_v: np.ndarray, series_d: np.ndarray, max_lag: int):
    # both arrays same length
    v = series_v - np.nanmean(series_v)
    d = series_d - np.nanmean(series_d)
    sv = np.nanstd(v)
    sd = np.nanstd(d)
    out = []
    for tau in range(-max_lag, max_lag + 1):
        if tau < 0:
            a = d[:tau]
            b = v[-tau:]
        elif tau > 0:
            a = d[tau:]
            b = v[:-tau]
        else:
            a = v
            b = d
        val = np.nanmean(a * b) / (sv * sd) if sv > 0 and sd > 0 else np.nan
        out.append(val)
    return np.array(out)


def phase_randomize_with_phases(x: np.ndarray, rand_phases: np.ndarray):
    """
    Phase-randomize a series using a pre-generated phase vector.
    This preserves the power spectrum and enables using the SAME
    random phase permutation across multiple series.
    """
    x = x - np.mean(x)
    n = len(x)
    fft = np.fft.rfft(x)
    mags = np.abs(fft)
    phases = np.angle(fft)

    new_phases = np.array(rand_phases, copy=True)
    new_phases[0] = phases[0]
    if len(new_phases) > 1:
        new_phases[-1] = phases[-1]

    new_fft = mags * np.exp(1j * new_phases)
    x_new = np.fft.irfft(new_fft, n=n)
    return x_new


def make_random_phases(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random phases for rfft length of a series of length n."""
    rfft_len = n // 2 + 1
    return rng.uniform(0, 2 * np.pi, size=rfft_len)


def load_bucket_pair(data_dir: Path, bucket: str, side: str):
    if bucket.endswith("events"):
        rlop_path = data_dir / f"rlop_{side}_{bucket}.csv"
        vol_path = data_dir / f"vol_{side}_{bucket}.csv"
        if not rlop_path.exists() or not vol_path.exists():
            raise FileNotFoundError(f"Could not find event bucket files for side={side}, bucket={bucket}")
        rlop = pd.read_csv(rlop_path)
        vol = pd.read_csv(vol_path)
        # align by bucket_id
        df = rlop.merge(vol, left_index=True, right_index=True, suffixes=("_d", "_v"))
        return df
    else:
        rlop_path = data_dir / f"rlop_{side}_{bucket}.csv"
        vol_path = data_dir / f"vol_{side}_{bucket}.csv"
        if not rlop_path.exists() or not vol_path.exists():
            raise FileNotFoundError(f"Could not find time bucket files for side={side}, bucket={bucket}")
        rlop = pd.read_csv(rlop_path, parse_dates=["time_paris"])
        vol = pd.read_csv(vol_path, parse_dates=["time_paris"])
        df = rlop.merge(vol, on="time_paris", suffixes=("_d", "_v"))
        return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(CAUSAL_ZOVKO_DIR / "data"))
    ap.add_argument("--bucket", default="10min")
    ap.add_argument("--side", type=str, default="ask", help="Side of the book to analyze.")
    ap.add_argument(
        "--d_cols",
        nargs="+",
        default=["delta_mean"],
        help="Delta columns (limit price) to average. Default: delta_mean",
    )
    ap.add_argument(
        "--v_cols",
        nargs="+",
        default=["vol_mean"],
        help="Volatility columns to average. Default: vol_mean",
    )
    ap.add_argument("--max_lag", type=int, default=10)
    ap.add_argument("--surrogates", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    # 使用模板作为默认值，而不是f-string
    ap.add_argument("--out_fig", default=str(CAUSAL_ZOVKO_DIR / "figures" / "zovko_xcf_{side}_{bucket}.png"))
    ap.add_argument("--out_csv", default=str(CAUSAL_ZOVKO_DIR / "results" / "xcorr_summary_{side}_{bucket}.csv"))
    args = ap.parse_args()

    # 在解析参数后，格式化输出路径
    out_fig_path = args.out_fig.format(side=args.side, bucket=args.bucket)
    out_csv_path = args.out_csv.format(side=args.side, bucket=args.bucket)

    data_dir = Path(args.data_dir)
    try:
        df = load_bucket_pair(data_dir, args.bucket, args.side)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # Forward-fill NaN values for requested columns
    d_cols = args.d_cols
    v_cols = args.v_cols
    if len(d_cols) != len(v_cols):
        print("Error: --d_cols and --v_cols must have the same number of columns.")
        return

    missing_cols = [c for c in d_cols + v_cols if c not in df.columns]
    if missing_cols:
        print(f"Error: missing columns: {missing_cols}")
        return

    for c in d_cols + v_cols:
        df[c] = df[c].ffill()

    series_d_list = [df[c].to_numpy(dtype=float) for c in d_cols]
    series_v_list = [df[c].to_numpy(dtype=float) for c in v_cols]

    # Observed XCF: average across series pairs
    xcf_obs_list = [
        xcf(series_v_list[i], series_d_list[i], args.max_lag)
        for i in range(len(series_d_list))
    ]
    xcf_obs = np.nanmean(np.vstack(xcf_obs_list), axis=0)

    # Null: phase randomize DELTA series with the SAME phases across all series
    rng = np.random.default_rng(args.seed)
    xcf_null = []
    for _ in range(args.surrogates):
        rand_phases = make_random_phases(len(series_d_list[0]), rng)
        xcf_null_list = []
        for i in range(len(series_d_list)):
            d_rand = phase_randomize_with_phases(series_d_list[i], rand_phases)
            xcf_null_list.append(xcf(series_v_list[i], d_rand, args.max_lag))
        xcf_null.append(np.nanmean(np.vstack(xcf_null_list), axis=0))
    xcf_null = np.array(xcf_null)

    lo = np.nanquantile(xcf_null, 0.025, axis=0)
    hi = np.nanquantile(xcf_null, 0.975, axis=0)

    lags = np.arange(-args.max_lag, args.max_lag + 1)

    # plot
    plt.figure(figsize=(10, 4))
    plt.fill_between(lags, lo, hi, color="lightgray", label="95% null band")
    plt.plot(lags, xcf_obs, marker="o", linewidth=1.5, label="XCF")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Lag (bucket steps)")
    plt.ylabel("XCF")
    plt.title(f"Zovko XCF (side={args.side}, bucket={args.bucket})")
    plt.legend()
    plt.tight_layout()
    Path(out_fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig_path, dpi=160)

    # save summary
    out = pd.DataFrame({"lag": lags, "xcf": xcf_obs, "null_lo": lo, "null_hi": hi})
    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv_path, index=False)

    print("saved:", out_fig_path, out_csv_path)


if __name__ == "__main__":
    main()
