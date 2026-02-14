# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import chi2
from scipy.optimize import brentq


PARQUET_PATH = r"C:\Users\Wentao\Desktop\EA_recherche\sanofi_book_snapshots_1s.parquet"
# Choose side: "bid" or "ask"
VOLUME_SIDE = "bid"
VALUE_COL_CANDIDATES = ["logvolume"]
TIME_COL_CANDIDATES = ["timestamp", "time", "date", "datetime", "trade_time_paris", "ts"]

# ====== TEST CONFIG ======
ALPHA = 0.01          # 1% significance level (consistent with the template)
K_BINS = 20           # Number of bins k (suggested 8~12)
USE_QUANTILE_BINS = True  # Use reference window quantiles for bins (strongly recommended)
WINDOW_SIZE = 3600    # rolling window size (samples). 1s data -> 3600 = 1 hour
STEP_SIZE = 30        # step size between windows (samples). set 1 for every-second roll.


# =========================================================
# 0) Parquet loader + column detection
# =========================================================
def _read_parquet(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as e:
        raise RuntimeError(
            "无法读取 parquet。请先安装 parquet 引擎，例如：\n"
            "  pip install pyarrow\n"
            "或：pip install fastparquet\n\n"
            f"原始报错：{repr(e)}"
        )


def _detect_time_col(df: pd.DataFrame) -> str:
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    if isinstance(df.index, pd.DatetimeIndex):
        return "__index__"
    raise ValueError("找不到时间列。请把时间列名加入 TIME_COL_CANDIDATES。")


def _detect_value_col(df: pd.DataFrame, side: str) -> str:
    side = side.strip().lower()
    if side not in ("bid", "ask"):
        raise ValueError("VOLUME_SIDE must be 'bid' or 'ask'")

    side_log_candidates = [
        f"log_{side}volume1",
        f"log_{side}_volume1",
        f"log_{side}volume",
    ]
    for c in side_log_candidates + VALUE_COL_CANDIDATES:
        if c in df.columns:
            return c

    base_col = f"{side}volume1"
    if base_col in df.columns:
        df[f"log_{base_col}"] = np.log(pd.to_numeric(df[base_col], errors="coerce"))
        return f"log_{base_col}"

    raise ValueError(
        f"Missing logvolume for {side}: no log_* column and no {base_col} to compute."
    )


def _to_datetime_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if time_col == "__index__":
        dfi = df.copy()
        idx = dfi.index
    else:
        dfi = df.copy()
        idx = pd.to_datetime(dfi[time_col], errors="coerce")
        dfi = dfi.drop(columns=[time_col])

    dfi.index = idx
    dfi = dfi[~dfi.index.isna()].sort_index()
    dfi.index.name = "time"
    return dfi


def _ensure_paris_tz(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        return idx.tz_localize("Europe/Paris")
    return idx.tz_convert("Europe/Paris")


# =========================================================
# 1) Discretization: logvolume -> bins -> categorical codes 0..k-1
# =========================================================
def make_bins_from_reference(x_ref: np.ndarray, k: int) -> np.ndarray:
    """
    Return bin edges of length k+1 (including -inf, +inf) using quantiles from reference.
    This keeps q_i away from zero as much as possible.
    """
    x_ref = x_ref[np.isfinite(x_ref)]
    if x_ref.size < max(50, 5 * k):
        raise ValueError(f"reference 样本太少（{x_ref.size}），不适合 k={k} 分箱。")

    if USE_QUANTILE_BINS:
        qs = np.linspace(0.0, 1.0, k + 1)
        edges = np.quantile(x_ref, qs)
        # De-duplicate to avoid repeated values making edges non-strictly increasing
        print("Quantile edges before deduplication:", edges)
        edges = np.unique(edges)
        print("Quantile edges after deduplication:", edges)
        if edges.size < 3:
            raise ValueError("reference 数据重复值太多，分位数边界退化。请减小 k 或改用等宽分箱。")
        # Rebuild to ensure full coverage: [-inf, internal..., +inf]
        internal = edges[1:-1]
        bins = np.concatenate(([-np.inf], internal, [np.inf]))
    else:
        lo, hi = np.quantile(x_ref, [0.005, 0.995])
        bins = np.linspace(lo, hi, k + 1)
        bins[0] = -np.inf
        bins[-1] = np.inf

    return bins


def codes_from_bins(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """
    Map each x into {0,...,k-1} by bins.
    """
    x = x[np.isfinite(x)]
    # np.digitize returns 1..k, so subtract 1 -> 0..k-1
    return np.digitize(x, bins[1:-1], right=False)


def empirical_probs_from_codes(codes: np.ndarray, k: int) -> np.ndarray:
    cnt = np.bincount(codes, minlength=k).astype(float)
    if cnt.sum() == 0:
        raise ValueError("Empty window after discretization.")
    return cnt / cnt.sum()


def kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    if np.any(q[mask] == 0):
        return float("inf")
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


# =========================================================
# 2) Thresholds 
# =========================================================
def thresh_asymptotic_one_sample(k: int, n: int, alpha: float = 0.01) -> float:
    # 2n KL ~ chi^2_{k-1}
    return chi2.ppf(1 - alpha, df=k - 1) / (2 * n)


def thresh_asymptotic_two_sample_equal(k: int, n: int, alpha: float = 0.01) -> float:
    # when n=m, 2*(nm/(n+m))*KL = n*KL ~ chi^2_{k-1}
    return chi2.ppf(1 - alpha, df=k - 1) / n


def thresh_M3(k: int, n: int, alpha: float = 0.01) -> float:
    # one-sample finite-sample bound
    def M3(x):
        return np.exp(-n * x) * (np.e * n * x / (k - 1)) ** (k - 1)

    x0 = (k - 1) / n * 1.0000001
    x1 = x0
    while M3(x1) > alpha:
        x1 *= 1.5
        if x1 > 50:
            break
    return brentq(lambda x: M3(x) - alpha, x0, x1)


def thresh_Mf3_star_equal(k: int, n: int, alpha: float = 0.01) -> float:
    # two-sample equal-size finite-sample bound
    def Mf3s(x):
        return np.exp(-n * x) * (np.e * n * x / (2 * (k - 1))) ** (2 * (k - 1))

    x0 = (k - 1) / n * 1.0000001
    x1 = x0
    while Mf3s(x1) > alpha:
        x1 *= 1.5
        if x1 > 50:
            break
    f0 = Mf3s(x0) - alpha
    f1 = Mf3s(x1) - alpha
    if f0 * f1 > 0:
        return None
    return brentq(lambda x: Mf3s(x) - alpha, x0, x1)


# =========================================================
# 3) Rolling KL vs first window
# =========================================================
def rolling_kl_vs_first(df: pd.DataFrame, value_col: str, k_bins: int, window_size: int, step: int):
    x = pd.to_numeric(df[value_col], errors="coerce")
    dfw = df.copy()
    dfw[value_col] = x
    dfw = dfw.dropna(subset=[value_col]).sort_index()

    values = dfw[value_col].to_numpy(dtype=float)
    times = dfw.index

    if values.size < window_size:
        raise RuntimeError("Not enough samples to form the first rolling window.")

    x_ref = values[:window_size]
    n_ref = int(np.isfinite(x_ref).sum())
    if n_ref == 0:
        raise RuntimeError("First rolling window has no valid samples.")

    bins = make_bins_from_reference(x_ref, k=k_bins)
    k_eff = len(bins) - 1
    codes_ref = codes_from_bins(x_ref, bins=bins)
    q = empirical_probs_from_codes(codes_ref, k=k_bins)

    results = []
    last_start = values.size - window_size
    for start in range(0, last_start + 1, step):
        end = start + window_size
        x_win = values[start:end]
        x_win = x_win[np.isfinite(x_win)]
        n = x_win.size
        if n == 0:
            continue
        codes_w = codes_from_bins(x_win, bins=bins)
        p = empirical_probs_from_codes(codes_w, k=k_bins)
        T = kl(p, q)
        t0 = times[start]
        results.append((t0, T, n))

    res = pd.DataFrame(results, columns=["time", "KL_vs_first", "n_obs"]).set_index("time")
    res.index = _ensure_paris_tz(res.index)
    return res, res.index[0], n_ref, k_eff


# =========================================================
# 4) Plot (template-like)
# =========================================================
def plot_like_template(res: pd.DataFrame, k: int, n_for_threshold: int, title: str):
    alpha = ALPHA

    th_one = thresh_asymptotic_one_sample(k, n_for_threshold, alpha)
    th_two = thresh_asymptotic_two_sample_equal(k, n_for_threshold, alpha)
    th_m3 = thresh_M3(k, n_for_threshold, alpha)
    th_mf3s = thresh_Mf3_star_equal(k, n_for_threshold, alpha)
    th_solid = 2 * (k - 1) / n_for_threshold  # same as template

    plt.figure(figsize=(11, 4.2))

    # only one curve (vs first window) - use black like template's "first window"
    plt.plot(res.index, res["KL_vs_first"], color="black", linewidth=1.4, label="vs first window")

    # thresholds: keep same style as template
    plt.axhline(th_one, color="red", linestyle="--", linewidth=1.0, label="asymptotic one-sample (1%)")
    plt.axhline(th_two, color="red", linestyle="--", linewidth=2.0, label="asymptotic two-sample equal (1%)")

    plt.axhline(th_m3, color="black", linestyle="--", linewidth=1.0, label="M3 bound (1%)")
    if th_mf3s is not None:
        plt.axhline(th_mf3s, color="black", linestyle="--", linewidth=2.0, label="Mf3* bound (1%)")
    else:
        print("Mf3* bound skipped: no sign change for root.")

    plt.axhline(th_solid, color="black", linestyle="-", linewidth=1.0, label="2(k-1)/n")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=res.index.tz))

    plt.title(f"{title}  (k={k}, n≈{n_for_threshold})")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()


def main():
    df = _read_parquet(PARQUET_PATH)

    time_col = _detect_time_col(df)
    value_col = _detect_value_col(df, VOLUME_SIDE)

    df = _to_datetime_index(df, time_col)
    df.index = _ensure_paris_tz(df.index)

    res, ref_time, n_ref, k_eff = rolling_kl_vs_first(
        df,
        value_col=value_col,
        k_bins=K_BINS,
        window_size=WINDOW_SIZE,
        step=STEP_SIZE,
    )

    print("Reference window start:", ref_time)
    print(res)

    # For thresholds: use median n_obs across windows
    n_med = int(np.median(res["n_obs"].values))
    plot_like_template(
        res,
        k=k_eff,
        n_for_threshold=n_med,
        title=f"Sanofi {value_col}: KL(window || first window) - {ref_time.date()}",
    )


if __name__ == "__main__":
    main()
