# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import brentq
import matplotlib.dates as mdates


# =========================================================
# 0) Data loader (from local CSV)
# =========================================================
def load_series_from_csv(
    file_path: str,
    date_col: str,
    value_col: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Loads a time series from a local CSV file.
    Assumption: the CSV datetime column is already in Paris local time (tz-naive),
    as you stated.

    Args:
        file_path: Path to the CSV file.
        date_col: Name of the column containing dates/timestamps.
        value_col: Name of the column containing the numeric time series values.
        start: Optional start date (YYYY-MM-DD).
        end: Optional end date (YYYY-MM-DD).
    """
    df = pd.read_csv(file_path, index_col=date_col, parse_dates=True)

    # --- 新增的代码 ---
    # pd.read_csv converts tz-aware strings to UTC by default.
    # We need to localize it back to the original Paris time.
    df.index = df.index.tz_localize(None) # Make naive first
    df.index = df.index.tz_localize("Europe/Paris") # Then localize to Paris
    # --------------------

    # Rename the index and the value column for consistency
    df.index.name = "date"
    df = df.rename(columns={value_col: "value"})

    # Convert to numeric, coercing errors, and drop missing values
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_index()

    # Filter by date range if specified
    if start:
        df = df.loc[pd.to_datetime(start):]
    if end:
        df = df.loc[: pd.to_datetime(end)]

    return df


# =========================================================
# 1) Discretization (paper): increment -> sign indicator -> triples -> categories
# =========================================================
def indicators_from_window(v_window: np.ndarray) -> np.ndarray:
    dv = np.diff(v_window)
    return (dv > 0).astype(int)


def triples_codes_from_ind(ind: np.ndarray) -> np.ndarray:
    b0 = ind[:-2]
    b1 = ind[1:-1]
    b2 = ind[2:]
    return (b0 << 2) + (b1 << 1) + b2  # 0..7


def map_codes_k6(codes8: np.ndarray) -> np.ndarray:
    # merge (0,0,0) with (1,1,1) and (0,1,0) with (1,0,1)
    mapping = {
        0: 0, 7: 0,   # trend merged
        2: 1, 5: 1,   # strict alternation merged
        1: 2,
        3: 3,
        4: 4,
        6: 5,
    }  
    return np.vectorize(mapping.get)(codes8)


def map_codes_k4_trend_strength(codes8: np.ndarray) -> np.ndarray:
    mapping = {
        0: 0, 7: 0,   # 强趋势 (DDD, UUU)
        1: 1, 6: 1,   # 趋势延续 (DDU, UUD)
        3: 2, 4: 2,   # 趋势反转 (DUU, UDD)
        2: 3, 5: 3,   # 震荡 (DUD, UDU)
    }
    return np.vectorize(mapping.get)(codes8)


def map_codes_k2_final_move(codes8: np.ndarray) -> np.ndarray:
    # 编码的最后一位是0代表下降，是1代表上升
    # codes8 % 2 即可得到最后一位
    return codes8 % 2


def map_codes_k3_volatility(codes8: np.ndarray) -> np.ndarray:
    mapping = {
        0: 0, 7: 0,   # 平滑运动
        1: 1, 3: 1, 4: 1, 6: 1, # 单次变向
        2: 2, 5: 2,   # 高频波动
    }
    return np.vectorize(mapping.get)(codes8)


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
# 2) 99% thresholds (Figure 10/11 style)
# =========================================================
def thresh_asymptotic_one_sample(k: int, n: int, alpha: float = 0.01) -> float:
    return chi2.ppf(1 - alpha, df=k - 1) / (2 * n)


def thresh_asymptotic_two_sample_equal(k: int, n: int, alpha: float = 0.01) -> float:
    return chi2.ppf(1 - alpha, df=k - 1) / n


def thresh_M3(k: int, n: int, alpha: float = 0.01) -> float:
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
    def Mf3s(x):
        return np.exp(-n * x) * (np.e * n * x / (2 * (k - 1))) ** (2 * (k - 1))

    x0 = (k - 1) / n * 1.0000001
    x1 = x0
    while Mf3s(x1) > alpha:
        x1 *= 1.5
        if x1 > 50:
            break
    return brentq(lambda x: Mf3s(x) - alpha, x0, x1)


# =========================================================
# 3) Rolling KL (KL points placed at the BEGINNING of the 2nd window)
# =========================================================
def window_probs(series: pd.Series, end_idx: int, window_size: int, k_mode: str):
    """
    Build empirical distribution of triple patterns within a window that ends at end_idx.

    Window indices: [end_idx-window_size+1 .. end_idx]
    """
    v = series.iloc[end_idx - window_size + 1: end_idx + 1].to_numpy(dtype=float)
    ind = indicators_from_window(v)
    codes8 = triples_codes_from_ind(ind)

    if k_mode == "k8":
        k = 8
        codes = codes8
    elif k_mode == "k6":
        k = 6
        codes = map_codes_k6(codes8)
    elif k_mode == "k4_trend_strength":
        k = 4
        codes = map_codes_k4_trend_strength(codes8)
    elif k_mode == "k2_final_move":
        k = 2
        codes = map_codes_k2_final_move(codes8)
    elif k_mode == "k3_volatility":
        k = 3
        codes = map_codes_k3_volatility(codes8)
    else:
        raise ValueError("k_mode must be 'k8', 'k6', 'k4_trend_strength', 'k2_final_move', or 'k3_volatility'")

    p = empirical_probs_from_codes(codes, k)
    return p, k


def rolling_kl(series: pd.Series, window_size: int, k_mode: str):
    """
    Compute rolling KL divergences as in the paper.
    Each KL value compares two adjacent windows of length window_size:

      prev window: [end_idx-2n+1 .. end_idx-n]
      cur  window: [end_idx-n+1 .. end_idx]

    IMPORTANT (your request):
    The timestamp for each KL value is set to the first index of the SECOND window:
      boundary_idx = end_idx - n + 1
    """
    if window_size < 4:
        raise ValueError("window_size must be >= 4 because triple-coding needs at least 3 increments.")

    p_first, k = window_probs(series, end_idx=window_size - 1, window_size=window_size, k_mode=k_mode)

    dates, kl_prev, kl_first = [], [], []

    start_end = 2 * window_size - 1
    for end_idx in range(start_end, len(series)):
        p_cur, k2 = window_probs(series, end_idx=end_idx, window_size=window_size, k_mode=k_mode)
        p_prev, _ = window_probs(series, end_idx=end_idx - window_size, window_size=window_size, k_mode=k_mode)

        if k2 != k:
            raise RuntimeError("k mismatch.")

        boundary_idx = end_idx - window_size + 1  # first index of the 2nd window
        dates.append(series.index[boundary_idx])

        kl_prev.append(kl(p_cur, p_prev))
        kl_first.append(kl(p_cur, p_first))

    return (
        pd.DataFrame(
            {"KL_vs_prev": kl_prev, "KL_vs_first": kl_first},
            index=pd.DatetimeIndex(dates),  # keep original (Paris local) time axis
        ),
        k,
    )


# =========================================================
# 4) Plot (paper-like styling). CSV time already in Paris time => no conversion.
# =========================================================
def plot_like_fig(res: pd.DataFrame, k: int, n_minutes: int, title: str, show_boundaries: bool = False):
    alpha = 0.01

    th_one = thresh_asymptotic_one_sample(k, n_minutes, alpha)
    th_two = thresh_asymptotic_two_sample_equal(k, n_minutes, alpha)
    th_m3 = thresh_M3(k, n_minutes, alpha)
    th_mf3s = thresh_Mf3_star_equal(k, n_minutes, alpha)
    th_solid = 2 * (k - 1) / n_minutes

    plt.figure(figsize=(11, 4.2))

    plt.plot(res.index, res["KL_vs_prev"], color="red", linewidth=1.2, label="previous window")
    plt.plot(res.index, res["KL_vs_first"], color="black", linewidth=1.2, label="first window")

    plt.axhline(th_one, color="red", linestyle="--", linewidth=1.0)
    plt.axhline(th_two, color="red", linestyle="--", linewidth=2.0)

    plt.axhline(th_m3, color="black", linestyle="--", linewidth=1.0)
    plt.axhline(th_mf3s, color="black", linestyle="--", linewidth=2.0)

    plt.axhline(th_solid, color="black", linestyle="-", linewidth=1.0)

    # Optional: show window boundary markers (every point is already a boundary; this is for visual debugging)
    if show_boundaries and len(res) > 0:
        # Show a light marker every (n_minutes) points, starting from the first KL point
        step = n_minutes
        idxs = np.arange(0, len(res), step)
        for i in idxs:
            plt.axvline(res.index[i], color="gray", alpha=0.15, linewidth=0.8)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=res.index.tz))

    plt.title(f"{title}  (k={k}, n={n_minutes})")
    plt.tight_layout()
    plt.show()


# =========================================================
# 5) Run
# =========================================================
if __name__ == "__main__":
    # --- 1. CONFIGURE YOUR INPUTS HERE ---
    CSV_FILE_PATH = r"C:\Users\Wentao\Desktop\EA_recherche\Trade\sanofi_1min_aggregation.csv"
    DATE_COLUMN = "trade_time_paris"
    VALUE_COLUMN = "imbalance"
    SERIES_NAME = "Sanofi " + VALUE_COLUMN
    START_DATE = None  # e.g. "2019-10-01"

    # --- 2. CONFIGURE ANALYSIS PARAMETERS ---
    WINDOW_SIZE_LONG = 60
    K_MODE_1 = "k4_trend_strength"

    WINDOW_SIZE_SHORT = 30
    K_MODE_2 = "k3_volatility"

    try:
        df = load_series_from_csv(
            file_path=CSV_FILE_PATH,
            date_col=DATE_COLUMN,
            value_col=VALUE_COLUMN,
            start=START_DATE,
        )
        print(df.head())
        series = df["value"]
        print(f"Successfully loaded {len(series)} data points for '{SERIES_NAME}'.")

        # Analysis 1
        print(f"\nRunning analysis 1: window_size={WINDOW_SIZE_LONG}, k_mode='{K_MODE_1}'")
        res1, k1 = rolling_kl(series, window_size=WINDOW_SIZE_LONG, k_mode=K_MODE_1)
        plot_like_fig(
            res1,
            k1,
            WINDOW_SIZE_LONG,
            f"{SERIES_NAME} ({WINDOW_SIZE_LONG}-minute windows, k={k1})",
            show_boundaries=False,
        )
        print(res1.head())
        print(res1.index)  # should be Europe/Paris
        # Analysis 2
        print(f"\nRunning analysis 2: window_size={WINDOW_SIZE_SHORT}, k_mode='{K_MODE_2}'")
        res2, k2 = rolling_kl(series, window_size=WINDOW_SIZE_SHORT, k_mode=K_MODE_2)
        plot_like_fig(
            res2,
            k2,
            WINDOW_SIZE_SHORT,
            f"{SERIES_NAME} ({WINDOW_SIZE_SHORT}-minute windows, k={k2})",
            show_boundaries=False,
        )

    except FileNotFoundError:
        print(f"ERROR: The file was not found at '{CSV_FILE_PATH}'")
        print("Please update the 'CSV_FILE_PATH' variable in the script.")
    except KeyError:
        print(f"ERROR: Columns not found. Make sure '{DATE_COLUMN}' and '{VALUE_COLUMN}' exist in your CSV.")
        print("Please update 'DATE_COLUMN' and 'VALUE_COLUMN' variables.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
