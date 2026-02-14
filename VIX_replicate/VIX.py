import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import brentq

# =========================================================
# 0) Data loader (FRED) -- replaced with your function
# =========================================================
def load_vix_from_fred(start="2000-01-01", end=None):
    """
    FRED series: VIXCLS
    Direct CSV endpoint (no API key needed):
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
    # Read the CSV, treat the first column as the index and parse it as dates.
    df = pd.read_csv(url, index_col=0, parse_dates=True)

    # Rename the index to 'date' and the VIX column to 'vix'
    df.index.name = 'date'
    df = df.rename(columns={"VIXCLS": "vix"})

    df["vix"] = pd.to_numeric(df["vix"], errors="coerce")
    df = df.dropna().sort_index()
    if end is not None:
        df = df.loc[pd.to_datetime(start):pd.to_datetime(end)]
    else:
        df = df.loc[pd.to_datetime(start):]
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
def thresh_asymptotic_one_sample(k: int, n: int, alpha=0.01) -> float:
    return chi2.ppf(1 - alpha, df=k - 1) / (2 * n)

def thresh_asymptotic_two_sample_equal(k: int, n: int, alpha=0.01) -> float:
    return chi2.ppf(1 - alpha, df=k - 1) / n

def thresh_M3(k: int, n: int, alpha=0.01) -> float:
    def M3(x):
        return np.exp(-n * x) * (np.e * n * x / (k - 1)) ** (k - 1)
    x0 = (k - 1) / n * 1.0000001
    x1 = x0
    while M3(x1) > alpha:
        x1 *= 1.5
        if x1 > 50:
            break
    return brentq(lambda x: M3(x) - alpha, x0, x1)

def thresh_Mf3_star_equal(k: int, n: int, alpha=0.01) -> float:
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
# 3) Rolling KL (end-at-date windows, exactly as caption describes)
# =========================================================
def window_probs(series: pd.Series, end_idx: int, n_days: int, k_mode: str):
    v = series.iloc[end_idx - n_days + 1 : end_idx + 1].to_numpy(dtype=float)
    ind = indicators_from_window(v)
    codes8 = triples_codes_from_ind(ind)

    if k_mode == "k8":
        k = 8
        codes = codes8
    elif k_mode == "k6":
        k = 6
        codes = map_codes_k6(codes8)
    else:
        raise ValueError("k_mode must be 'k8' or 'k6'")

    p = empirical_probs_from_codes(codes, k)
    return p, k

def rolling_kl(series: pd.Series, n_days: int, k_mode: str):
    p_first, k = window_probs(series, end_idx=n_days - 1, n_days=n_days, k_mode=k_mode)

    dates, kl_prev, kl_first = [], [], []

    start_end = 2 * n_days - 1
    for end_idx in range(start_end, len(series)):
        p_cur, k2 = window_probs(series, end_idx=end_idx, n_days=n_days, k_mode=k_mode)
        p_prev, _ = window_probs(series, end_idx=end_idx - n_days, n_days=n_days, k_mode=k_mode)
        if k2 != k:
            raise RuntimeError("k mismatch.")

        dates.append(series.index[end_idx])
        kl_prev.append(kl(p_cur, p_prev))
        kl_first.append(kl(p_cur, p_first))

    return pd.DataFrame({"KL_vs_prev": kl_prev, "KL_vs_first": kl_first}, index=pd.to_datetime(dates)), k


# =========================================================
# 4) Plot (paper-like styling)
# =========================================================
def plot_like_fig(res: pd.DataFrame, k: int, n_days: int, title: str):
    alpha = 0.01

    th_one = thresh_asymptotic_one_sample(k, n_days, alpha)        # thin red dashed
    th_two = thresh_asymptotic_two_sample_equal(k, n_days, alpha)  # thick red dashed
    th_m3 = thresh_M3(k, n_days, alpha)                            # thin black dashed
    th_mf3s = thresh_Mf3_star_equal(k, n_days, alpha)              # thick black dashed
    th_solid = 2 * (k - 1) / n_days                                # solid black

    plt.figure(figsize=(11, 4.2))

    plt.plot(res.index, res["KL_vs_prev"], color="red", linewidth=1.2, label="previous window")
    plt.plot(res.index, res["KL_vs_first"], color="black", linewidth=1.2, label="first window")

    plt.axhline(th_one, color="red", linestyle="--", linewidth=1.0)
    plt.axhline(th_two, color="red", linestyle="--", linewidth=2.0)

    plt.axhline(th_m3, color="black", linestyle="--", linewidth=1.0)
    plt.axhline(th_mf3s, color="black", linestyle="--", linewidth=2.0)

    plt.axhline(th_solid, color="black", linestyle="-", linewidth=1.0)

    plt.title(f"{title}  (k={k}, n={n_days})")
    plt.tight_layout()
    plt.show()


# =========================================================
# 5) Run
# =========================================================
if __name__ == "__main__":
    df = load_vix_from_fred(start="2000-01-01")
    vix = df["vix"]

    # Figure 10 style: ~1 year windows, k=8
    res_1y, k1 = rolling_kl(vix, n_days=252, k_mode="k8")
    plot_like_fig(res_1y, k1, 252, "VIX (≈1-year windows, Figure-10 style)")

    # Figure 11 style: ~3 months windows, k=6 (merged)
    res_3m, k2 = rolling_kl(vix, n_days=63, k_mode="k6")
    plot_like_fig(res_3m, k2, 63, "VIX (≈3-month windows, Figure-11 style)")
