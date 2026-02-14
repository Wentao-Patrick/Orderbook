"""
Sanofi (Yahoo Finance) — Hawkes (µ, α, β) MLE, time-split stability test, PRFIB bootstrap, and rescaling diagnostics.

Goal (close to Cavaliere et al., 2023, Sec. 8.1 "Dow Jones Index"):
1) Pull daily close prices from Yahoo Finance (SAN.PA) OR read from a local CSV.
2) Define events by extreme returns (default: log_return > 0.02).
3) Split by *time span* into two halves (NOT by number of events).
4) Fit univariate Hawkes with exponential kernel on each half by MLE (µ, α, β).
5) Test H0: θ1 = θ2 using LR statistic:
     LR = 2 [ ℓ1(θ̂1) + ℓ2(θ̂2) - (ℓ1(θ̂0) + ℓ2(θ̂0)) ],
   where θ̂0 maximizes ℓ1(θ)+ℓ2(θ) (common parameters across halves).
   - Theoretical p-value: LR ~ χ^2(df=3) under H0 (asymptotic).
   - Bootstrap p-value: PRFIB (parametric fixed-intensity bootstrap) under H0.
6) Evaluate fit via time-rescaling: transformed waiting times v̂_i should be i.i.d. Exp(1).

Notes:
- Time unit is "trading days" (index positions), like Cavaliere et al.'s DJI application.
- Parameterization uses (µ, a, β) internally with α = a β and a∈(0,1) to enforce stationarity (α<β).

Dependencies:
  pip install yfinance scipy tqdm
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import chi2, kstest

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# ============================================================
# 0) User parameters
# ============================================================
SYMBOL = "SAN.PA"         # Sanofi on Euronext Paris (Yahoo Finance)
RETURN_THR = 0.02         # event if log_return > RETURN_THR
DIRECTION = "gt"          # {"gt","lt","abs_gt"}
B_BOOT = 3000              # number of bootstrap replications
SEED = 7

# If you already saved Yahoo daily data to a CSV, set:
PRICE_CSV: Optional[str] = None  # e.g. "sanofi_SAN.PA_yahoo_daily.csv"
# Otherwise, we will try to download via yfinance.
USE_YFINANCE_IF_NO_CSV = True

# Diagnostics output
OUT_DIR = "."
SAVE_FIGS = True


# ============================================================
# 1) Data utilities
# ============================================================
def fetch_yahoo_daily(symbol: str = SYMBOL) -> pd.DataFrame:
    import yfinance as yf

    t = yf.Ticker(symbol)
    hist = t.history(period="max", interval="1d")
    if hist is None or hist.empty:
        raise RuntimeError("Yahoo Finance download failed (empty dataframe). Check internet/ticker.")
    hist = hist.reset_index()
    # Standardize column names
    if "Date" not in hist.columns:
        # yfinance uses 'Date' for daily; but keep robust
        for c in hist.columns:
            if str(c).lower().startswith("date"):
                hist = hist.rename(columns={c: "Date"})
                break
    hist["Date"] = pd.to_datetime(hist["Date"])
    return hist[["Date", "Close"]].dropna().sort_values("Date").reset_index(drop=True)


def read_price_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Try to detect date/close columns
    date_col = None
    for c in df.columns:
        if str(c).lower() in {"date", "datetime", "time"}:
            date_col = c
            break
    if date_col is None:
        raise ValueError("CSV must contain a date column named 'Date' (or similar).")
    close_col = None
    for c in df.columns:
        if str(c).lower() == "close":
            close_col = c
            break
    if close_col is None:
        raise ValueError("CSV must contain a 'Close' column.")
    out = df[[date_col, close_col]].rename(columns={date_col: "Date", close_col: "Close"}).copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.dropna().sort_values("Date").reset_index(drop=True)
    return out


def make_event_times_from_returns(
    price_df: pd.DataFrame,
    thr: float = RETURN_THR,
    direction: str = DIRECTION,
) -> Tuple[np.ndarray, int, pd.DataFrame]:
    """
    Convert daily closes into event times t_i measured in trading-day index.

    We define:
      r_t = log(C_t / C_{t-1})
    Event occurs at trading day t if condition on r_t holds.

    Returns:
      event_times: np.array of event times in [0, T], as floats (integers)
      T: total span length (last day index)
      df: dataframe with Date, Close, log_return, t_index, is_event
    """
    df = price_df.copy()
    df["log_return"] = np.log(df["Close"]).diff()
    df["t_index"] = np.arange(len(df), dtype=float)

    if direction == "gt":
        df["is_event"] = df["log_return"] > thr
    elif direction == "lt":
        df["is_event"] = df["log_return"] < -thr
    elif direction == "abs_gt":
        df["is_event"] = df["log_return"].abs() > thr
    else:
        raise ValueError("direction must be one of {'gt','lt','abs_gt'}.")

    # Events at day t correspond to time index t (skip t=0 because return is NaN)
    ev = df.loc[df["is_event"].fillna(False), "t_index"].to_numpy(dtype=float)
    # Total horizon: last index
    T = int(df["t_index"].iloc[-1])
    return ev, T, df


def split_by_time_half(event_times: np.ndarray, T: int) -> Tuple[np.ndarray, int, np.ndarray, int, int]:
    """
    Split the observation window [0, T] into two halves by time:
      [0, T1] and (T1, T] with T1=floor(T/2).
    NOT by number of events.

    Returns:
      t1, T1, t2_shifted, T2, split_point
    """
    split_point = T // 2
    T1 = split_point
    T2 = T - split_point

    t1 = event_times[event_times <= split_point].copy()
    t2 = event_times[event_times > split_point].copy() - split_point
    return t1, T1, t2, T2, split_point


# ============================================================
# 2) Hawkes (exp kernel) log-likelihood and MLE
# ============================================================
@dataclass(frozen=True)
class HawkesParams:
    mu: float
    alpha: float
    beta: float

    @property
    def a(self) -> float:
        return self.alpha / self.beta if self.beta > 0 else np.nan


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def unpack_u_to_params(u: np.ndarray) -> HawkesParams:
    """
    u = [log_mu, log_beta, logit_a]  -> (mu, alpha, beta) with alpha=a*beta
    Keep a away from 1 to ensure stationarity.
    """
    log_mu, log_beta, logit_a = float(u[0]), float(u[1]), float(u[2])
    mu = math.exp(log_mu)
    beta = math.exp(log_beta)
    a = _sigmoid(logit_a)
    a = min(max(a, 1e-6), 0.999)  # enforce 0<a<1
    alpha = a * beta
    return HawkesParams(mu=mu, alpha=alpha, beta=beta)


def hawkes_loglik(params: HawkesParams, t: np.ndarray, T: float) -> float:
    """
    Univariate Hawkes with exponential kernel:
      lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta (t - t_i))
    Log-likelihood:
      l = sum_i log lambda(t_i) - ∫_0^T lambda(s) ds

    Efficient recursion for the sum at event times:
      g_0 = 0
      g_i = exp(-beta Δ_i) * (1 + g_{i-1})  for i>=1
      lambda(t_i) = mu + alpha * g_i
    """
    mu, alpha, beta = params.mu, params.alpha, params.beta
    if mu <= 0 or alpha <= 0 or beta <= 0:
        return -np.inf

    n = int(t.size)
    if n == 0:
        return -mu * T

    # Ensure sorted and within (0,T]
    tt = np.asarray(t, dtype=float)
    tt = tt[(tt > 0) & (tt <= T)]
    tt.sort()
    n = tt.size
    if n == 0:
        return -mu * T

    # Sum of logs
    g = 0.0
    sum_log = 0.0
    prev = tt[0]
    # i=0, g=0 -> lambda=mu
    lam0 = mu
    if lam0 <= 0:
        return -np.inf
    sum_log += math.log(lam0)

    for i in range(1, n):
        dt = tt[i] - prev
        if dt < 0:
            return -np.inf
        g = math.exp(-beta * dt) * (1.0 + g)
        lam = mu + alpha * g
        if lam <= 0:
            return -np.inf
        sum_log += math.log(lam)
        prev = tt[i]

    # Integral part: mu*T + (alpha/beta) * sum_{i=1..n} (1 - exp(-beta (T - t_i)))
    tail = T - tt
    integral = mu * T + (alpha / beta) * float(np.sum(1.0 - np.exp(-beta * tail)))
    return sum_log - integral


def neg_loglik_u(u: np.ndarray, t: np.ndarray, T: float) -> float:
    return -hawkes_loglik(unpack_u_to_params(u), t, T)


def fit_hawkes_mle(t: np.ndarray, T: float, x0: Optional[np.ndarray] = None) -> Tuple[HawkesParams, float, Dict]:
    """
    MLE via L-BFGS-B in unconstrained variables u=[log_mu, log_beta, logit_a].
    """
    t = np.asarray(t, dtype=float)
    t = t[(t > 0) & (t <= T)]
    t.sort()
    n = t.size

    # Heuristic init
    if x0 is None:
        mu0 = max(n / max(T, 1.0), 1e-4)
        beta0 = 1.0
        a0 = 0.5
        x0 = np.array([math.log(mu0), math.log(beta0), math.log(a0 / (1 - a0))], dtype=float)

    res = minimize(
        neg_loglik_u,
        x0=x0,
        args=(t, T),
        method="L-BFGS-B",
        options=dict(maxiter=2000, ftol=1e-9),
    )
    p = unpack_u_to_params(res.x)
    ll = hawkes_loglik(p, t, T)
    info = dict(
        success=bool(res.success),
        message=str(res.message),
        n_events=int(n),
        T=float(T),
        fun=float(res.fun),
        nit=int(getattr(res, "nit", -1)),
    )
    return p, ll, info


def fit_restricted_mle(t1: np.ndarray, T1: float, t2: np.ndarray, T2: float, x0: Optional[np.ndarray] = None) -> Tuple[HawkesParams, float, Dict]:
    """
    Restricted MLE under H0: common parameters across two halves.
    Maximize l1(theta)+l2(theta).
    """
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)

    def neg_ll(u: np.ndarray) -> float:
        p = unpack_u_to_params(u)
        return -(hawkes_loglik(p, t1, T1) + hawkes_loglik(p, t2, T2))

    # init: from pooled Poisson
    if x0 is None:
        n = int(((t1 > 0) & (t1 <= T1)).sum() + ((t2 > 0) & (t2 <= T2)).sum())
        T = float(T1 + T2)
        mu0 = max(n / max(T, 1.0), 1e-4)
        beta0 = 1.0
        a0 = 0.5
        x0 = np.array([math.log(mu0), math.log(beta0), math.log(a0 / (1 - a0))], dtype=float)

    res = minimize(
        neg_ll,
        x0=x0,
        method="L-BFGS-B",
        options=dict(maxiter=2000, ftol=1e-9),
    )
    p = unpack_u_to_params(res.x)
    ll = hawkes_loglik(p, t1, T1) + hawkes_loglik(p, t2, T2)
    info = dict(
        success=bool(res.success),
        message=str(res.message),
        n_events=int(((t1 > 0) & (t1 <= T1)).sum() + ((t2 > 0) & (t2 <= T2)).sum()),
        T=float(T1 + T2),
        fun=float(res.fun),
        nit=int(getattr(res, "nit", -1)),
    )
    return p, ll, info


# ============================================================
# 3) Time-rescaling diagnostics (v_i should be i.i.d. Exp(1))
# ============================================================
def transformed_waiting_times(params: HawkesParams, t: np.ndarray, T: float) -> np.ndarray:
    """
    v_i = Λ(t_i) - Λ(t_{i-1}), with t_0 = 0.
    Under correct specification: v_i i.i.d Exp(1).
    """
    mu, alpha, beta = params.mu, params.alpha, params.beta
    tt = np.asarray(t, dtype=float)
    tt = tt[(tt > 0) & (tt <= T)]
    tt.sort()
    n = tt.size
    if n == 0:
        return np.array([], dtype=float)

    # Cumulative Λ at event times via segment recursion
    # Between two event times, the set of past events is fixed.
    # We compute Λ increment exactly on each inter-event interval using the past excitation at the left boundary.
    v = np.empty(n, dtype=float)

    # Past excitation sum at t0=0 is 0.
    g_left = 0.0
    t_left = 0.0

    for i in range(n):
        t_right = tt[i]
        dt = t_right - t_left
        # Integral over [t_left, t_right):
        # ∫ (mu + alpha * g_left * exp(-beta (s - t_left))) ds
        # where g_left = sum_{t_j < t_left} exp(-beta (t_left - t_j))
        inc = mu * dt + (alpha * g_left / beta) * (1.0 - math.exp(-beta * dt))
        v[i] = inc

        # Update g at the event time (right limit before the jump):
        # g_right = exp(-beta dt) * g_left  + 1 (new event contributes exp(-beta*0)=1 for future)
        g_left = math.exp(-beta * dt) * g_left + 1.0
        t_left = t_right

    return v


def plot_rescaling_diagnostics(v: np.ndarray, title_prefix: str, out_prefix: str) -> Dict[str, float]:
    """
    Save QQ plot and ECDF vs Exp(1) + report KS test p-value.
    """
    out = {}
    if v.size == 0:
        return {"ks_stat": np.nan, "ks_pvalue": np.nan}

    # KS test against Exp(1) (mean 1)
    ks = kstest(v, "expon", args=(0, 1))
    out["ks_stat"] = float(ks.statistic)
    out["ks_pvalue"] = float(ks.pvalue)

    # QQ plot vs Exp(1)
    vv = np.sort(v)
    n = vv.size
    p = (np.arange(1, n + 1) - 0.5) / n
    q_theory = -np.log(1.0 - p)  # Exp(1) quantiles

    plt.figure(figsize=(5, 5))
    plt.scatter(q_theory, vv, s=10)
    m = max(q_theory.max(), vv.max())
    plt.plot([0, m], [0, m], linewidth=1)
    plt.grid(alpha=0.3)
    plt.xlabel("Quantiles Exp(1)")
    plt.ylabel("Quantiles empiriques de $\\hat v_i$")
    plt.title(f"{title_prefix} — QQ plot (rescaling)")
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(f"{OUT_DIR}/{out_prefix}_qq.png", dpi=160)
    plt.show()

    # ECDF vs Exp(1)
    ecdf_y = np.arange(1, n + 1) / n
    exp_cdf = 1.0 - np.exp(-vv)

    plt.figure(figsize=(6, 4))
    plt.plot(vv, ecdf_y, label="ECDF($\\hat v_i$)")
    plt.plot(vv, exp_cdf, label="CDF Exp(1)")
    plt.grid(alpha=0.3)
    plt.xlabel("$v$")
    plt.ylabel("F(v)")
    plt.title(f"{title_prefix} — ECDF vs Exp(1) (KS p={out['ks_pvalue']:.3g})")
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(f"{OUT_DIR}/{out_prefix}_ecdf.png", dpi=160)
    plt.show()

    return out


# ============================================================
# 4) PRFIB (parametric fixed-intensity bootstrap) generator
#    N*(t) = Q*(Lambda_hat(t)) with Q* ~ Poisson(1)
# ============================================================
@dataclass
class FixedIntensityInverse:
    t_knots: np.ndarray        # [0, t1, ..., tn, T]
    L_knots: np.ndarray        # Lambda_hat(t_knots)
    prefix_exp: np.ndarray     # prefix_exp[i] = sum_{j< i} exp(beta * t_j) for i segments
    params: HawkesParams
    T: float

    def Lambda(self, t: float) -> float:
        """Exact Lambda_hat(t) for fixed intensity based on original event times."""
        mu, alpha, beta = self.params.mu, self.params.alpha, self.params.beta
        if t <= 0:
            return 0.0
        if t >= self.T:
            return float(self.L_knots[-1])

        # find segment i s.t. t in [t_knots[i], t_knots[i+1])
        i = int(np.searchsorted(self.t_knots, t, side="right") - 1)
        i = max(0, min(i, len(self.t_knots) - 2))
        t_left = float(self.t_knots[i])
        A = float(self.L_knots[i])
        S = float(self.prefix_exp[i])  # sum_{events < t_left} exp(beta * t_j)

        # Lambda(t) = A + mu*(t-t_left) + alpha*S*(exp(-beta t_left) - exp(-beta t))/beta
        return A + mu * (t - t_left) + (alpha * S / beta) * (math.exp(-beta * t_left) - math.exp(-beta * t))

    def inv_Lambda(self, s: float, max_iter: int = 60) -> float:
        """
        Invert Lambda_hat(t)=s by bisection on the segment where it lives.
        """
        if s <= 0:
            return 0.0
        if s >= float(self.L_knots[-1]):
            return float(self.T)

        # locate segment
        i = int(np.searchsorted(self.L_knots, s, side="right") - 1)
        i = max(0, min(i, len(self.L_knots) - 2))
        lo = float(self.t_knots[i])
        hi = float(self.t_knots[i + 1])

        # bisection
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            val = self.Lambda(mid)
            if val < s:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def build_fixed_intensity_inverse(events: np.ndarray, T: float, params: HawkesParams) -> FixedIntensityInverse:
    """
    Build Lambda_hat(t) and its inverse for PRFIB, where intensity is fixed from original events.
    """
    mu, alpha, beta = params.mu, params.alpha, params.beta
    tt = np.asarray(events, dtype=float)
    tt = tt[(tt > 0) & (tt <= T)]
    tt.sort()
    n = tt.size

    t_knots = np.concatenate(([0.0], tt, [float(T)]))
    # prefix_exp[i] = sum_{j< i} exp(beta*t_j), where t_j are events tt (j from 0..n-1)
    prefix_exp = np.zeros(n + 1, dtype=float)
    if n > 0:
        exp_bt = np.exp(beta * tt)
        prefix_exp[1:] = np.cumsum(exp_bt)

    L_knots = np.zeros(n + 2, dtype=float)
    # Iterate segments i=0..n (segment i has S = prefix_exp[i])
    for i in range(n + 1):
        t_left = float(t_knots[i])
        t_right = float(t_knots[i + 1])
        S = float(prefix_exp[i])
        dt = t_right - t_left
        inc = mu * dt + (alpha * S / beta) * (math.exp(-beta * t_left) - math.exp(-beta * t_right))
        L_knots[i + 1] = L_knots[i] + inc

    return FixedIntensityInverse(
        t_knots=t_knots,
        L_knots=L_knots,
        prefix_exp=prefix_exp,
        params=params,
        T=float(T),
    )


def simulate_prfib(events_orig: np.ndarray, T: float, params: HawkesParams, rng: np.random.Generator) -> np.ndarray:
    """
    PRFIB: simulate N*(t) = Q*(Lambda_hat(t)), Q* homogeneous Poisson(1).
    Lambda_hat is built from original events and params (fixed intensity).
    """
    inv = build_fixed_intensity_inverse(events_orig, T, params)
    L_T = float(inv.L_knots[-1])

    # Homogeneous Poisson(1) arrival times on [0, L_T]
    s_list: List[float] = []
    s = 0.0
    while True:
        s += float(rng.exponential(scale=1.0))
        if s >= L_T:
            break
        s_list.append(s)

    if not s_list:
        return np.array([], dtype=float)

    # Map back through inverse
    t_star = np.array([inv.inv_Lambda(ss) for ss in s_list], dtype=float)
    # Remove possible numerical boundary issues
    t_star = t_star[(t_star > 0) & (t_star <= T)]
    t_star.sort()
    return t_star


# ============================================================
# 5) LR test + bootstrap p-value
# ============================================================
@dataclass
class LRTestResult:
    theta1: HawkesParams
    theta2: HawkesParams
    theta0: HawkesParams
    ll1: float
    ll2: float
    ll0: float
    LR: float
    p_theory: float
    p_boot: Optional[float]
    boot_LR: Optional[np.ndarray]


def lr_test_and_bootstrap(
    t1: np.ndarray, T1: float,
    t2: np.ndarray, T2: float,
    B: int = B_BOOT,
    seed: int = SEED,
    do_bootstrap: bool = True,
) -> LRTestResult:
    # Unrestricted fits
    th1, ll1, info1 = fit_hawkes_mle(t1, T1)
    th2, ll2, info2 = fit_hawkes_mle(t2, T2)
    # Restricted fit
    th0, ll0, info0 = fit_restricted_mle(t1, T1, t2, T2)

    LR = 2.0 * ((ll1 + ll2) - ll0)
    p_theory = 1.0 - float(chi2.cdf(LR, df=3))

    boot_LR = None
    p_boot = None

    if do_bootstrap:
        rng = np.random.default_rng(seed)
        LRs = np.empty(B, dtype=float)

        it = range(B)
        if tqdm is not None:
            it = tqdm(it, desc="PRFIB bootstrap (LR null distribution)", leave=True)

        for b in it:
            # Generate bootstrap samples under H0 (fixed intensity based on original events + theta0)
            t1s = simulate_prfib(t1, T1, th0, rng)
            t2s = simulate_prfib(t2, T2, th0, rng)

            # Refit (unrestricted + restricted) on bootstrap world
            th1s, ll1s, _ = fit_hawkes_mle(t1s, T1)
            th2s, ll2s, _ = fit_hawkes_mle(t2s, T2)
            th0s, ll0s, _ = fit_restricted_mle(t1s, T1, t2s, T2)

            LRs[b] = 2.0 * ((ll1s + ll2s) - ll0s)

        boot_LR = LRs
        # One-sided LR test: reject for large LR
        p_boot = (1.0 + float(np.sum(LRs >= LR))) / (B + 1.0)

    return LRTestResult(
        theta1=th1, theta2=th2, theta0=th0,
        ll1=float(ll1), ll2=float(ll2), ll0=float(ll0),
        LR=float(LR),
        p_theory=float(p_theory),
        p_boot=None if p_boot is None else float(p_boot),
        boot_LR=boot_LR,
    )


def format_params(p: HawkesParams) -> str:
    return f"mu={p.mu:.6g}, alpha={p.alpha:.6g}, beta={p.beta:.6g}, a=alpha/beta={p.a:.4f}"


def plot_bootstrap_hist(LR_obs: float, LR_boot: np.ndarray, out_path: str):
    plt.figure(figsize=(6, 4))
    plt.hist(LR_boot, bins=30, density=True, alpha=0.8)
    plt.axvline(LR_obs, linewidth=2, label=f"LR_obs={LR_obs:.3g}")
    plt.grid(alpha=0.3)
    plt.xlabel("LR* (bootstrap)")
    plt.ylabel("Density")
    plt.title("PRFIB bootstrap null distribution of LR")
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(out_path, dpi=160)
    plt.show()


# ============================================================
# 6) Main
# ============================================================
def main():
    # Load prices
    if PRICE_CSV is not None:
        price = read_price_csv(PRICE_CSV)
        print(f"[OK] Read prices from CSV: {PRICE_CSV} ({len(price)} rows)")
    else:
        if not USE_YFINANCE_IF_NO_CSV:
            raise RuntimeError("PRICE_CSV is None and USE_YFINANCE_IF_NO_CSV is False.")
        price = fetch_yahoo_daily(SYMBOL)
        print(f"[OK] Downloaded prices from Yahoo: {SYMBOL} ({len(price)} rows)")

    events, T, df = make_event_times_from_returns(price, thr=RETURN_THR, direction=DIRECTION)
    print(f"[INFO] Total span T={T} trading-days, n_events={len(events)} (direction={DIRECTION}, thr={RETURN_THR})")

    t1, T1, t2, T2, split_point = split_by_time_half(events, T)
    print(f"[INFO] Split at t={split_point}: first half T1={T1}, n1={len(t1)}; second half T2={T2}, n2={len(t2)}")

    # LR test + PRFIB bootstrap
    res = lr_test_and_bootstrap(t1, T1, t2, T2, B=B_BOOT, seed=SEED, do_bootstrap=True)

    print("\n===== MLE parameters =====")
    print("[Half 1] ", format_params(res.theta1))
    print("[Half 2] ", format_params(res.theta2))
    print("[H0 pool]", format_params(res.theta0))

    print("\n===== LR stability test: H0: theta1 = theta2 =====")
    print(f"LR_obs = {res.LR:.6g}")
    print(f"p_theory (chi^2_3) = {res.p_theory:.6g}")
    if res.p_boot is not None:
        print(f"p_boot   (PRFIB, B={B_BOOT}) = {res.p_boot:.6g}")

    if res.boot_LR is not None:
        plot_bootstrap_hist(res.LR, res.boot_LR, out_path=f"{OUT_DIR}/LR_boot_hist.png")

    # Rescaling diagnostics on each half
    print("\n===== Rescaling diagnostics (Exp(1) target) =====")
    v1 = transformed_waiting_times(res.theta1, t1, T1)
    d1 = plot_rescaling_diagnostics(v1, title_prefix="Half 1", out_prefix="half1_rescaling")
    print(f"[Half 1] KS stat={d1['ks_stat']:.4g}, p={d1['ks_pvalue']:.4g}")

    v2 = transformed_waiting_times(res.theta2, t2, T2)
    d2 = plot_rescaling_diagnostics(v2, title_prefix="Half 2", out_prefix="half2_rescaling")
    print(f"[Half 2] KS stat={d2['ks_stat']:.4g}, p={d2['ks_pvalue']:.4g}")

    # Optional: save the event dataframe
    out_csv = f"{OUT_DIR}/sanofi_yahoo_events_from_returns.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved event-marked price series: {out_csv}")


if __name__ == "__main__":
    main()
