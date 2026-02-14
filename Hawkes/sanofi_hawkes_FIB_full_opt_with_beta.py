# -*- coding: utf-8 -*-
"""
Sanofi (one day) — Hawkes change test with FIB bootstrap
FULL MLE: optimize (mu_buy, mu_sell, a_self, a_cross, beta), all constants in time.

What this script does
---------------------
1) Read two marked event-time series (buy vs sell) from CSV.
2) Split the observation window into two equal halves.
3) Fit a symmetric bivariate Hawkes with exponential kernel on each half:
      lambda_buy(t)  = mu_buy  + aS * S_buy(t)  + aC * S_sell(t)
      lambda_sell(t) = mu_sell + aC * S_buy(t)  + aS * S_sell(t)
      S_m(t) = sum_{ti^m < t} exp(-beta (t - ti^m))
   All parameters (mu_buy, mu_sell, aS, aC, beta) are optimized by MLE.
4) Likelihood-ratio test:
      H0: same parameter vector on both halves
      H1: different vectors on each half
   LR theoretical p-value uses chi2(df=5).
5) FIB bootstrap (Fixed-Intensity Bootstrap, Cavaliere et al., 2023):
   - Under H0, estimate theta_tilde (pooled MLE).
   - For each half, build the *fixed* estimated intensity path based on the original observed history
     with theta_tilde, then:
        * generate Exp(1) waiting times in transformed time,
        * map back by Λ_hat^{-1} to get bootstrap times,
        * assign marks using fixed component intensities at those times.
   - Refit Hawkes MLE (FULL parameters) on each bootstrap sample (standard Hawkes likelihood),
     compute LR*, and estimate p_boot.
   A progress bar is shown during the bootstrap loop.

6) MLE fit diagnostics (enabled by default):
   Ogata time-rescaling residuals for the *superposed process*:
      v_i = ∫_{t_{i-1}}^{t_i} lambda_total(u) du,  lambda_total = lambda_buy + lambda_sell
   Under correct model: v_i ≈ i.i.d. Exp(1), and u_i = 1-exp(-v_i) ≈ U(0,1).
   Saves:
     - rescaled_half1_fullopt.csv, rescaled_half2_fullopt.csv
     - hist/QQ plots vs Exp(1) and Uniform(0,1)

Inputs
------
- sanofi_hawkes_buy_times.csv  (column: t_seconds_jitter)
- sanofi_hawkes_sell_times.csv (column: t_seconds_jitter)

Run
---
python sanofi_hawkes_FIB_full_opt_with_beta.py --B 1000 --seed 7

"""

from __future__ import annotations
import argparse
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import expon, kstest, chi2

# progress bar
try:
    from tqdm import trange
except Exception:
    trange = None


# -----------------------------
# I/O and event construction
# -----------------------------
def load_times_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    if "t_seconds_jitter" not in df.columns:
        raise ValueError(f"Missing column t_seconds_jitter in {path}")
    t = np.sort(df["t_seconds_jitter"].to_numpy(float))
    if len(t) == 0:
        raise ValueError(f"No times found in {path}")
    return t


def split_and_shift(times: np.ndarray, split: float) -> tuple[np.ndarray, np.ndarray]:
    return times[times <= split].copy(), (times[times > split].copy() - split)


def merge_events(buy_t: np.ndarray, sell_t: np.ndarray) -> np.ndarray:
    t = np.concatenate([buy_t, sell_t])
    m = np.concatenate([np.zeros(len(buy_t), int), np.ones(len(sell_t), int)])
    idx = np.argsort(t)
    return np.column_stack([t[idx], m[idx]])


# -----------------------------
# Full log-likelihood and gradient (optimize mu, aS, aC, beta)
# log-parameterization ensures positivity
# params = [logmu_buy, logmu_sell, logaS, logaC, logbeta]
# -----------------------------
def ll_grad_full_logparams(params: np.ndarray, events: np.ndarray, T: float):
    lmu1, lmu2, laS, laC, lbe = map(float, params)
    mu1 = math.exp(lmu1)
    mu2 = math.exp(lmu2)
    aS = math.exp(laS)
    aC = math.exp(laC)
    beta = math.exp(lbe)

    # stability: (aS+aC)/beta < 1
    if (aS + aC) >= 0.999 * beta:
        return -np.inf, np.zeros(5, float)

    ll = 0.0
    g_mu1 = 0.0
    g_mu2 = 0.0
    g_aS = 0.0
    g_aC = 0.0
    g_beta = 0.0  # d/d beta

    # states at start of interval (after previous event)
    s1 = s2 = 0.0
    ds1 = ds2 = 0.0  # derivatives wrt beta

    # u=s1+s2 for integral, and du/d beta
    u = 0.0
    du = 0.0
    Iu = 0.0
    dIu = 0.0

    t_prev = 0.0
    for tk, mk in events:
        dt = float(tk - t_prev)
        if dt < 0:
            return -np.inf, np.zeros(5, float)

        e = math.exp(-beta * dt) if dt != 0 else 1.0

        # integral over [t_prev, tk): u(t) = u_start * exp(-beta (t-t_prev))
        f = (1.0 - e) / beta
        df = (beta * (dt * e) - (1.0 - e)) / (beta * beta)  # d/dbeta of f
        Iu += u * f
        dIu += du * f + u * df

        # decay s1,s2 and their beta-derivatives
        s1_old, s2_old = s1, s2
        ds1_old, ds2_old = ds1, ds2

        s1 = s1_old * e
        s2 = s2_old * e
        ds1 = e * (ds1_old - dt * s1_old)
        ds2 = e * (ds2_old - dt * s2_old)

        lam1 = mu1 + aS * s1 + aC * s2
        lam2 = mu2 + aC * s1 + aS * s2
        if lam1 <= 0 or lam2 <= 0 or (not np.isfinite(lam1)) or (not np.isfinite(lam2)):
            return -np.inf, np.zeros(5, float)

        dlam1_db = aS * ds1 + aC * ds2
        dlam2_db = aC * ds1 + aS * ds2

        if int(mk) == 0:
            ll += math.log(lam1)
            inv = 1.0 / lam1
            g_mu1 += inv
            g_aS += s1 * inv
            g_aC += s2 * inv
            g_beta += dlam1_db * inv
            # jump
            s1 += 1.0
        else:
            ll += math.log(lam2)
            inv = 1.0 / lam2
            g_mu2 += inv
            g_aS += s2 * inv
            g_aC += s1 * inv
            g_beta += dlam2_db * inv
            # jump
            s2 += 1.0

        u = s1 + s2
        du = ds1 + ds2
        t_prev = float(tk)

    # last interval to T
    dt = float(T - t_prev)
    if dt < 0:
        return -np.inf, np.zeros(5, float)

    e = math.exp(-beta * dt) if dt != 0 else 1.0
    f = (1.0 - e) / beta
    df = (beta * (dt * e) - (1.0 - e)) / (beta * beta)
    Iu += u * f
    dIu += du * f + u * df

    # subtract integral of total intensity
    mu_tot = mu1 + mu2
    a_tot = aS + aC
    ll -= mu_tot * T + a_tot * Iu

    g_mu1 -= T
    g_mu2 -= T
    g_aS -= Iu
    g_aC -= Iu
    g_beta -= a_tot * dIu

    # chain rule to log-params
    g = np.array([
        g_mu1 * mu1,
        g_mu2 * mu2,
        g_aS * aS,
        g_aC * aC,
        g_beta * beta,
    ], float)
    return ll, g


def fit_full(events: np.ndarray, T: float, seed: int = 0):
    """
    Returns theta_hat = [mu1, mu2, aS, aC, beta], and ll_hat.
    """
    rng = np.random.default_rng(seed)
    n_buy = int(np.sum(events[:, 1] == 0))
    n_sell = int(np.sum(events[:, 1] == 1))
    mu1_0 = max(n_buy / T, 1e-6)
    mu2_0 = max(n_sell / T, 1e-6)
    beta0 = 10.0
    aS0 = 0.15 * beta0
    aC0 = 0.03 * beta0

    x0 = np.array([math.log(mu1_0), math.log(mu2_0), math.log(aS0), math.log(aC0), math.log(beta0)], float)

    # bounds in log-space
    bnds = [(-30, 10), (-30, 10), (-30, 10), (-30, 10), (math.log(1e-3), math.log(1e3))]

    def fun(x):
        ll, _ = ll_grad_full_logparams(x, events, T)
        return 1e100 if not np.isfinite(ll) else -ll

    def jac(x):
        ll, g = ll_grad_full_logparams(x, events, T)
        return np.zeros_like(x) if not np.isfinite(ll) else -g

    best = None
    best_res = None
    for k in range(4):
        x_init = x0.copy()
        if k > 0:
            x_init += rng.normal(0, 0.35, size=5)

        res = minimize(fun, x_init, jac=jac, method="L-BFGS-B", bounds=bnds, options={"maxiter": 1200})
        if best is None or res.fun < best:
            best = res.fun
            best_res = res

    lmu1, lmu2, laS, laC, lbe = map(float, best_res.x)
    theta = np.array([math.exp(lmu1), math.exp(lmu2), math.exp(laS), math.exp(laC), math.exp(lbe)], float)
    ll_hat = -float(best_res.fun)
    return theta, ll_hat


def fit_full_pooled(ev1: np.ndarray, ev2: np.ndarray, T: float):
    """
    Pooled MLE under H0: one theta for both halves.
    Returns theta_tilde, ll_sum.
    """
    # init from pooled counts
    n_buy = int(np.sum(ev1[:, 1] == 0) + np.sum(ev2[:, 1] == 0))
    n_sell = int(np.sum(ev1[:, 1] == 1) + np.sum(ev2[:, 1] == 1))
    mu1_0 = max(n_buy / (2 * T), 1e-6)
    mu2_0 = max(n_sell / (2 * T), 1e-6)
    beta0 = 10.0
    aS0 = 0.15 * beta0
    aC0 = 0.03 * beta0
    x0 = np.array([math.log(mu1_0), math.log(mu2_0), math.log(aS0), math.log(aC0), math.log(beta0)], float)
    bnds = [(-30, 10), (-30, 10), (-30, 10), (-30, 10), (math.log(1e-3), math.log(1e3))]

    def fun(x):
        ll1, _ = ll_grad_full_logparams(x, ev1, T)
        ll2, _ = ll_grad_full_logparams(x, ev2, T)
        ll = ll1 + ll2
        return 1e100 if not np.isfinite(ll) else -ll

    def jac(x):
        ll1, g1 = ll_grad_full_logparams(x, ev1, T)
        ll2, g2 = ll_grad_full_logparams(x, ev2, T)
        ll = ll1 + ll2
        return np.zeros_like(x) if not np.isfinite(ll) else -(g1 + g2)

    res = minimize(fun, x0, jac=jac, method="L-BFGS-B", bounds=bnds, options={"maxiter": 1600})
    lmu1, lmu2, laS, laC, lbe = map(float, res.x)
    theta = np.array([math.exp(lmu1), math.exp(lmu2), math.exp(laS), math.exp(laC), math.exp(lbe)], float)
    ll_hat = -float(res.fun)
    return theta, ll_hat


# -----------------------------
# Time-rescaling diagnostics (superposed process)
# -----------------------------
def time_rescaled_interarrivals_total(events: np.ndarray, mu1: float, mu2: float, aS: float, aC: float, beta: float) -> np.ndarray:
    """
    v_i = ∫_{t_{i-1}}^{t_i} lambda_total(u) du,
    where lambda_total(t) = (mu1+mu2) + (aS+aC) * u(t),
    u(t) = sum_{ti < t} exp(-beta (t - ti)) over ALL events.
    """
    mu_tot = mu1 + mu2
    a_tot = aS + aC
    t = events[:, 0].astype(float)
    v = np.empty(len(t), float)

    u = 0.0
    t_prev = 0.0
    for i, ti in enumerate(t):
        dt = float(ti - t_prev)
        e = math.exp(-beta * dt) if dt != 0 else 1.0
        v[i] = mu_tot * dt + a_tot * u * (1.0 - e) / beta
        u = u * e + 1.0
        t_prev = float(ti)
    return v


def save_rescaling_outputs(v: np.ndarray, prefix: str):
    v = np.asarray(v, float)
    u = 1.0 - np.exp(-v)

    out_csv = f"rescaled_{prefix}.csv"
    pd.DataFrame({"v_exp1": v, "u_unif01": u}).to_csv(out_csv, index=False)

    fig1 = f"{prefix}_hist_exp.png"
    plt.figure(figsize=(6, 4))
    plt.hist(v, bins=40, density=True)
    x = np.linspace(0, max(8, np.percentile(v, 99)), 400)
    plt.plot(x, expon.pdf(x, scale=1.0))
    plt.title(f"Rescaled inter-arrivals v_i — histogram vs Exp(1)\n{prefix}")
    plt.xlabel("v_i")
    plt.ylabel("density")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig1, dpi=160)
    plt.close()

    fig2 = f"{prefix}_qq_exp.png"
    plt.figure(figsize=(5, 5))
    v_sorted = np.sort(v)
    n = len(v_sorted)
    p = (np.arange(1, n + 1) - 0.5) / n
    q = expon.ppf(p, scale=1.0)
    plt.plot(q, v_sorted, marker=".", linestyle="None")
    mx = max(q[-1], v_sorted[-1])
    plt.plot([0, mx], [0, mx])
    plt.title(f"QQ plot: v_i vs Exp(1)\n{prefix}")
    plt.xlabel("Exp(1) quantiles")
    plt.ylabel("empirical quantiles")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig2, dpi=160)
    plt.close()

    fig3 = f"{prefix}_qq_unif.png"
    plt.figure(figsize=(5, 5))
    u_sorted = np.sort(u)
    plt.plot(p, u_sorted, marker=".", linestyle="None")
    plt.plot([0, 1], [0, 1])
    plt.title(f"QQ plot: 1-exp(-v_i) vs U(0,1)\n{prefix}")
    plt.xlabel("U(0,1) quantiles")
    plt.ylabel("empirical quantiles")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig3, dpi=160)
    plt.close()

    summary = {
        "n": int(len(v)),
        "mean_v": float(np.mean(v)),
        "median_v": float(np.median(v)),
        "q10_v": float(np.quantile(v, 0.10)),
        "q90_v": float(np.quantile(v, 0.90)),
        "q99_v": float(np.quantile(v, 0.99)),
    }
    ks_exp = kstest(v, "expon", args=(0.0, 1.0))
    ks_unif = kstest(u, "uniform", args=(0.0, 1.0))
    return out_csv, fig1, fig2, fig3, summary, ks_exp, ks_unif


# -----------------------------
# FIB bootstrap: fixed intensity path based on original observed history and theta_tilde
# -----------------------------
@dataclass
class FIBPrecomp:
    t_obs: np.ndarray
    m_obs: np.ndarray
    T: float
    mu1: float
    mu2: float
    aS: float
    aC: float
    beta: float
    s1_after: np.ndarray
    s2_after: np.ndarray
    t_grid: np.ndarray
    u_interval: np.ndarray
    Lam_grid: np.ndarray


def build_fib_precomp(events: np.ndarray, T: float, theta: np.ndarray) -> FIBPrecomp:
    mu1, mu2, aS, aC, beta = map(float, theta)
    t = events[:, 0].astype(float)
    m = events[:, 1].astype(int)
    n = len(t)

    s1_after = np.zeros(n, float)
    s2_after = np.zeros(n, float)

    s1 = s2 = 0.0
    t_prev = 0.0
    for i, (ti, mi) in enumerate(zip(t, m)):
        dt = float(ti - t_prev)
        e = math.exp(-beta * dt) if dt != 0 else 1.0
        s1 *= e
        s2 *= e
        if int(mi) == 0:
            s1 += 1.0
        else:
            s2 += 1.0
        s1_after[i] = s1
        s2_after[i] = s2
        t_prev = float(ti)

    t_grid = np.concatenate([[0.0], t, [float(T)]])
    u_interval = np.empty(len(t_grid) - 1, float)
    u_interval[0] = 0.0
    if n > 0:
        u_interval[1:] = s1_after + s2_after

    Lam = np.zeros_like(t_grid)
    mu_tot = mu1 + mu2
    a_tot = aS + aC
    for k in range(len(u_interval)):
        dt = float(t_grid[k + 1] - t_grid[k])
        e = math.exp(-beta * dt) if dt != 0 else 1.0
        inc = mu_tot * dt + a_tot * u_interval[k] * (1.0 - e) / beta
        Lam[k + 1] = Lam[k] + inc

    return FIBPrecomp(
        t_obs=t, m_obs=m, T=float(T),
        mu1=mu1, mu2=mu2, aS=aS, aC=aC, beta=beta,
        s1_after=s1_after, s2_after=s2_after,
        t_grid=t_grid, u_interval=u_interval, Lam_grid=Lam
    )


def S_at_vec(pre: FIBPrecomp, t_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t_query = np.asarray(t_query, float)
    idx = np.searchsorted(pre.t_obs, t_query, side="right") - 1
    s1 = np.zeros_like(t_query)
    s2 = np.zeros_like(t_query)
    mask = idx >= 0
    if np.any(mask):
        idxm = idx[mask]
        dt = t_query[mask] - pre.t_obs[idxm]
        decay = np.exp(-pre.beta * dt)
        s1[mask] = pre.s1_after[idxm] * decay
        s2[mask] = pre.s2_after[idxm] * decay
    return s1, s2


def invert_Lambda(pre: FIBPrecomp, s: float) -> float:
    Lam = pre.Lam_grid
    k = int(np.searchsorted(Lam, s, side="right") - 1)
    if k < 0:
        return 0.0
    if k >= len(pre.u_interval):
        return float(pre.T)

    s0 = float(Lam[k])
    target = float(s - s0)
    t0 = float(pre.t_grid[k])
    dt_max = float(pre.t_grid[k + 1] - pre.t_grid[k])
    u0 = float(pre.u_interval[k])

    mu_tot = pre.mu1 + pre.mu2
    a_tot = pre.aS + pre.aC
    beta = pre.beta

    lo, hi = 0.0, dt_max
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        e = math.exp(-beta * mid) if mid != 0 else 1.0
        val = mu_tot * mid + a_tot * u0 * (1.0 - e) / beta
        if val < target:
            lo = mid
        else:
            hi = mid
    return t0 + 0.5 * (lo + hi)


def simulate_FIB_marked(pre: FIBPrecomp, rng: np.random.Generator) -> np.ndarray:
    """
    Returns events* as (n,2) array: [time, mark], with mark 0=buy, 1=sell.
    """
    s = 0.0
    S_T = float(pre.Lam_grid[-1])
    s_list = []
    while True:
        s += float(rng.exponential(1.0))
        if s > S_T:
            break
        s_list.append(s)

    if not s_list:
        return np.zeros((0, 2), float)

    t_star = np.array([invert_Lambda(pre, si) for si in s_list], float)

    s1, s2 = S_at_vec(pre, t_star)
    lam1 = pre.mu1 + pre.aS * s1 + pre.aC * s2
    lam2 = pre.mu2 + pre.aC * s1 + pre.aS * s2
    p_buy = lam1 / (lam1 + lam2)

    u = rng.random(len(t_star))
    m_star = (u >= p_buy).astype(int)
    return np.column_stack([t_star, m_star])


def fib_bootstrap_LR(ev1_obs: np.ndarray, ev2_obs: np.ndarray, T: float,
                     theta0: np.ndarray, LR_obs: float, B: int, seed: int = 7):
    pre1 = build_fib_precomp(ev1_obs, T, theta0)
    pre2 = build_fib_precomp(ev2_obs, T, theta0)

    rng = np.random.default_rng(seed)
    LR = np.empty(B, float)

    if trange is None:
        # simple fallback
        it = range(B)
        def progress_iter():
            for b in it:
                if b % max(1, B // 10) == 0:
                    print(f"Bootstrap {b}/{B} ...")
                yield b
        iterator = progress_iter()
    else:
        iterator = trange(B, desc="FIB bootstrap", unit="rep")

    for b in iterator:
        ev1_star = simulate_FIB_marked(pre1, rng)
        ev2_star = simulate_FIB_marked(pre2, rng)

        th1, ll1 = fit_full(ev1_star, T, seed=int(rng.integers(0, 2**31 - 1)))
        th2, ll2 = fit_full(ev2_star, T, seed=int(rng.integers(0, 2**31 - 1)))
        th0, ll0 = fit_full_pooled(ev1_star, ev2_star, T)

        LR[b] = 2.0 * ((ll1 + ll2) - ll0)

    p_boot = (1.0 + float(np.sum(LR >= LR_obs))) / (B + 1.0)
    return LR, p_boot


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buy_csv", default=r"C:\Users\Wentao\Desktop\EA_recherche\Hawkes\sanofi_hawkes_buy_times.csv")
    ap.add_argument("--sell_csv", default=r"C:\Users\Wentao\Desktop\EA_recherche\Hawkes\sanofi_hawkes_sell_times.csv")
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip_diagnostics", action="store_true")
    args = ap.parse_args()

    t_buy = load_times_csv(args.buy_csv)
    t_sell = load_times_csv(args.sell_csv)

    T_end = float(max(t_buy[-1], t_sell[-1]))
    T_half = 0.5 * T_end

    buy1, buy2 = split_and_shift(t_buy, T_half)
    sell1, sell2 = split_and_shift(t_sell, T_half)

    ev1 = merge_events(buy1, sell1)
    ev2 = merge_events(buy2, sell2)

    # ---- Fit on each half ----
    th1, ll1 = fit_full(ev1, T_half, seed=1)
    th2, ll2 = fit_full(ev2, T_half, seed=2)
    th0, ll0 = fit_full_pooled(ev1, ev2, T_half)

    LR_obs = 2.0 * ((ll1 + ll2) - ll0)
    p_theory = chi2.sf(LR_obs, df=5)

    print("\n=== FULL MLE (mu, aS, aC, beta optimized) ===")
    print("Half 1 theta =", th1, " ll =", ll1)
    print("Half 2 theta =", th2, " ll =", ll2)
    print("H0 pooled theta =", th0, " ll_sum =", ll0)
    print(f"LR_obs = {LR_obs:.6f}   asymptotic p (chi2, df=5) = {p_theory:.3e}")

    # ---- Diagnostics ----
    if not args.skip_diagnostics:
        v1 = time_rescaled_interarrivals_total(ev1, th1[0], th1[1], th1[2], th1[3], th1[4])
        v2 = time_rescaled_interarrivals_total(ev2, th2[0], th2[1], th2[2], th2[3], th2[4])

        out1 = save_rescaling_outputs(v1, prefix="half1_fullopt")
        out2 = save_rescaling_outputs(v2, prefix="half2_fullopt")

        print("\n=== Time-rescaling diagnostics (saved) ===")
        print("Half 1:", out1[0], out1[1], out1[2], out1[3])
        print("Half 1 summary:", out1[4])
        print("Half 1 KS Exp(1):", out1[5])
        print("Half 1 KS Unif(0,1):", out1[6])

        print("\nHalf 2:", out2[0], out2[1], out2[2], out2[3])
        print("Half 2 summary:", out2[4])
        print("Half 2 KS Exp(1):", out2[5])
        print("Half 2 KS Unif(0,1):", out2[6])

    # ---- FIB bootstrap under H0 ----
    LR_star, p_boot = fib_bootstrap_LR(ev1, ev2, T_half, th0, LR_obs, B=args.B, seed=args.seed)
    np.savetxt("LR_star_FIB_fullopt.csv", LR_star, delimiter=",")

    print("\n=== FIB bootstrap (FULL MLE refit) ===")
    print(f"B = {args.B}  seed = {args.seed}")
    print(f"p_boot (FIB) = {p_boot:.6f}")
    print("Saved: LR_star_FIB_fullopt.csv")

    plt.figure(figsize=(6, 4))
    plt.hist(LR_star, bins=40, density=True)
    plt.axvline(LR_obs)
    plt.title(f"FIB bootstrap LR* (full opt) — B={args.B}")
    plt.xlabel("LR*")
    plt.ylabel("density")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("LR_star_FIB_fullopt_hist.png", dpi=160)
    plt.close()
    print("Saved: LR_star_FIB_fullopt_hist.png")


if __name__ == "__main__":
    main()
