# -*- coding: utf-8 -*-
"""
Sanofi (one day) — Hawkes structural break test + FIB bootstrap + time-rescaling diagnostics

Inputs (CSV):
  - sanofi_hawkes_buy_times.csv  with column: t_seconds_jitter
  - sanofi_hawkes_sell_times.csv with column: t_seconds_jitter

Model (bivariate, symmetric Hawkes with exponential kernel, beta fixed):
  lambda_buy(t)  = mu_buy  + aS * S_buy(t)  + aC * S_sell(t)
  lambda_sell(t) = mu_sell + aC * S_buy(t)  + aS * S_sell(t)
  S_m(t) = sum_{ti^m < t} exp(-beta (t - ti^m))

Test (two halves):
  H0: same parameter vector theta on both halves
  H1: different parameters on each half
  Use LR statistic with chi-square(df=4) asymptotic p-value (theoretical)
  and Fixed-Intensity Bootstrap (FIB) p-value.

Diagnostics:
  Ogata time-rescaling residuals using total intensity (superposed process)
  v_i = ∫_{t_{i-1}}^{t_i} lambda_total(u) du, lambda_total = lambda_buy + lambda_sell
  Under correct model, v_i ≈ i.i.d. Exp(1), and u_i = 1-exp(-v_i) ≈ U(0,1).

References:
  Cavaliere et al. (2023) "Bootstrap inference for Hawkes and general point processes"
  - Algorithm 1 (FIB): generate Exp(1) waiting times in transformed time and invert Λ-hat.
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
from tqdm import tqdm


# -----------------------------
# Helpers: event construction
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
# "Real" MLE (uses the sample history itself)
# -----------------------------
def ll_grad_symmetric(theta: np.ndarray, events: np.ndarray, T: float, beta: float):
    """
    Symmetric bivariate Hawkes log-likelihood and gradient (beta fixed),
    where intensity depends on the *same* events passed in 'events'.
    """
    mu1, mu2, aS, aC = map(float, theta)
    if mu1 <= 0 or mu2 <= 0 or aS < 0 or aC < 0 or beta <= 0:
        return -np.inf, np.zeros(4, float)
    if (aS + aC) >= 0.999 * beta:
        return -np.inf, np.zeros(4, float)

    s1 = s2 = 0.0
    t_prev = 0.0
    ll = 0.0
    g = np.zeros(4, float)

    for tk, mk in events:
        dt = float(tk - t_prev)
        if dt < 0:
            return -np.inf, np.zeros(4, float)
        decay = math.exp(-beta * dt) if dt != 0 else 1.0
        s1 *= decay
        s2 *= decay

        lam1 = mu1 + aS * s1 + aC * s2
        lam2 = mu2 + aC * s1 + aS * s2
        if lam1 <= 0 or lam2 <= 0 or (not np.isfinite(lam1)) or (not np.isfinite(lam2)):
            return -np.inf, np.zeros(4, float)

        if int(mk) == 0:
            ll += math.log(lam1)
            inv = 1.0 / lam1
            g[0] += inv
            g[2] += s1 * inv
            g[3] += s2 * inv
            s1 += 1.0
        else:
            ll += math.log(lam2)
            inv = 1.0 / lam2
            g[1] += inv
            g[2] += s2 * inv
            g[3] += s1 * inv
            s2 += 1.0

        t_prev = float(tk)

    # Integral of total intensity: (mu1+mu2)T + (aS+aC) ∫ u(t)dt, u=s1+s2
    mu_tot = mu1 + mu2
    a_tot = aS + aC

    u = 0.0
    t_prev = 0.0
    Iu = 0.0
    for tk, _mk in events:
        dt = float(tk - t_prev)
        Iu += u * (1.0 - math.exp(-beta * dt)) / beta
        u = u * math.exp(-beta * dt) + 1.0
        t_prev = float(tk)

    dt = float(T - t_prev)
    Iu += u * (1.0 - math.exp(-beta * dt)) / beta

    ll -= mu_tot * T + a_tot * Iu
    g[0] -= T
    g[1] -= T
    g[2] -= Iu
    g[3] -= Iu

    return ll, g


def fit_sym(events: np.ndarray, T: float, beta: float, seed: int = 0) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n_buy = int(np.sum(events[:, 1] == 0))
    n_sell = int(np.sum(events[:, 1] == 1))
    mu1_0 = max(n_buy / T, 1e-6)
    mu2_0 = max(n_sell / T, 1e-6)
    x0 = np.array([mu1_0, mu2_0, 0.15 * beta, 0.03 * beta], float)
    bounds = [(1e-8, None), (1e-8, None), (0.0, None), (0.0, None)]

    def fun(x):
        ll, _ = ll_grad_symmetric(x, events, T, beta)
        return 1e100 if not np.isfinite(ll) else -ll

    def jac(x):
        ll, g = ll_grad_symmetric(x, events, T, beta)
        return np.zeros_like(x) if not np.isfinite(ll) else -g

    best = None
    best_res = None
    for k in range(3):
        x_init = x0.copy()
        if k > 0:
            x_init[0] *= float(np.exp(rng.normal(0, 0.3)))
            x_init[1] *= float(np.exp(rng.normal(0, 0.3)))
            x_init[2] *= float(np.exp(rng.normal(0, 0.4)))
            x_init[3] *= float(np.exp(rng.normal(0, 0.4)))

        res = minimize(fun, x_init, jac=jac, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
        if best is None or res.fun < best:
            best = res.fun
            best_res = res

    return best_res.x, -best_res.fun


def fit_sym_restricted(ev1: np.ndarray, ev2: np.ndarray, T: float, beta: float) -> tuple[np.ndarray, float]:
    n_buy = int(np.sum(ev1[:, 1] == 0) + np.sum(ev2[:, 1] == 0))
    n_sell = int(np.sum(ev1[:, 1] == 1) + np.sum(ev2[:, 1] == 1))
    mu1_0 = max(n_buy / (2 * T), 1e-6)
    mu2_0 = max(n_sell / (2 * T), 1e-6)
    x0 = np.array([mu1_0, mu2_0, 0.15 * beta, 0.03 * beta], float)
    bounds = [(1e-8, None), (1e-8, None), (0.0, None), (0.0, None)]

    def fun(x):
        ll1, _ = ll_grad_symmetric(x, ev1, T, beta)
        ll2, _ = ll_grad_symmetric(x, ev2, T, beta)
        ll = ll1 + ll2
        return 1e100 if not np.isfinite(ll) else -ll

    def jac(x):
        ll1, g1 = ll_grad_symmetric(x, ev1, T, beta)
        ll2, g2 = ll_grad_symmetric(x, ev2, T, beta)
        ll = ll1 + ll2
        return np.zeros_like(x) if not np.isfinite(ll) else -(g1 + g2)

    res = minimize(fun, x0, jac=jac, method="L-BFGS-B", bounds=bounds, options={"maxiter": 600})
    return res.x, -res.fun


# -----------------------------
# Time-rescaling diagnostics (superposed process)
# -----------------------------
def time_rescaled_interarrivals(events: np.ndarray, beta: float, theta: np.ndarray) -> np.ndarray:
    mu1, mu2, aS, aC = map(float, theta)
    mu_tot = mu1 + mu2
    a_tot = aS + aC

    t = events[:, 0].astype(float)
    v = np.empty(len(t), float)

    u = 0.0
    t_prev = 0.0
    for i, ti in enumerate(t):
        w = float(ti - t_prev)
        v[i] = mu_tot * w + a_tot * u * (1.0 - math.exp(-beta * w)) / beta
        u = u * math.exp(-beta * w) + 1.0
        t_prev = float(ti)
    return v


def save_rescaling_plots(v: np.ndarray, prefix: str) -> dict:
    out = {}

    # Histogram vs Exp(1)
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
    out["hist"] = fig1

    # QQ vs Exp(1)
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
    out["qq_exp"] = fig2

    # Uniform transform
    u = 1.0 - np.exp(-v)
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
    out["qq_unif"] = fig3

    # KS tests (heuristic: parameters are estimated)
    out["ks_exp"] = kstest(v, "expon", args=(0.0, 1.0))
    out["ks_unif"] = kstest(u, "uniform", args=(0.0, 1.0))
    return out


# -----------------------------
# Fixed-Intensity Bootstrap (FIB) — fixed design precomputation
# -----------------------------
@dataclass
class FixedDesignPrecomp:
    t_events: np.ndarray
    m_events: np.ndarray
    T: float
    beta: float
    s1_after: np.ndarray
    s2_after: np.ndarray
    u_after: np.ndarray
    Iu: float
    t_grid: np.ndarray
    u_interval: np.ndarray


def build_fixed_design_precomp(events: np.ndarray, T: float, beta: float) -> FixedDesignPrecomp:
    t = events[:, 0].astype(float)
    m = events[:, 1].astype(int)
    n = len(t)

    s1_after = np.zeros(n, float)
    s2_after = np.zeros(n, float)
    u_after = np.zeros(n, float)

    s1 = s2 = 0.0
    t_prev = 0.0
    for i, (ti, mi) in enumerate(zip(t, m)):
        dt = float(ti - t_prev)
        decay = math.exp(-beta * dt) if dt != 0 else 1.0
        s1 *= decay
        s2 *= decay
        if int(mi) == 0:
            s1 += 1.0
        else:
            s2 += 1.0
        s1_after[i] = s1
        s2_after[i] = s2
        u_after[i] = s1 + s2
        t_prev = float(ti)

    # Iu = ∫ u(t) dt using original history
    Iu = 0.0
    u = 0.0
    t_prev = 0.0
    for ti in t:
        dt = float(ti - t_prev)
        Iu += u * (1.0 - math.exp(-beta * dt)) / beta
        u = u * math.exp(-beta * dt) + 1.0
        t_prev = float(ti)
    dt = float(T - t_prev)
    Iu += u * (1.0 - math.exp(-beta * dt)) / beta

    t_grid = np.concatenate([[0.0], t, [float(T)]])
    u_interval = np.empty(len(t_grid) - 1, float)  # n+1
    u_interval[0] = 0.0
    if n > 0:
        u_interval[1:] = u_after  # start of each post-event interval
    return FixedDesignPrecomp(t_events=t, m_events=m, T=float(T), beta=float(beta),
                              s1_after=s1_after, s2_after=s2_after, u_after=u_after,
                              Iu=float(Iu), t_grid=t_grid, u_interval=u_interval)


def S_at_vec(pre: FixedDesignPrecomp, t_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t_star = np.asarray(t_star, float)
    idx = np.searchsorted(pre.t_events, t_star, side="right") - 1
    s1 = np.zeros_like(t_star)
    s2 = np.zeros_like(t_star)
    mask = idx >= 0
    if np.any(mask):
        idxm = idx[mask]
        dt = t_star[mask] - pre.t_events[idxm]
        decay = np.exp(-pre.beta * dt)
        s1[mask] = pre.s1_after[idxm] * decay
        s2[mask] = pre.s2_after[idxm] * decay
    return s1, s2


def fixed_loglik_and_grad(theta: np.ndarray, t_star: np.ndarray, m_star: np.ndarray, pre: FixedDesignPrecomp):
    mu1, mu2, aS, aC = map(float, theta)
    beta = pre.beta
    if mu1 <= 0 or mu2 <= 0 or aS < 0 or aC < 0 or beta <= 0 or (aS + aC) >= 0.999 * beta:
        return -np.inf, np.zeros(4, float)

    t_star = np.asarray(t_star, float)
    m_star = np.asarray(m_star, int)

    if len(t_star) == 0:
        ll = - (mu1 + mu2) * pre.T - (aS + aC) * pre.Iu
        g = np.array([-pre.T, -pre.T, -pre.Iu, -pre.Iu], float)
        return ll, g

    s1, s2 = S_at_vec(pre, t_star)
    lam1 = mu1 + aS * s1 + aC * s2
    lam2 = mu2 + aC * s1 + aS * s2
    if np.any(lam1 <= 0) or np.any(lam2 <= 0) or (not np.all(np.isfinite(lam1))) or (not np.all(np.isfinite(lam2))):
        return -np.inf, np.zeros(4, float)

    buy = (m_star == 0)
    sell = ~buy

    ll = float(np.sum(np.log(lam1[buy])) + np.sum(np.log(lam2[sell])))

    g = np.zeros(4, float)
    inv1 = 1.0 / lam1[buy] if np.any(buy) else np.zeros(0, float)
    inv2 = 1.0 / lam2[sell] if np.any(sell) else np.zeros(0, float)
    g[0] = float(np.sum(inv1))
    g[1] = float(np.sum(inv2))
    g[2] = float(np.sum(s1[buy] * inv1) + np.sum(s2[sell] * inv2))
    g[3] = float(np.sum(s2[buy] * inv1) + np.sum(s1[sell] * inv2))

    # subtract integral term and its derivatives
    ll -= (mu1 + mu2) * pre.T + (aS + aC) * pre.Iu
    g[0] -= pre.T
    g[1] -= pre.T
    g[2] -= pre.Iu
    g[3] -= pre.Iu
    return ll, g


def fit_fixed_design(t_star: np.ndarray, m_star: np.ndarray, pre: FixedDesignPrecomp, x_init: np.ndarray,
                     maxiter: int = 120) -> tuple[np.ndarray, float]:
    bounds = [(1e-8, None), (1e-8, None), (0.0, None), (0.0, None)]

    def fun(x):
        ll, _ = fixed_loglik_and_grad(x, t_star, m_star, pre)
        return 1e100 if not np.isfinite(ll) else -ll

    def jac(x):
        ll, g = fixed_loglik_and_grad(x, t_star, m_star, pre)
        return np.zeros_like(x) if not np.isfinite(ll) else -g

    res = minimize(fun, x_init, jac=jac, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})
    return res.x, -res.fun


def fit_fixed_design_pooled(pre1: FixedDesignPrecomp, pre2: FixedDesignPrecomp,
                            t1: np.ndarray, m1: np.ndarray, t2: np.ndarray, m2: np.ndarray,
                            x_init: np.ndarray, maxiter: int = 150) -> tuple[np.ndarray, float]:
    bounds = [(1e-8, None), (1e-8, None), (0.0, None), (0.0, None)]

    def fun(x):
        ll1, _ = fixed_loglik_and_grad(x, t1, m1, pre1)
        ll2, _ = fixed_loglik_and_grad(x, t2, m2, pre2)
        ll = ll1 + ll2
        return 1e100 if not np.isfinite(ll) else -ll

    def jac(x):
        ll1, g1 = fixed_loglik_and_grad(x, t1, m1, pre1)
        ll2, g2 = fixed_loglik_and_grad(x, t2, m2, pre2)
        ll = ll1 + ll2
        return np.zeros_like(x) if not np.isfinite(ll) else -(g1 + g2)

    res = minimize(fun, x_init, jac=jac, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})
    return res.x, -res.fun


# -----------------------------
# FIB simulation: invert Λ-hat and sample marks
# -----------------------------
def precompute_Lambda_grid(pre: FixedDesignPrecomp, theta_star: np.ndarray) -> np.ndarray:
    mu1, mu2, aS, aC = map(float, theta_star)
    mu_tot = mu1 + mu2
    a_tot = aS + aC

    Lam = np.zeros_like(pre.t_grid)
    for k in range(len(pre.u_interval)):
        dt = float(pre.t_grid[k + 1] - pre.t_grid[k])
        inc = mu_tot * dt + a_tot * pre.u_interval[k] * (1.0 - math.exp(-pre.beta * dt)) / pre.beta
        Lam[k + 1] = Lam[k] + inc
    return Lam


def invert_Lambda(pre: FixedDesignPrecomp, Lam_grid: np.ndarray, theta_star: np.ndarray, s: float) -> float:
    # Find interval k such that Lam_grid[k] <= s < Lam_grid[k+1]
    k = int(np.searchsorted(Lam_grid, s, side="right") - 1)
    if k < 0:
        return 0.0
    if k >= len(pre.u_interval):
        return float(pre.T)

    s0 = float(Lam_grid[k])
    target = float(s - s0)

    mu1, mu2, aS, aC = map(float, theta_star)
    mu_tot = mu1 + mu2
    a_tot = aS + aC
    u0 = float(pre.u_interval[k])
    dt_max = float(pre.t_grid[k + 1] - pre.t_grid[k])

    # Monotone root: bisection
    lo, hi = 0.0, dt_max
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        val = mu_tot * mid + a_tot * u0 * (1.0 - math.exp(-pre.beta * mid)) / pre.beta
        if val < target:
            lo = mid
        else:
            hi = mid
    return float(pre.t_grid[k] + 0.5 * (lo + hi))


def simulate_FIB_marked(pre: FixedDesignPrecomp, theta_star: np.ndarray, Lam_grid: np.ndarray,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a marked inhomogeneous Poisson process on [0,T] with (fixed) intensity path:
      lambda_hat(t) = lambda(t; theta_star) computed from the *original* history (FIB).

    Steps (mirrors Algorithm 1 idea):
      - generate s_k* by accumulating Exp(1) waiting times
      - map t_k* = Λ_hat^{-1}(s_k*)
      - draw mark using component probabilities lambda_buy_hat / (lambda_buy_hat + lambda_sell_hat)
    """
    s = 0.0
    S_T = float(Lam_grid[-1])
    s_list = []
    while True:
        s += float(rng.exponential(1.0))
        if s > S_T:
            break
        s_list.append(s)

    if not s_list:
        return np.zeros(0, float), np.zeros(0, int)

    t_star = np.array([invert_Lambda(pre, Lam_grid, theta_star, si) for si in s_list], float)

    # marks
    s1, s2 = S_at_vec(pre, t_star)
    mu1, mu2, aS, aC = map(float, theta_star)
    lam1 = mu1 + aS * s1 + aC * s2
    lam2 = mu2 + aC * s1 + aS * s2
    p_buy = lam1 / (lam1 + lam2)

    u = rng.random(len(t_star))
    m_star = (u >= p_buy).astype(int)  # 0=buy, 1=sell
    return t_star, m_star


def fib_bootstrap_LR(pre1: FixedDesignPrecomp, pre2: FixedDesignPrecomp,
                     Lam1: np.ndarray, Lam2: np.ndarray,
                     theta_star: np.ndarray, LR_obs: float,
                     B: int, seed: int = 1234) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    LR = np.empty(B, float)

    for b in tqdm(range(B), desc="Running FIB bootstrap"):
        t1, m1 = simulate_FIB_marked(pre1, theta_star, Lam1, rng)
        t2, m2 = simulate_FIB_marked(pre2, theta_star, Lam2, rng)

        x1, ll1 = fit_fixed_design(t1, m1, pre1, theta_star)
        x2, ll2 = fit_fixed_design(t2, m2, pre2, theta_star)
        x0, ll0 = fit_fixed_design_pooled(pre1, pre2, t1, m1, t2, m2, theta_star)

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
    ap.add_argument("--beta", type=float, default=5.0, help="Exponential kernel decay beta (fixed)")
    ap.add_argument("--B", type=int, default=500, help="Number of bootstrap replications")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--do_diagnostics", action="store_true", help="Save time-rescaling diagnostic plots")
    args = ap.parse_args()

    t_buy = load_times_csv(args.buy_csv)
    t_sell = load_times_csv(args.sell_csv)

    T_end = float(max(t_buy[-1], t_sell[-1]))
    T_half = 0.5 * T_end

    buy1, buy2 = split_and_shift(t_buy, T_half)
    sell1, sell2 = split_and_shift(t_sell, T_half)

    ev1 = merge_events(buy1, sell1)
    ev2 = merge_events(buy2, sell2)

    # ---- Estimate (real MLE) ----
    x1_hat, ll1_hat = fit_sym(ev1, T_half, args.beta, seed=1)
    x2_hat, ll2_hat = fit_sym(ev2, T_half, args.beta, seed=2)
    x0_hat, ll0_hat = fit_sym_restricted(ev1, ev2, T_half, args.beta)

    LR_obs = 2.0 * ((ll1_hat + ll2_hat) - ll0_hat)
    p_theory = chi2.sf(LR_obs, df=4)

    print("\n=== Real MLE (two halves) ===")
    print("beta =", args.beta)
    print("theta_hat_1 =", x1_hat)
    print("theta_hat_2 =", x2_hat)
    print("theta_tilde (H0 pooled) =", x0_hat)
    print(f"LR_obs = {LR_obs:.6f}   asymptotic p (chi2, df=4) = {p_theory:.3e}")

    # ---- Diagnostics (optional) ----
    if args.do_diagnostics:
        v1 = time_rescaled_interarrivals(ev1, args.beta, x1_hat)
        v2 = time_rescaled_interarrivals(ev2, args.beta, x2_hat)

        out1 = save_rescaling_plots(v1, prefix="half1")
        out2 = save_rescaling_plots(v2, prefix="half2")

        print("\n=== Time-rescaling diagnostics (heuristic KS tests) ===")
        print("Half 1 KS Exp(1):", out1["ks_exp"])
        print("Half 1 KS Unif(0,1):", out1["ks_unif"])
        print("Half 2 KS Exp(1):", out2["ks_exp"])
        print("Half 2 KS Unif(0,1):", out2["ks_unif"])
        print("Saved plots:", out1["hist"], out1["qq_exp"], out1["qq_unif"], out2["hist"], out2["qq_exp"], out2["qq_unif"])

    # ---- FIB bootstrap under H0 ----
    pre1 = build_fixed_design_precomp(ev1, T_half, args.beta)
    pre2 = build_fixed_design_precomp(ev2, T_half, args.beta)
    Lam1 = precompute_Lambda_grid(pre1, x0_hat)
    Lam2 = precompute_Lambda_grid(pre2, x0_hat)

    LR_star, p_boot = fib_bootstrap_LR(pre1, pre2, Lam1, Lam2, x0_hat, LR_obs, B=args.B, seed=args.seed)

    np.savetxt("LR_star_FIB.csv", LR_star, delimiter=",")
    print("\n=== FIB bootstrap (restricted, H0) ===")
    print(f"B = {args.B}  seed = {args.seed}")
    print(f"p_boot (FIB) = {p_boot:.6f}")
    print("Saved: LR_star_FIB.csv")

    # Histogram
    plt.figure(figsize=(6, 4))
    plt.hist(LR_star, bins=40, density=True)
    plt.axvline(LR_obs)
    plt.title(f"FIB bootstrap LR* (B={args.B}) — observed LR shown as vertical line")
    plt.xlabel("LR*")
    plt.ylabel("density")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("LR_star_FIB_hist.png", dpi=160)
    plt.close()
    print("Saved: LR_star_FIB_hist.png")


if __name__ == "__main__":
    main()
