# -*- coding: utf-8 -*-
"""
Sanofi (one day) — Univariate Hawkes (no buy/sell cross-dependence) change test with FIB bootstrap
FULL MLE: optimize (mu, alpha, beta), all constants in time.

Interpretation (your request)
-----------------------------
- We REMOVE buy/sell interaction by IGNORING marks.
- We fit ONE univariate Hawkes process to the superposed events (buys ∪ sells).
  Intensity depends only on its own past:
      lambda(t) = mu + alpha * sum_{ti < t} exp(-beta (t - ti))

Tasks covered
-------------
1) Build event times from buy+sell CSVs, merge and sort.
2) Split the window into two equal halves.
3) Estimate a Hawkes (mu, alpha, beta) on each half (MLE).
4) LR test of parameter change:
      H0: same (mu, alpha, beta) on both halves
      H1: different (mu, alpha, beta) on each half
   Theoretical p-value: chi2(df=3).
5) FIB bootstrap (Fixed-Intensity Bootstrap, Cavaliere et al. 2023):
   - Estimate theta_tilde under H0 (pooled MLE).
   - For each half, build fixed estimated Λ_hat(t) based on ORIGINAL observed history in that half and theta_tilde.
   - Generate bootstrap times via Exp(1) in transformed time and inversion Λ_hat^{-1}.
   - Refit Hawkes (mu, alpha, beta) on each bootstrap sample and compute LR*.
   - Bootstrap p-value from empirical LR* distribution.
   A progress bar is shown during the bootstrap loop.
6) Goodness-of-fit (enabled by default):
   Ogata time-rescaling residuals:
      v_i = ∫_{t_{i-1}}^{t_i} lambda(u) du  should be ~ i.i.d. Exp(1),
      u_i = 1-exp(-v_i) should be ~ U(0,1).
   Saves CSV + histogram/QQ plots.

Inputs
------
- sanofi_hawkes_buy_times.csv  (column: t_seconds_jitter)
- sanofi_hawkes_sell_times.csv (column: t_seconds_jitter)

Run
---
python sanofi_hawkes_FIB_univariate_no_cross.py --B 1000 --seed 7
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
# I/O and time utilities
# -----------------------------
def load_times_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    if "t_seconds_jitter" not in df.columns:
        raise ValueError(f"Missing column t_seconds_jitter in {path}")
    t = np.sort(df["t_seconds_jitter"].to_numpy(float))
    if len(t) == 0:
        raise ValueError(f"No times found in {path}")
    return t


def merge_univariate(buy_t: np.ndarray, sell_t: np.ndarray) -> np.ndarray:
    t = np.concatenate([buy_t, sell_t]).astype(float)
    t.sort()
    return t


def split_and_shift(times: np.ndarray, split: float) -> tuple[np.ndarray, np.ndarray]:
    return times[times <= split].copy(), (times[times > split].copy() - split)


# -----------------------------
# Univariate Hawkes log-likelihood and gradient (log-parameterization)
# params = [logmu, logalpha, logbeta]  -> mu, alpha, beta > 0
# Stationarity: alpha/beta < 1
# -----------------------------
def ll_grad_uni_logparams(params: np.ndarray, t: np.ndarray, T: float):
    lmu, la, lb = map(float, params)
    mu = math.exp(lmu)
    alpha = math.exp(la)
    beta = math.exp(lb)

    # stability (branching ratio)
    if alpha >= 0.999 * beta:
        return -np.inf, np.zeros(3, float)

    t = np.asarray(t, float)
    n = len(t)

    ll = 0.0
    g_mu = 0.0
    g_alpha = 0.0
    g_beta = 0.0  # d/d beta

    # s(t): self-excitation state at start of interval; ds = d s / d beta
    s = 0.0
    ds = 0.0

    # I = ∫ s(u) du ; dI = dI/d beta
    I = 0.0
    dI = 0.0

    t_prev = 0.0
    for i in range(n):
        ti = float(t[i])
        dt = ti - t_prev
        if dt < 0:
            return -np.inf, np.zeros(3, float)

        e = math.exp(-beta * dt) if dt != 0 else 1.0

        # integral over interval: s_start * (1-e)/beta
        f = (1.0 - e) / beta
        # df/dbeta = [beta*(dt*e) - (1-e)] / beta^2
        df = (beta * (dt * e) - (1.0 - e)) / (beta * beta)

        I += s * f
        dI += ds * f + s * df

        # decay state and its beta-derivative
        s_old, ds_old = s, ds
        s = s_old * e
        ds = e * (ds_old - dt * s_old)

        lam = mu + alpha * s
        if lam <= 0 or (not np.isfinite(lam)):
            return -np.inf, np.zeros(3, float)

        ll += math.log(lam)

        inv = 1.0 / lam
        g_mu += inv
        g_alpha += s * inv
        g_beta += (alpha * ds) * inv

        # jump at event
        s += 1.0
        # ds unchanged by +1

        t_prev = ti

    # last interval to T
    dt = float(T - t_prev)
    if dt < 0:
        return -np.inf, np.zeros(3, float)

    e = math.exp(-beta * dt) if dt != 0 else 1.0
    f = (1.0 - e) / beta
    df = (beta * (dt * e) - (1.0 - e)) / (beta * beta)
    I += s * f
    dI += ds * f + s * df

    # subtract integral of intensity: mu*T + alpha*I
    ll -= mu * T + alpha * I
    g_mu -= T
    g_alpha -= I
    g_beta -= alpha * dI

    # chain rule to log-params
    grad = np.array([g_mu * mu, g_alpha * alpha, g_beta * beta], float)
    return ll, grad


def fit_uni(t: np.ndarray, T: float, seed: int = 0):
    """
    Returns theta_hat = [mu, alpha, beta], and ll_hat.
    """
    rng = np.random.default_rng(seed)
    n = len(t)
    mu0 = max(n / T, 1e-6)
    beta0 = 10.0
    alpha0 = 0.2 * beta0  # branching ratio ~0.2

    x0 = np.array([math.log(mu0), math.log(alpha0), math.log(beta0)], float)
    bnds = [(-30, 10), (-30, 10), (math.log(1e-3), math.log(1e3))]

    def fun(x):
        ll, _ = ll_grad_uni_logparams(x, t, T)
        return 1e100 if not np.isfinite(ll) else -ll

    def jac(x):
        ll, g = ll_grad_uni_logparams(x, t, T)
        return np.zeros_like(x) if not np.isfinite(ll) else -g

    best = None
    best_res = None
    for k in range(5):
        x_init = x0.copy()
        if k > 0:
            x_init += rng.normal(0, 0.35, size=3)
        res = minimize(fun, x_init, jac=jac, method="L-BFGS-B", bounds=bnds, options={"maxiter": 1200})
        if best is None or res.fun < best:
            best = res.fun
            best_res = res

    lmu, la, lb = map(float, best_res.x)
    theta = np.array([math.exp(lmu), math.exp(la), math.exp(lb)], float)
    ll_hat = -float(best_res.fun)
    return theta, ll_hat


def fit_uni_pooled(t1: np.ndarray, t2: np.ndarray, T: float):
    """
    Pooled MLE under H0: one theta for both halves.
    Returns theta_tilde, ll_sum.
    """
    n = len(t1) + len(t2)
    mu0 = max(n / (2 * T), 1e-6)
    beta0 = 10.0
    alpha0 = 0.2 * beta0

    x0 = np.array([math.log(mu0), math.log(alpha0), math.log(beta0)], float)
    bnds = [(-30, 10), (-30, 10), (math.log(1e-3), math.log(1e3))]

    def fun(x):
        ll1, _ = ll_grad_uni_logparams(x, t1, T)
        ll2, _ = ll_grad_uni_logparams(x, t2, T)
        ll = ll1 + ll2
        return 1e100 if not np.isfinite(ll) else -ll

    def jac(x):
        ll1, g1 = ll_grad_uni_logparams(x, t1, T)
        ll2, g2 = ll_grad_uni_logparams(x, t2, T)
        ll = ll1 + ll2
        return np.zeros_like(x) if not np.isfinite(ll) else -(g1 + g2)

    res = minimize(fun, x0, jac=jac, method="L-BFGS-B", bounds=bnds, options={"maxiter": 1600})
    lmu, la, lb = map(float, res.x)
    theta = np.array([math.exp(lmu), math.exp(la), math.exp(lb)], float)
    ll_hat = -float(res.fun)
    return theta, ll_hat


# -----------------------------
# Time-rescaling diagnostics
# -----------------------------
def rescaled_interarrivals(t: np.ndarray, T: float, mu: float, alpha: float, beta: float) -> np.ndarray:
    """
    v_i = ∫_{t_{i-1}}^{t_i} lambda(u) du,
    where lambda(t) = mu + alpha * s(t), s(t)=sum exp(-beta (t-ti)).
    """
    t = np.asarray(t, float)
    v = np.empty(len(t), float)

    s = 0.0
    t_prev = 0.0
    for i, ti in enumerate(t):
        dt = float(ti - t_prev)
        e = math.exp(-beta * dt) if dt != 0 else 1.0
        v[i] = mu * dt + alpha * s * (1.0 - e) / beta
        s = s * e + 1.0
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
# FIB precomputation and simulation (univariate)
# -----------------------------
@dataclass
class FIBPrecomp:
    t_obs: np.ndarray
    T: float
    mu: float
    alpha: float
    beta: float
    s_after: np.ndarray
    t_grid: np.ndarray
    s_interval: np.ndarray
    Lam_grid: np.ndarray


def build_fib_precomp(t_obs: np.ndarray, T: float, theta: np.ndarray) -> FIBPrecomp:
    mu, alpha, beta = map(float, theta)
    t = np.asarray(t_obs, float)
    n = len(t)

    s_after = np.zeros(n, float)
    s = 0.0
    t_prev = 0.0
    for i, ti in enumerate(t):
        dt = float(ti - t_prev)
        e = math.exp(-beta * dt) if dt != 0 else 1.0
        s = s * e + 1.0
        s_after[i] = s
        t_prev = float(ti)

    t_grid = np.concatenate([[0.0], t, [float(T)]])
    s_interval = np.empty(len(t_grid) - 1, float)
    s_interval[0] = 0.0
    if n > 0:
        s_interval[1:] = s_after

    Lam = np.zeros_like(t_grid)
    for k in range(len(s_interval)):
        dt = float(t_grid[k + 1] - t_grid[k])
        e = math.exp(-beta * dt) if dt != 0 else 1.0
        inc = mu * dt + alpha * s_interval[k] * (1.0 - e) / beta
        Lam[k + 1] = Lam[k] + inc

    return FIBPrecomp(
        t_obs=t, T=float(T), mu=mu, alpha=alpha, beta=beta,
        s_after=s_after, t_grid=t_grid, s_interval=s_interval, Lam_grid=Lam
    )


def invert_Lambda(pre: FIBPrecomp, s: float) -> float:
    Lam = pre.Lam_grid
    k = int(np.searchsorted(Lam, s, side="right") - 1)
    if k < 0:
        return 0.0
    if k >= len(pre.s_interval):
        return float(pre.T)

    s0 = float(Lam[k])
    target = float(s - s0)
    t0 = float(pre.t_grid[k])
    dt_max = float(pre.t_grid[k + 1] - pre.t_grid[k])
    s0_int = float(pre.s_interval[k])

    mu, alpha, beta = pre.mu, pre.alpha, pre.beta

    lo, hi = 0.0, dt_max
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        e = math.exp(-beta * mid) if mid != 0 else 1.0
        val = mu * mid + alpha * s0_int * (1.0 - e) / beta
        if val < target:
            lo = mid
        else:
            hi = mid
    return t0 + 0.5 * (lo + hi)


def simulate_FIB(pre: FIBPrecomp, rng: np.random.Generator) -> np.ndarray:
    """
    Generate N*(t) = Q*(Λ_hat(t)) with Q* homogeneous Poisson rate 1 in transformed time.
    Returns bootstrap times t* in [0,T].
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
        return np.zeros(0, float)

    t_star = np.array([invert_Lambda(pre, si) for si in s_list], float)
    return t_star


def fib_bootstrap_LR(t1_obs: np.ndarray, t2_obs: np.ndarray, T: float,
                     theta0: np.ndarray, LR_obs: float, B: int, seed: int = 7):
    pre1 = build_fib_precomp(t1_obs, T, theta0)
    pre2 = build_fib_precomp(t2_obs, T, theta0)

    rng = np.random.default_rng(seed)
    LR = np.empty(B, float)

    if trange is None:
        # fallback progress printing
        def iterator():
            for b in range(B):
                if b % max(1, B // 10) == 0:
                    print(f"Bootstrap {b}/{B} ...")
                yield b
        it = iterator()
    else:
        it = trange(B, desc="FIB bootstrap", unit="rep")

    for b in it:
        t1_star = simulate_FIB(pre1, rng)
        t2_star = simulate_FIB(pre2, rng)

        th1, ll1 = fit_uni(t1_star, T, seed=int(rng.integers(0, 2**31 - 1)))
        th2, ll2 = fit_uni(t2_star, T, seed=int(rng.integers(0, 2**31 - 1)))
        th0, ll0 = fit_uni_pooled(t1_star, t2_star, T)

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
    t_all = merge_univariate(t_buy, t_sell)

    T_end = float(t_all[-1])
    T_half = 0.5 * T_end

    t1, t2 = split_and_shift(t_all, T_half)

    # ---- MLE fits ----
    th1, ll1 = fit_uni(t1, T_half, seed=1)
    th2, ll2 = fit_uni(t2, T_half, seed=2)
    th0, ll0 = fit_uni_pooled(t1, t2, T_half)

    LR_obs = 2.0 * ((ll1 + ll2) - ll0)
    p_theory = chi2.sf(LR_obs, df=3)

    print("\n=== Univariate Hawkes (no buy/sell cross) — MLE ===")
    print("Half 1 theta = [mu, alpha, beta] =", th1, " ll =", ll1)
    print("Half 2 theta = [mu, alpha, beta] =", th2, " ll =", ll2)
    print("H0 pooled theta = [mu, alpha, beta] =", th0, " ll_sum =", ll0)
    print(f"LR_obs = {LR_obs:.6f}   asymptotic p (chi2, df=3) = {p_theory:.3e}")

    # ---- Diagnostics ----
    if not args.skip_diagnostics:
        v1 = rescaled_interarrivals(t1, T_half, th1[0], th1[1], th1[2])
        v2 = rescaled_interarrivals(t2, T_half, th2[0], th2[1], th2[2])

        out1 = save_rescaling_outputs(v1, prefix="half1_uni")
        out2 = save_rescaling_outputs(v2, prefix="half2_uni")

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
    LR_star, p_boot = fib_bootstrap_LR(t1, t2, T_half, th0, LR_obs, B=args.B, seed=args.seed)
    np.savetxt("LR_star_FIB_univariate.csv", LR_star, delimiter=",")

    print("\n=== FIB bootstrap (univariate, full refit) ===")
    print(f"B = {args.B}  seed = {args.seed}")
    print(f"p_boot (FIB) = {p_boot:.6f}")
    print("Saved: LR_star_FIB_univariate.csv")

    plt.figure(figsize=(6, 4))
    plt.hist(LR_star, bins=40, density=True)
    plt.axvline(LR_obs)
    plt.title(f"FIB bootstrap LR* (univariate) — B={args.B}")
    plt.xlabel("LR*")
    plt.ylabel("density")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("LR_star_FIB_univariate_hist.png", dpi=160)
    plt.close()
    print("Saved: LR_star_FIB_univariate_hist.png")


if __name__ == "__main__":
    main()
