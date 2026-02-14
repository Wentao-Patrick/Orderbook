# -*- coding: utf-8 -*-
"""
Sanofi Hawkes change-point style test on two event series (buy/sell).

Input
-----
- sanofi_hawkes_buy_times.csv
- sanofi_hawkes_sell_times.csv

Method
------
1) Build a 2D point process: mark 0=buy, 1=sell (Scheme A).
2) Split the observation window into two halves of equal duration.
3) Fit a 2D Hawkes with exponential kernel (beta fixed) on each half:
     lambda_buy(t)  = mu_buy  + a_self * S_buy(t) + a_cross * S_sell(t)
     lambda_sell(t) = mu_sell + a_cross * S_buy(t) + a_self  * S_sell(t)
   where S_x(t) = sum_{events of type x before t} exp(-beta*(t - t_k)).
4) Likelihood ratio test:
     H0: same parameters on both halves
     H1: different parameters
   LR = 2[(ll1_hat + ll2_hat) - ll0_hat]
   Theoretical p-value: chi2(df=4) survival function.
5) Parametric bootstrap (fixed-parameter):
   simulate two independent halves under H0 (using x0_hat),
   refit x1*, x2*, compute LR* using x0_hat fixed.
   p_boot ≈ (1 + #{LR* >= LR_obs}) / (B + 1).

Notes
-----
- beta is fixed to keep the demo fast/robust. You can tune beta (e.g. 5, 10, 20).
- Increase B (e.g. 500–2000) for a stable bootstrap p-value.

Run
---
python sanofi_hawkes_change_test.py --beta 10 --B 500
"""

from __future__ import annotations

import argparse
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
from tqdm import tqdm


def merge_events(buy_t: np.ndarray, sell_t: np.ndarray) -> np.ndarray:
    t = np.concatenate([buy_t, sell_t])
    m = np.concatenate([np.zeros(len(buy_t), int), np.ones(len(sell_t), int)])
    idx = np.argsort(t)
    return np.column_stack([t[idx], m[idx]])


def split_and_shift(times: np.ndarray, split: float):
    return times[times <= split].copy(), (times[times > split].copy() - split)


def ll_grad_symmetric(x, events, T, beta):
    # x = [mu_buy, mu_sell, a_self, a_cross]
    mu1, mu2, aS, aC = x
    if mu1 <= 0 or mu2 <= 0 or aS < 0 or aC < 0 or beta <= 0:
        return -np.inf, np.zeros_like(x)
    if (aS + aC) >= 0.999 * beta:
        return -np.inf, np.zeros_like(x)

    t = events[:, 0]
    m = events[:, 1].astype(int)
    t_buy = t[m == 0]
    t_sell = t[m == 1]

    Sbuy = float(np.sum(1.0 - np.exp(-beta * (T - t_buy)))) if len(t_buy) else 0.0
    Ssell = float(np.sum(1.0 - np.exp(-beta * (T - t_sell)))) if len(t_sell) else 0.0

    integral = (mu1 + mu2) * T + ((aS + aC) * (Sbuy + Ssell)) / beta

    s1 = s2 = 0.0
    t_prev = 0.0
    ll = 0.0
    g_mu1 = g_mu2 = g_aS = g_aC = 0.0

    for tk, mk in events:
        dt = tk - t_prev
        if dt < 0:
            return -np.inf, np.zeros_like(x)
        decay = math.exp(-beta * dt) if dt != 0 else 1.0
        s1 *= decay
        s2 *= decay

        lam1 = mu1 + aS * s1 + aC * s2
        lam2 = mu2 + aC * s1 + aS * s2
        if lam1 <= 0 or lam2 <= 0 or (not np.isfinite(lam1)) or (not np.isfinite(lam2)):
            return -np.inf, np.zeros_like(x)

        if mk == 0:
            ll += math.log(lam1)
            inv = 1.0 / lam1
            g_mu1 += inv
            g_aS += s1 * inv
            g_aC += s2 * inv
            s1 += 1.0
        else:
            ll += math.log(lam2)
            inv = 1.0 / lam2
            g_mu2 += inv
            g_aS += s2 * inv
            g_aC += s1 * inv
            s2 += 1.0

        t_prev = tk

    ll -= integral
    g_mu1 -= T
    g_mu2 -= T
    g_aS -= (Sbuy + Ssell) / beta
    g_aC -= (Sbuy + Ssell) / beta

    return ll, np.array([g_mu1, g_mu2, g_aS, g_aC], float)


def fit_sym_once(events, T, beta, x_init, maxiter=80):
    bounds = [(1e-8, None), (1e-8, None), (0.0, None), (0.0, None)]

    def fun(x):
        ll, _ = ll_grad_symmetric(x, events, T, beta)
        return 1e100 if not np.isfinite(ll) else -ll

    def jac(x):
        ll, g = ll_grad_symmetric(x, events, T, beta)
        return np.zeros_like(x) if not np.isfinite(ll) else -g

    res = minimize(fun, x_init, jac=jac, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})
    return res.x, -res.fun


def fit_sym_restricted(ev1, ev2, T, beta):
    n_buy = int(np.sum(ev1[:, 1] == 0) + np.sum(ev2[:, 1] == 0))
    n_sell = int(np.sum(ev1[:, 1] == 1) + np.sum(ev2[:, 1] == 1))
    mu1_0 = max(n_buy / (2 * T), 1e-6)
    mu2_0 = max(n_sell / (2 * T), 1e-6)
    aS0 = 0.15 * beta
    aC0 = 0.03 * beta
    x_init = np.array([mu1_0, mu2_0, aS0, aC0], float)

    bounds = [(1e-8, None), (1e-8, None), (0.0, None), (0.0, None)]

    def fun(x):
        ll_a, _ = ll_grad_symmetric(x, ev1, T, beta)
        ll_b, _ = ll_grad_symmetric(x, ev2, T, beta)
        ll = ll_a + ll_b
        return 1e100 if not np.isfinite(ll) else -ll

    def jac(x):
        ll_a, ga = ll_grad_symmetric(x, ev1, T, beta)
        ll_b, gb = ll_grad_symmetric(x, ev2, T, beta)
        ll = ll_a + ll_b
        return np.zeros_like(x) if not np.isfinite(ll) else -(ga + gb)

    res = minimize(fun, x_init, jac=jac, method="L-BFGS-B", bounds=bounds, options={"maxiter": 120})
    return res.x, -res.fun


def simulate_hawkes_sym(x, beta, T, seed=0):
    mu1, mu2, aS, aC = x
    rng = np.random.default_rng(seed)
    s1 = s2 = 0.0
    t = 0.0
    events = []

    lam1 = mu1 + aS * s1 + aC * s2
    lam2 = mu2 + aC * s1 + aS * s2

    while True:
        lam_tot = lam1 + lam2
        if lam_tot <= 0:
            break
        M = lam_tot
        w = rng.exponential(1.0 / M)
        t_c = t + w
        if t_c > T:
            break
        decay = math.exp(-beta * w) if w != 0 else 1.0
        s1 *= decay
        s2 *= decay
        lam1_c = mu1 + aS * s1 + aC * s2
        lam2_c = mu2 + aC * s1 + aS * s2
        lam_tot_c = lam1_c + lam2_c

        if rng.random() * M <= lam_tot_c:
            if rng.random() * lam_tot_c <= lam1_c:
                mark = 0
                s1 += 1.0
            else:
                mark = 1
                s2 += 1.0
            t = t_c
            events.append((t, mark))
            lam1 = mu1 + aS * s1 + aC * s2
            lam2 = mu2 + aC * s1 + aS * s2
        else:
            t = t_c
            lam1, lam2 = lam1_c, lam2_c

    return np.array(events, float) if len(events) else np.zeros((0, 2), float)


def loglik_only(x, events, T, beta):
    ll, _ = ll_grad_symmetric(x, events, T, beta)
    return ll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buy_csv", default=r"C:\Users\Wentao\Desktop\EA_recherche\Hawkes\sanofi_hawkes_buy_times.csv")
    ap.add_argument("--sell_csv", default=r"C:\Users\Wentao\Desktop\EA_recherche\Hawkes\sanofi_hawkes_sell_times.csv")
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    buy = pd.read_csv(args.buy_csv)
    sell = pd.read_csv(args.sell_csv)
    t_buy = np.sort(buy["t_seconds_jitter"].to_numpy(float))
    t_sell = np.sort(sell["t_seconds_jitter"].to_numpy(float))

    T_end = float(max(t_buy[-1], t_sell[-1]))
    T_half = T_end / 2.0

    buy1, buy2 = split_and_shift(t_buy, T_half)
    sell1, sell2 = split_and_shift(t_sell, T_half)

    ev1 = merge_events(buy1, sell1)
    ev2 = merge_events(buy2, sell2)

    beta = args.beta

    # Fit on real data
    x_init = np.array([0.05, 0.05, 0.15 * beta, 0.03 * beta], float)
    x1_hat, ll1_hat = fit_sym_once(ev1, T_half, beta, x_init=x_init, maxiter=120)
    x2_hat, ll2_hat = fit_sym_once(ev2, T_half, beta, x_init=x_init, maxiter=120)
    x0_hat, ll0_sum_hat = fit_sym_restricted(ev1, ev2, T_half, beta)

    LR_obs = 2.0 * ((ll1_hat + ll2_hat) - ll0_sum_hat)
    p_theory = chi2.sf(LR_obs, df=4)

    print("=== Real data estimates (beta fixed) ===")
    print("beta =", beta)
    print("x1_hat (mu_buy, mu_sell, a_self, a_cross) =", x1_hat)
    print("x2_hat (mu_buy, mu_sell, a_self, a_cross) =", x2_hat)
    print("x0_hat (pooled, H0)                     =", x0_hat)
    print("LR_obs =", LR_obs)
    print("p_theory (chi2 df=4) =", p_theory)

    # Bootstrap (fixed-parameter)
    print("\n=== Parametric Bootstrap ===")
    rng = np.random.default_rng(args.seed)
    LR_star = np.empty(args.B, float)
    for b in tqdm(range(args.B), desc="Running bootstrap"):
        s = int(rng.integers(0, 2**31 - 1))
        ev1b = simulate_hawkes_sym(x0_hat, beta, T_half, seed=s)
        ev2b = simulate_hawkes_sym(x0_hat, beta, T_half, seed=s + 1)
        x1b, ll1b = fit_sym_once(ev1b, T_half, beta, x_init=x0_hat, maxiter=80)
        x2b, ll2b = fit_sym_once(ev2b, T_half, beta, x_init=x0_hat, maxiter=80)
        ll0b = loglik_only(x0_hat, ev1b, T_half, beta) + loglik_only(x0_hat, ev2b, T_half, beta)
        LR_star[b] = 2.0 * ((ll1b + ll2b) - ll0b)

    p_boot = (1.0 + np.sum(LR_star >= LR_obs)) / (args.B + 1.0)
    print("p_boot (parametric bootstrap, fixed-parameter) =", p_boot)


if __name__ == "__main__":
    main()
