# -*- coding: utf-8 -*-
"""
Fit a univariate power-law Hawkes model on thresholded RLOP exceedance events.

Workflow
--------
1) Read thresholded exceedance events exported from rlop_events.csv.
2) Fit a univariate Hawkes process with power-law kernel
       lambda(t) = mu + A * sum_j (t - t_j + c)^(-p),
   parameterized by (mu, eta, c, p), where eta is the branching ratio and
       A = eta * (p - 1) * c^(p - 1),   0 < eta < 1, p > 1, c > 0.
3) Split the sample into two halves, fit half-specific models and a pooled H0
   model, and compute a likelihood-ratio statistic.
4) Evaluate the fitted model using time-rescaling diagnostics.
5) Simulate one path from the full-sample fit and compare cumulative counts.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, expon, kstest


SCRIPT_DIR = Path(__file__).resolve().parent
RLOP_POWERLAW_DIR = SCRIPT_DIR.parent
DATA_DERIVED_DIR = RLOP_POWERLAW_DIR / "data" / "derived"
RESULTS_DIR = RLOP_POWERLAW_DIR / "results"
FIGURES_DIR = RLOP_POWERLAW_DIR / "figures"


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def threshold_tag_from_input(path: Path) -> str:
    return path.stem


@dataclass
class FitContext:
    t: np.ndarray
    T: float
    lags: np.ndarray
    mask: np.ndarray
    tails: np.ndarray


def make_context(t: np.ndarray, T: float) -> FitContext:
    t = np.asarray(t, dtype=float)
    lags = t[:, None] - t[None, :]
    mask = lags > 0.0
    tails = T - t
    return FitContext(t=t, T=float(T), lags=lags, mask=mask, tails=tails)


def unpack_params(u: np.ndarray) -> tuple[float, float, float, float, float]:
    mu = math.exp(float(u[0]))
    eta = sigmoid(float(u[1]))
    c = math.exp(float(u[2]))
    p = 1.0 + math.exp(float(u[3]))
    amp = eta * (p - 1.0) * (c ** (p - 1.0))
    return mu, eta, c, p, amp


def powerlaw_loglik(u: np.ndarray, ctx: FitContext) -> float:
    mu, eta, c, p, amp = unpack_params(u)
    if mu <= 0.0 or not (0.0 < eta < 1.0) or c <= 0.0 or p <= 1.0:
        return -np.inf

    kernel_terms = np.zeros_like(ctx.lags)
    np.power(ctx.lags + c, -p, out=kernel_terms, where=ctx.mask)
    excitation = amp * kernel_terms.sum(axis=1)
    intensity = mu + excitation
    if np.any(intensity <= 0.0) or not np.all(np.isfinite(intensity)):
        return -np.inf

    integral = mu * ctx.T + eta * np.sum(1.0 - np.power(1.0 + ctx.tails / c, 1.0 - p))
    ll = float(np.sum(np.log(intensity)) - integral)
    return ll if np.isfinite(ll) else -np.inf


def fit_powerlaw_hawkes(t: np.ndarray, T: float, seed: int = 0, restarts: int = 6) -> tuple[dict[str, float], float]:
    t = np.asarray(t, dtype=float)
    if len(t) == 0:
        raise ValueError("No events provided for fitting.")

    ctx = make_context(t, T)
    rng = np.random.default_rng(seed)
    n = len(t)
    event_rate = max(n / T, 1e-8)
    gaps = np.diff(t)
    positive_gaps = gaps[gaps > 0]
    c0 = float(np.median(positive_gaps)) if positive_gaps.size else max(T / max(10, n), 1e-3)
    c0 = max(c0, 1e-6)

    mu0 = max(event_rate * 0.7, 1e-6)
    eta0 = 0.25
    p0 = 1.5
    x0 = np.array(
        [
            math.log(mu0),
            math.log(eta0 / (1.0 - eta0)),
            math.log(c0),
            math.log(p0 - 1.0),
        ],
        dtype=float,
    )

    max_mu = max(event_rate * 100.0, 10.0)
    bounds = [
        (math.log(1e-8), math.log(max_mu)),
        (-8.0, 8.0),
        (math.log(1e-6), math.log(max(T, 1.0))),
        (math.log(1e-3), math.log(20.0)),
    ]

    def objective(x: np.ndarray) -> float:
        ll = powerlaw_loglik(x, ctx)
        return 1e100 if not np.isfinite(ll) else -ll

    best_res = None
    for k in range(restarts):
        x_init = x0.copy()
        if k > 0:
            x_init += rng.normal(0.0, [0.5, 1.0, 0.8, 0.6], size=4)
        res = minimize(
            objective,
            x_init,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 400},
        )
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    mu, eta, c, p, amp = unpack_params(best_res.x)
    params = {
        "mu": float(mu),
        "eta": float(eta),
        "c": float(c),
        "p": float(p),
        "amplitude": float(amp),
    }
    return params, -float(best_res.fun)


def fit_powerlaw_pooled(t1: np.ndarray, t2: np.ndarray, T: float, seed: int = 0) -> tuple[dict[str, float], float]:
    ctx1 = make_context(t1, T)
    ctx2 = make_context(t2, T)
    rng = np.random.default_rng(seed)

    n = len(t1) + len(t2)
    event_rate = max(n / (2.0 * T), 1e-8)
    gaps = np.diff(np.sort(np.concatenate([t1, t2])))
    positive_gaps = gaps[gaps > 0]
    c0 = float(np.median(positive_gaps)) if positive_gaps.size else max(T / max(10, n), 1e-3)
    c0 = max(c0, 1e-6)

    mu0 = max(event_rate * 0.7, 1e-6)
    eta0 = 0.25
    p0 = 1.5
    x0 = np.array(
        [
            math.log(mu0),
            math.log(eta0 / (1.0 - eta0)),
            math.log(c0),
            math.log(p0 - 1.0),
        ],
        dtype=float,
    )

    max_mu = max(event_rate * 100.0, 10.0)
    bounds = [
        (math.log(1e-8), math.log(max_mu)),
        (-8.0, 8.0),
        (math.log(1e-6), math.log(max(T, 1.0))),
        (math.log(1e-3), math.log(20.0)),
    ]

    def objective(x: np.ndarray) -> float:
        ll1 = powerlaw_loglik(x, ctx1)
        ll2 = powerlaw_loglik(x, ctx2)
        ll = ll1 + ll2
        return 1e100 if not np.isfinite(ll) else -ll

    best_res = None
    for k in range(6):
        x_init = x0.copy()
        if k > 0:
            x_init += rng.normal(0.0, [0.5, 1.0, 0.8, 0.6], size=4)
        res = minimize(
            objective,
            x_init,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500},
        )
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    mu, eta, c, p, amp = unpack_params(best_res.x)
    params = {
        "mu": float(mu),
        "eta": float(eta),
        "c": float(c),
        "p": float(p),
        "amplitude": float(amp),
    }
    return params, -float(best_res.fun)


def split_and_shift(times: np.ndarray, split: float) -> tuple[np.ndarray, np.ndarray]:
    t1 = times[times <= split].copy()
    t2 = times[times > split].copy() - split
    return t1, t2


def rescaled_interarrivals(t: np.ndarray, params: dict[str, float]) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    mu = params["mu"]
    c = params["c"]
    p = params["p"]
    amp = params["amplitude"]

    v = np.empty(len(t), dtype=float)
    start = 0.0
    for i, ti in enumerate(t):
        dt = ti - start
        base = mu * dt
        if i == 0:
            exc = 0.0
        else:
            past = t[:i]
            left = np.power(start - past + c, 1.0 - p)
            right = np.power(ti - past + c, 1.0 - p)
            exc = amp * np.sum(left - right) / (p - 1.0)
        v[i] = base + exc
        start = ti
    return v


def save_rescaling_outputs(v: np.ndarray, prefix: str, results_dir: Path, figures_dir: Path) -> dict[str, object]:
    v = np.asarray(v, dtype=float)
    u = 1.0 - np.exp(-v)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    out_csv = results_dir / f"rescaled_{prefix}.csv"
    pd.DataFrame({"v_exp1": v, "u_unif01": u}).to_csv(out_csv, index=False)

    fig_hist = figures_dir / f"{prefix}_hist_exp.png"
    plt.figure(figsize=(6, 4))
    plt.hist(v, bins=40, density=True)
    x = np.linspace(0.0, max(8.0, float(np.percentile(v, 99))), 400)
    plt.plot(x, expon.pdf(x, scale=1.0))
    plt.title(f"Rescaled inter-arrivals v_i vs Exp(1)\n{prefix}")
    plt.xlabel("v_i")
    plt.ylabel("density")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_hist, dpi=160)
    plt.close()

    fig_qq_exp = figures_dir / f"{prefix}_qq_exp.png"
    plt.figure(figsize=(5, 5))
    v_sorted = np.sort(v)
    n = len(v_sorted)
    p_grid = (np.arange(1, n + 1) - 0.5) / n
    q = expon.ppf(p_grid, scale=1.0)
    plt.plot(q, v_sorted, marker=".", linestyle="None")
    mx = max(float(q[-1]), float(v_sorted[-1]))
    plt.plot([0.0, mx], [0.0, mx])
    plt.title(f"QQ plot: v_i vs Exp(1)\n{prefix}")
    plt.xlabel("Exp(1) quantiles")
    plt.ylabel("empirical quantiles")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_qq_exp, dpi=160)
    plt.close()

    fig_qq_unif = figures_dir / f"{prefix}_qq_unif.png"
    plt.figure(figsize=(5, 5))
    u_sorted = np.sort(u)
    plt.plot(p_grid, u_sorted, marker=".", linestyle="None")
    plt.plot([0.0, 1.0], [0.0, 1.0])
    plt.title(f"QQ plot: 1-exp(-v_i) vs U(0,1)\n{prefix}")
    plt.xlabel("U(0,1) quantiles")
    plt.ylabel("empirical quantiles")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_qq_unif, dpi=160)
    plt.close()

    ks_exp = kstest(v, "expon", args=(0.0, 1.0))
    ks_unif = kstest(u, "uniform", args=(0.0, 1.0))
    return {
        "rescaled_csv": str(out_csv),
        "hist_exp": str(fig_hist),
        "qq_exp": str(fig_qq_exp),
        "qq_unif": str(fig_qq_unif),
        "n": int(len(v)),
        "mean_v": float(np.mean(v)),
        "median_v": float(np.median(v)),
        "ks_exp_stat": float(ks_exp.statistic),
        "ks_exp_pvalue": float(ks_exp.pvalue),
        "ks_unif_stat": float(ks_unif.statistic),
        "ks_unif_pvalue": float(ks_unif.pvalue),
    }


def intensity_at_time(t_now: float, history: np.ndarray, params: dict[str, float]) -> float:
    mu = params["mu"]
    if len(history) == 0:
        return mu
    amp = params["amplitude"]
    c = params["c"]
    p = params["p"]
    return float(mu + amp * np.sum(np.power(t_now - history + c, -p)))


def simulate_powerlaw_hawkes(params: dict[str, float], T: float, seed: int = 0, max_events: int = 200000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    events: list[float] = []
    t = 0.0

    while t < T and len(events) < max_events:
        history = np.asarray(events, dtype=float)
        M = intensity_at_time(t, history, params)
        if M <= 0.0 or not np.isfinite(M):
            break
        w = float(rng.exponential(1.0 / M))
        t_candidate = t + w
        if t_candidate > T:
            break

        lam_cand = intensity_at_time(t_candidate, history, params)
        if rng.random() * M <= lam_cand:
            events.append(t_candidate)
        t = t_candidate

    return np.asarray(events, dtype=float)


def plot_cumulative_counts(
    obs_t: np.ndarray,
    sim_t: np.ndarray,
    T: float,
    output: Path,
    title: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, T, 500)
    obs_counts = np.searchsorted(obs_t, grid, side="right")
    sim_counts = np.searchsorted(sim_t, grid, side="right")

    plt.figure(figsize=(11, 4.2))
    plt.plot(grid / 3600.0, obs_counts, color="black", linewidth=1.6, label="observed")
    plt.plot(grid / 3600.0, sim_counts, color="#d62728", linewidth=1.3, label="simulated")
    plt.xlabel("Hours since observation start")
    plt.ylabel("Cumulative event count")
    plt.title(title)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def params_with_prefix(prefix: str, params: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{k}": float(v) for k, v in params.items()}


def infer_paths(events_csv: Path) -> tuple[Path, Path, Path, Path]:
    tag = threshold_tag_from_input(events_csv)
    results_csv = RESULTS_DIR / f"powerlaw_fit_summary_{tag}.csv"
    diag_csv = RESULTS_DIR / f"powerlaw_rescaling_summary_{tag}.csv"
    sim_csv = RESULTS_DIR / f"powerlaw_simulated_events_{tag}.csv"
    sim_fig = FIGURES_DIR / f"powerlaw_cumulative_counts_{tag}.png"
    return results_csv, diag_csv, sim_csv, sim_fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--events_csv",
        default=str(DATA_DERIVED_DIR / "rlop_bid_events_q99.csv"),
        help="Thresholded RLOP event CSV produced by export_rlop_threshold_events.py",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip_diagnostics", action="store_true")
    ap.add_argument("--results_csv", default=None)
    ap.add_argument("--diag_csv", default=None)
    ap.add_argument("--sim_csv", default=None)
    ap.add_argument("--sim_fig", default=None)
    args = ap.parse_args()

    events_csv = Path(args.events_csv)
    results_csv_default, diag_csv_default, sim_csv_default, sim_fig_default = infer_paths(events_csv)
    results_csv = Path(args.results_csv) if args.results_csv else results_csv_default
    diag_csv = Path(args.diag_csv) if args.diag_csv else diag_csv_default
    sim_csv = Path(args.sim_csv) if args.sim_csv else sim_csv_default
    sim_fig = Path(args.sim_fig) if args.sim_fig else sim_fig_default

    df = pd.read_csv(events_csv, parse_dates=["time_paris"])
    if "t_seconds_jitter" not in df.columns:
        raise ValueError("Missing t_seconds_jitter in events_csv.")
    if df.empty:
        raise ValueError("No events found in events_csv.")

    t = np.sort(df["t_seconds_jitter"].to_numpy(dtype=float))
    if "T_observation_seconds" in df.columns:
        T = float(df["T_observation_seconds"].iloc[0])
    else:
        T = float(t[-1])
    if T <= 0.0:
        raise ValueError("Invalid observation horizon T.")

    side = str(df["side_label"].iloc[0]) if "side_label" in df.columns else "unknown"
    threshold_mode = str(df["threshold_mode"].iloc[0]) if "threshold_mode" in df.columns else "unknown"
    threshold_input = float(df["threshold_value_input"].iloc[0]) if "threshold_value_input" in df.columns else float("nan")
    threshold_actual = float(df["threshold_actual"].iloc[0]) if "threshold_actual" in df.columns else float("nan")

    T_half = 0.5 * T
    t1, t2 = split_and_shift(t, T_half)

    full_params, full_ll = fit_powerlaw_hawkes(t, T, seed=args.seed)
    half1_params, ll1 = fit_powerlaw_hawkes(t1, T_half, seed=args.seed + 1)
    half2_params, ll2 = fit_powerlaw_hawkes(t2, T_half, seed=args.seed + 2)
    pooled_params, ll0 = fit_powerlaw_pooled(t1, t2, T_half, seed=args.seed + 3)

    LR_obs = 2.0 * ((ll1 + ll2) - ll0)
    p_theory = float(chi2.sf(LR_obs, df=4))

    sim_t = simulate_powerlaw_hawkes(full_params, T, seed=args.seed + 1000)
    plot_cumulative_counts(
        obs_t=t,
        sim_t=sim_t,
        T=T,
        output=sim_fig,
        title=f"RLOP power-law Hawkes cumulative counts ({side}, {threshold_tag_from_input(events_csv)})",
    )

    sim_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"t_seconds_simulated": sim_t}).to_csv(sim_csv, index=False)

    results_summary = {
        "events_csv": str(events_csv),
        "side": side,
        "threshold_mode": threshold_mode,
        "threshold_value_input": threshold_input,
        "threshold_actual": threshold_actual,
        "n_events_total": int(len(t)),
        "n_events_half1": int(len(t1)),
        "n_events_half2": int(len(t2)),
        "T_observation_seconds": float(T),
        "T_half_seconds": float(T_half),
        "full_loglik": float(full_ll),
        "half1_loglik": float(ll1),
        "half2_loglik": float(ll2),
        "pooled_loglik": float(ll0),
        "LR_obs": float(LR_obs),
        "LR_pvalue_chi2_df4": p_theory,
        "simulated_event_count": int(len(sim_t)),
        "simulated_mean_interarrival": float(np.mean(np.diff(sim_t))) if len(sim_t) > 1 else float("nan"),
        "observed_mean_interarrival": float(np.mean(np.diff(t))) if len(t) > 1 else float("nan"),
        "simulated_events_csv": str(sim_csv),
        "simulated_counts_figure": str(sim_fig),
    }
    results_summary.update(params_with_prefix("full", full_params))
    results_summary.update(params_with_prefix("half1", half1_params))
    results_summary.update(params_with_prefix("half2", half2_params))
    results_summary.update(params_with_prefix("pooled", pooled_params))

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results_summary]).to_csv(results_csv, index=False)

    print("\n=== Power-law Hawkes fit on thresholded RLOP events ===")
    print("events_csv =", events_csv)
    print("side =", side, " threshold_actual =", threshold_actual, " n_events =", len(t))
    print("full_params =", full_params, " ll =", full_ll)
    print("half1_params =", half1_params, " ll =", ll1)
    print("half2_params =", half2_params, " ll =", ll2)
    print("pooled_params =", pooled_params, " ll =", ll0)
    print(f"LR_obs = {LR_obs:.6f}   asymptotic p (chi2, df=4) = {p_theory:.3e}")
    print("saved:", results_csv)
    print("saved:", sim_csv)
    print("saved:", sim_fig)

    if not args.skip_diagnostics:
        diag1 = save_rescaling_outputs(
            rescaled_interarrivals(t1, half1_params),
            prefix=f"half1_{threshold_tag_from_input(events_csv)}",
            results_dir=RESULTS_DIR,
            figures_dir=FIGURES_DIR,
        )
        diag2 = save_rescaling_outputs(
            rescaled_interarrivals(t2, half2_params),
            prefix=f"half2_{threshold_tag_from_input(events_csv)}",
            results_dir=RESULTS_DIR,
            figures_dir=FIGURES_DIR,
        )
        diag_summary = {
            "events_csv": str(events_csv),
            "side": side,
            "threshold_actual": threshold_actual,
        }
        diag_summary.update({f"half1_{k}": v for k, v in diag1.items() if not isinstance(v, str) or k.endswith((".csv", ".png"))})
        diag_summary.update({f"half2_{k}": v for k, v in diag2.items() if not isinstance(v, str) or k.endswith((".csv", ".png"))})
        diag_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([diag_summary]).to_csv(diag_csv, index=False)
        print("saved:", diag_csv)


if __name__ == "__main__":
    main()
