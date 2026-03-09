"""
Validate that the CIR process stationary distribution is Gamma.

Model:
    dX_t = kappa * (theta - X_t) dt + sigma * sqrt(X_t) dW_t

For CIR under standard conditions, the invariant law is:
    X_infty ~ Gamma(shape=alpha, scale=beta)
where:
    alpha = 2 * kappa * theta / sigma^2
    beta  = sigma^2 / (2 * kappa)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma, kstest, ncx2


@dataclass
class CIRParams:
    kappa: float
    theta: float
    sigma: float
    dt: float


def stationary_gamma_params(params: CIRParams) -> tuple[float, float]:
    alpha = 2.0 * params.kappa * params.theta / (params.sigma ** 2)
    beta = (params.sigma ** 2) / (2.0 * params.kappa)
    return alpha, beta


def simulate_cir_exact(
    params: CIRParams,
    n_steps: int,
    x0: float,
    seed: int,
) -> np.ndarray:
    """
    Exact one-step transition:
        X_{t+dt} = c * Y, Y ~ noncentral-chi-square(df=d, nc=lambda)
    """
    if n_steps <= 1:
        raise ValueError("n_steps must be >= 2")
    if params.kappa <= 0 or params.theta <= 0 or params.sigma <= 0 or params.dt <= 0:
        raise ValueError("kappa/theta/sigma/dt must all be > 0")
    if x0 < 0:
        raise ValueError("x0 must be >= 0")

    rng = np.random.default_rng(seed)
    exp_kdt = np.exp(-params.kappa * params.dt)
    c = (params.sigma ** 2) * (1.0 - exp_kdt) / (4.0 * params.kappa)
    d = 4.0 * params.kappa * params.theta / (params.sigma ** 2)

    path = np.empty(n_steps, dtype=float)
    path[0] = x0
    for t in range(1, n_steps):
        lam = (exp_kdt / c) * path[t - 1]
        path[t] = c * ncx2.rvs(df=d, nc=lam, random_state=rng)

    return path


def summarize_fit(samples: np.ndarray, params: CIRParams, ks_thin: int) -> dict[str, float]:
    alpha, beta = stationary_gamma_params(params)
    mean_theory = params.theta
    var_theory = params.theta * (params.sigma ** 2) / (2.0 * params.kappa)
    mean_emp = float(np.mean(samples))
    var_emp = float(np.var(samples))
    ks_stat_full, ks_pvalue_full = kstest(samples, "gamma", args=(alpha, 0.0, beta))

    thin = max(1, int(ks_thin))
    ks_samples = samples[::thin]
    ks_stat, ks_pvalue = kstest(ks_samples, "gamma", args=(alpha, 0.0, beta))

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "mean_theory": float(mean_theory),
        "var_theory": float(var_theory),
        "mean_emp": mean_emp,
        "var_emp": var_emp,
        "ks_sample_size": int(ks_samples.size),
        "ks_thin": int(thin),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "ks_stat_full": float(ks_stat_full),
        "ks_pvalue_full": float(ks_pvalue_full),
    }


def make_plot(samples: np.ndarray, params: CIRParams, output: Path, bins: int) -> None:
    alpha, beta = stationary_gamma_params(params)

    q_hi = np.quantile(samples, 0.999)
    x_max = max(float(q_hi), float(np.max(samples)))
    x = np.linspace(0.0, x_max, 500)
    pdf = gamma.pdf(x, a=alpha, loc=0.0, scale=beta)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=bins, density=True, alpha=0.55, color="#4C78A8", label="Empirical stationary sample")
    plt.plot(x, pdf, color="#E45756", linewidth=2.2, label="Theoretical Gamma density")
    plt.title("CIR Stationary Distribution vs Theoretical Gamma")
    plt.xlabel("X")
    plt.ylabel("Density")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Validate CIR stationary Gamma distribution.")
    parser.add_argument("--kappa", type=float, default=3.0)
    parser.add_argument("--theta", type=float, default=1.5)
    parser.add_argument("--sigma", type=float, default=1.2)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--n-steps", type=int, default=220_000)
    parser.add_argument("--burn-in", type=int, default=20_000)
    parser.add_argument("--x0", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument(
        "--ks-thin",
        type=int,
        default=20,
        help="Use every k-th sample for KS test to reduce serial dependence impact (k>=1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "results" / "cir_stationary_gamma_validation.png",
        help="Output figure path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.burn_in >= args.n_steps:
        raise ValueError("--burn-in must be smaller than --n-steps")

    params = CIRParams(
        kappa=args.kappa,
        theta=args.theta,
        sigma=args.sigma,
        dt=args.dt,
    )

    feller_gap = 2.0 * params.kappa * params.theta - (params.sigma ** 2)
    if feller_gap <= 0:
        print("Warning: Feller condition not satisfied (2*kappa*theta <= sigma^2).")
        print("The process may hit zero; stationary Gamma comparison can still be checked numerically.")

    path = simulate_cir_exact(params, n_steps=args.n_steps, x0=args.x0, seed=args.seed)
    samples = path[args.burn_in :]

    stats = summarize_fit(samples, params, ks_thin=args.ks_thin)
    make_plot(samples, params, output=args.output, bins=args.bins)

    print("CIR stationary Gamma validation")
    print("=" * 42)
    print(f"Parameters: kappa={params.kappa}, theta={params.theta}, sigma={params.sigma}, dt={params.dt}")
    print(f"Samples used: {samples.size:,} (burn-in={args.burn_in:,})")
    print(f"Theoretical Gamma: shape(alpha)={stats['alpha']:.6f}, scale(beta)={stats['beta']:.6f}")
    print(f"Mean: empirical={stats['mean_emp']:.6f}, theory={stats['mean_theory']:.6f}")
    print(f"Variance: empirical={stats['var_emp']:.6f}, theory={stats['var_theory']:.6f}")
    print(
        f"KS test (thinned, every {stats['ks_thin']} steps, n={stats['ks_sample_size']:,}): "
        f"statistic={stats['ks_stat']:.6f}, p-value={stats['ks_pvalue']:.6g}"
    )
    print(
        "KS test (full dependent sample, reference only): "
        f"statistic={stats['ks_stat_full']:.6f}, p-value={stats['ks_pvalue_full']:.6g}"
    )
    if stats["ks_pvalue"] >= 0.05:
        print("Conclusion: fail to reject Gamma stationary distribution at alpha=0.05.")
    else:
        print("Conclusion: reject Gamma stationary distribution at alpha=0.05.")
    print(f"Figure saved to: {args.output}")


if __name__ == "__main__":
    main()
