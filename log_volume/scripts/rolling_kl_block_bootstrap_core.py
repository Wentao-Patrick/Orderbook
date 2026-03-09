from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from log_volume.scripts.time_varying_kde_rolling_kl import (
    EPS,
    first_day_hour_mask,
    init_dynamic_density,
    kl_divergence_continuous,
    make_x_grid,
    normalize_density,
    rolling_dynamic_kl,
    select_h_omega,
    static_kde_density,
)


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


@dataclass
class RollingKlBootstrapResult:
    times: pd.DatetimeIndex
    observed_kl: np.ndarray
    pointwise_threshold: np.ndarray
    pointwise_p_value: np.ndarray
    global_threshold: float
    global_p_value: float
    bootstrap_kl: np.ndarray
    h_star: float
    omega_star: float
    objective: float
    reference_start: pd.Timestamp
    reference_end: pd.Timestamp
    dynamic_start: pd.Timestamp
    reference_points: int
    calib_t0: int
    nu_used: int


def circular_block_bootstrap_indices(
    source_size: int,
    sample_size: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if source_size <= 0:
        raise ValueError("source_size must be positive.")
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative.")
    if sample_size == 0:
        return np.empty(0, dtype=int)

    block = max(1, min(int(block_length), source_size))
    n_blocks = int(np.ceil(sample_size / block))
    starts = rng.integers(0, source_size, size=n_blocks)
    offsets = np.arange(block, dtype=int)
    indices = (starts[:, None] + offsets[None, :]) % source_size
    return indices.reshape(-1)[:sample_size]


def build_kernel_table(grid: np.ndarray, x_source: np.ndarray, h: float) -> np.ndarray:
    u = (grid[None, :] - x_source[:, None]) / h
    kernels = np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u ** 2) / h, 0.0)
    integrals = np.trapezoid(kernels, grid, axis=1)
    valid = integrals > EPS
    kernels[valid] = kernels[valid] / integrals[valid, None]
    if np.any(~valid):
        kernels[~valid] = 1.0 / (grid[-1] - grid[0])
    return kernels


def bootstrap_null_kl_paths(
    init_sample: np.ndarray,
    source_sample: np.ndarray,
    grid: np.ndarray,
    h: float,
    omega: float,
    reference_density: np.ndarray,
    n_total: int,
    t0: int,
    step_points: int,
    n_bootstrap: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    step = max(1, int(step_points))
    n_post = n_total - t0
    if n_post < 0:
        raise ValueError("n_total must be >= t0.")

    kernel_table = build_kernel_table(grid=grid, x_source=source_sample, h=h)
    init_density = init_dynamic_density(grid=grid, x_init=init_sample, h=h, omega=omega)
    reference_density = normalize_density(grid, reference_density)

    n_kl = 1 + int(np.floor(n_post / step))
    kl_paths = np.empty((n_bootstrap, n_kl), dtype=float)
    rng = np.random.default_rng(seed)

    for b in tqdm(range(n_bootstrap), desc="Block bootstrap", leave=False):
        density = init_density.copy()
        kl_paths[b, 0] = kl_divergence_continuous(grid=grid, p=density, q=reference_density)

        boot_idx = circular_block_bootstrap_indices(
            source_size=source_sample.size,
            sample_size=n_post,
            block_length=block_length,
            rng=rng,
        )

        out_pos = 1
        for offset, src_idx in enumerate(boot_idx, start=1):
            density = omega * density + (1.0 - omega) * kernel_table[src_idx]
            if offset % step == 0:
                kl_paths[b, out_pos] = kl_divergence_continuous(
                    grid=grid,
                    p=density,
                    q=reference_density,
                )
                out_pos += 1

    return kl_paths


def compute_bootstrap_thresholds(
    observed_kl: np.ndarray,
    bootstrap_kl: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    upper_q = np.quantile(bootstrap_kl, 1.0 - alpha, axis=0)
    pointwise_p = (1.0 + np.sum(bootstrap_kl >= observed_kl[None, :], axis=0)) / (
        bootstrap_kl.shape[0] + 1.0
    )
    bootstrap_max = np.max(bootstrap_kl, axis=1)
    observed_max = float(np.max(observed_kl))
    global_threshold = float(np.quantile(bootstrap_max, 1.0 - alpha))
    global_p = float((1.0 + np.sum(bootstrap_max >= observed_max)) / (bootstrap_kl.shape[0] + 1.0))
    return upper_q, pointwise_p, global_threshold, global_p


def run_block_bootstrap_analysis(
    times: pd.DatetimeIndex,
    x: np.ndarray,
    nu_used: int,
    calib_t0: int,
    h_grid_size: int,
    omega_grid_size: int,
    grid_size: int,
    grid_quantile_low: float,
    grid_quantile_high: float,
    kl_step_points: int,
    reference_hour_start: int,
    reference_hour_end: int,
    block_length: int,
    n_bootstrap: int,
    alpha: float,
    seed: int,
    h_fixed: float | None = None,
    omega_fixed: float | None = None,
) -> RollingKlBootstrapResult:
    ref_mask = first_day_hour_mask(
        times=times,
        start_hour=reference_hour_start,
        end_hour=reference_hour_end,
    )
    if not np.any(ref_mask):
        raise RuntimeError("No observations found in the requested reference window.")

    ref_indices = np.where(ref_mask)[0]
    x_ref = x[ref_indices]
    ref_times = times[ref_mask]

    t0 = int(ref_indices[-1]) + 1
    if t0 >= x.size:
        raise RuntimeError("No observations after reference window to run dynamic KDE.")

    calib_grid = make_x_grid(
        x=x_ref,
        grid_size=grid_size,
        q_low=grid_quantile_low,
        q_high=grid_quantile_high,
    )
    if h_fixed is None or omega_fixed is None:
        selection = select_h_omega(
            x_calib=x_ref,
            t0=calib_t0,
            nu=nu_used,
            grid=calib_grid,
            h_grid_size=h_grid_size,
            omega_grid_size=omega_grid_size,
            constrained=True,
        )
    else:
        selection = type("FixedSelection", (), {})()
        selection.h_star = float(h_fixed)
        selection.omega_star = float(omega_fixed)
        selection.objective = float("nan")

    full_grid = make_x_grid(
        x=x,
        grid_size=grid_size,
        q_low=grid_quantile_low,
        q_high=grid_quantile_high,
    )
    reference_density = static_kde_density(full_grid, x_ref, selection.h_star)
    reference_time = ref_times[-1]

    kl_times, observed_kl = rolling_dynamic_kl(
        x=x,
        times=times,
        h=selection.h_star,
        omega=selection.omega_star,
        t0=t0,
        grid=full_grid,
        reference_density=reference_density,
        reference_time=reference_time,
        step_points=kl_step_points,
    )

    bootstrap_kl = bootstrap_null_kl_paths(
        init_sample=x[:t0],
        source_sample=x_ref,
        grid=full_grid,
        h=selection.h_star,
        omega=selection.omega_star,
        reference_density=reference_density,
        n_total=x.size,
        t0=t0,
        step_points=kl_step_points,
        n_bootstrap=n_bootstrap,
        block_length=block_length,
        seed=seed,
    )
    pointwise_threshold, pointwise_p, global_threshold, global_p = compute_bootstrap_thresholds(
        observed_kl=observed_kl,
        bootstrap_kl=bootstrap_kl,
        alpha=alpha,
    )

    return RollingKlBootstrapResult(
        times=kl_times,
        observed_kl=observed_kl,
        pointwise_threshold=pointwise_threshold,
        pointwise_p_value=pointwise_p,
        global_threshold=global_threshold,
        global_p_value=global_p,
        bootstrap_kl=bootstrap_kl,
        h_star=selection.h_star,
        omega_star=selection.omega_star,
        objective=selection.objective,
        reference_start=ref_times[0],
        reference_end=ref_times[-1],
        dynamic_start=times[t0],
        reference_points=int(x_ref.size),
        calib_t0=int(calib_t0),
        nu_used=int(nu_used),
    )


def plot_kl_with_thresholds(
    times: pd.DatetimeIndex,
    observed_kl: np.ndarray,
    pointwise_threshold: np.ndarray,
    global_threshold: float,
    output: Path,
    title: str,
) -> None:
    plt.figure(figsize=(11, 4.2))
    plt.plot(times, observed_kl, color="black", linewidth=1.5, label="observed KL")
    plt.plot(times, pointwise_threshold, color="tab:red", linewidth=1.1, linestyle="--", label="pointwise 95% threshold")
    plt.axhline(global_threshold, color="tab:blue", linewidth=1.1, linestyle=":", label="global 95% threshold")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=times.tz))
    ax.grid(alpha=0.2)
    plt.title(title)
    plt.ylabel("KL divergence")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close()


def save_bootstrap_outputs(
    result: RollingKlBootstrapResult,
    output_csv: Path,
    summary_csv: Path,
    extra_summary: dict[str, object],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    threshold_df = pd.DataFrame(
        {
            "time_paris": result.times,
            "kl_observed": result.observed_kl,
            "kl_pointwise_threshold_95": result.pointwise_threshold,
            "pointwise_p_value": result.pointwise_p_value,
            "is_pointwise_significant_95": result.observed_kl > result.pointwise_threshold,
            "kl_global_threshold_95": result.global_threshold,
            "is_global_significant_95": result.observed_kl > result.global_threshold,
        }
    )
    threshold_df.to_csv(output_csv, index=False)

    summary = {
        "reference_start": result.reference_start.isoformat(),
        "reference_end": result.reference_end.isoformat(),
        "dynamic_start": result.dynamic_start.isoformat(),
        "reference_points": result.reference_points,
        "h_star": result.h_star,
        "omega_star": result.omega_star,
        "objective_d_nu": result.objective,
        "calib_t0": result.calib_t0,
        "nu_used": result.nu_used,
        "n_bootstrap": int(result.bootstrap_kl.shape[0]),
        "global_threshold_95": result.global_threshold,
        "observed_max_kl": float(np.max(result.observed_kl)),
        "global_p_value": result.global_p_value,
        "pointwise_exceedance_count_95": int(np.sum(result.observed_kl > result.pointwise_threshold)),
        "global_exceedance_count_95": int(np.sum(result.observed_kl > result.global_threshold)),
        **extra_summary,
    }
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
