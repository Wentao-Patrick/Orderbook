"""
Time-varying KDE + rolling KL on log-volume level 1 (paper-style calibration).

This script implements the dynamic kernel density estimator described in:
Garcin, Klein, Laaribi (2023), "Estimation of time-varying kernel densities..."

Main implemented ingredients
----------------------------
1) Dynamic KDE recursion (Eq. (4) in the paper) with Epanechnikov kernel:
       f_{t+1}(x) = omega * f_t(x) + (1-omega) * K_h(x - X_{t+1}).

2) PIT series for parameter selection:
       Z_t^{h,omega} = F_{t-1}^{h,omega}(X_t),
   where F is the cdf associated with the dynamic KDE.

3) Joint selection of bandwidth h and discount omega by minimizing the paper's
   d_nu criterion (Eq. (6)) built from Kolmogorov-style discrepancies on PITs,
   with constrained omega bounds as in Eq. (7):
       1 - 1/nu < omega < 1.

4) Once (h*, omega*) are selected, compute rolling KL divergences between the
   current dynamic density and a reference dynamic density, and plot over time.

Notes
-----
- The kernel is Epanechnikov: K(u) = (3/4)(1 - u^2) for |u| <= 1.
- CDF is obtained by direct recursion using the kernel's primitive
  (antiderivative), as described in the paper, not by numerical integration.
- For speed and numerical stability on long 1-second series, densities are
  represented on a fixed x-grid.
- The d_nu optimization is done by grid-search over (h, omega), which is a
  practical approximation to the continuous argmin in the paper.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import imageio.v2 as imageio
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir.parent / "sanofi_book_snapshots_1s.parquet"
    default_output = script_dir / "rolling_kl_time_varying_kde.png"
    default_video_output = script_dir / "log_volume_1s_video.mp4"
    default_four_pdf_output = script_dir / "pdf_10_12_14_16.png"

    parser = argparse.ArgumentParser(description="Time-varying KDE rolling KL on log volume1.")
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to parquet snapshots file (must contain bidvolume1/askvolume1 and a datetime column).",
    )
    parser.add_argument(
        "--side",
        type=str,
        choices=["bid", "ask"],
        default="bid",
        help="Book side used to build log_{side}volume1.",
    )
    parser.add_argument(
        "--tz",
        type=str,
        default="Europe/Paris",
        help="Timezone for plotting.",
    )
    parser.add_argument(
        "--nu",
        type=int,
        default=60,
        help="Max lag used in d_nu criterion; also defines constrained lower bound for omega.",
    )
    parser.add_argument(
        "--calib-init-frac",
        type=float,
        default=0.33,
        help="Fraction of 9:00-10:00 data used for initialization in calibration (rest for PIT evaluation).",
    )
    parser.add_argument("--h-grid-size", type=int, default=12)
    parser.add_argument("--omega-grid-size", type=int, default=12)
    parser.add_argument(
        "--grid-size",
        type=int,
        default=256,
        help="Number of grid points used for density/cdf representation.",
    )
    parser.add_argument(
        "--grid-quantile-low",
        type=float,
        default=0.002,
        help="Lower quantile for x-grid support.",
    )
    parser.add_argument(
        "--grid-quantile-high",
        type=float,
        default=0.998,
        help="Upper quantile for x-grid support.",
    )
    parser.add_argument(
        "--kl-step-points",
        type=int,
        default=60,
        help="Sampling step (in observations) for KL curve points.",
    )
    parser.add_argument(
        "--reference-index",
        type=int,
        default=-1,
        help="Reference time index for KL. If -1, uses t0-1 (end of initialization).",
    )
    parser.add_argument(
        "--reference-hour-start",
        type=int,
        default=9,
        help="Start hour (inclusive) for reference window on first available day.",
    )
    parser.add_argument(
        "--reference-hour-end",
        type=int,
        default=10,
        help="End hour (exclusive) for reference window on first available day.",
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        default=default_video_output,
        help="MP4 output path for per-second dynamic PDF video.",
    )
    parser.add_argument(
        "--video-max-frames",
        type=int,
        default=0,
        help="Maximum number of frames in video (0 = full 09:00-17:30 session).",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=30,
        help="Frames per second for MP4 video.",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="If set, do not generate MP4 video.",
    )
    parser.add_argument(
        "--four-pdf-output",
        type=Path,
        default=default_four_pdf_output,
        help="Output path for the 10:00/12:00/14:00/16:00 PDF comparison figure.",
    )
    parser.add_argument(
        "--pdf-hours",
        type=int,
        nargs=4,
        default=[10, 12, 14, 16],
        help="Four hours (same day) used for PDF overlay figure, e.g. 10 12 14 16.",
    )
    parser.add_argument("--output", type=Path, default=default_output)
    return parser.parse_args()


def _find_datetime_col(df: pd.DataFrame) -> str | None:
    candidates = [
        "datetime",
        "time",
        "timestamp",
        "DateTime",
        "date",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _to_tz_index(ts: pd.Series, tz: str) -> pd.DatetimeIndex:
    dt = pd.to_datetime(ts, errors="coerce")
    if dt.isna().all():
        raise ValueError("Datetime column could not be parsed.")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(tz)
    else:
        dt = dt.dt.tz_convert(tz)
    return pd.DatetimeIndex(dt)


def load_log_volume_series(path: Path, side: str, tz: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_parquet(path)
    dt_col = _find_datetime_col(df)
    value_col = f"{side}volume1"
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in input file.")

    if dt_col is not None:
        idx = _to_tz_index(df[dt_col], tz=tz)
    elif isinstance(df.index, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is None:
            idx = idx.tz_localize(tz)
        else:
            idx = idx.tz_convert(tz)
    else:
        idx = _to_tz_index(pd.Series(df.index), tz=tz)
    vals = pd.to_numeric(df[value_col], errors="coerce").to_numpy()
    mask = np.isfinite(vals) & (vals > 0) & ~pd.isna(idx)
    idx = idx[mask]
    x = np.log(vals[mask])

    order = np.argsort(idx.values)
    idx = idx[order]
    x = x[order]

    if x.size < 200:
        raise ValueError("Not enough valid observations after filtering.")
    return idx, x


def make_x_grid(x: np.ndarray, grid_size: int, q_low: float, q_high: float) -> np.ndarray:
    lo = float(np.quantile(x, q_low))
    hi = float(np.quantile(x, q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(x))
        hi = float(np.max(x))
    if hi <= lo:
        hi = lo + 1.0
    pad = 0.15 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, grid_size)


def normalize_density(grid: np.ndarray, density: np.ndarray) -> np.ndarray:
    z = np.trapezoid(density, grid)
    if z <= EPS:
        return np.ones_like(density) / (grid[-1] - grid[0])
    return density / z


def epanechnikov_kernel_pdf(grid: np.ndarray, x0: float, h: float) -> np.ndarray:
    u = (grid - x0) / h
    return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u ** 2) / h, 0.0)


def epanechnikov_kernel_cdf(grid: np.ndarray, x0: float, h: float) -> np.ndarray:
    """Primitive (antiderivative) of the Epanechnikov kernel.

    K(u) = (3/4)(1 - u^2)  for |u| <= 1
    => CDF:  K_cdf(u) = 1/2 + 3/4 u - 1/4 u^3   for |u| <= 1
                       = 0                        for u < -1
                       = 1                        for u > 1
    """
    u = (grid - x0) / h
    cdf_vals = np.where(
        u < -1.0,
        0.0,
        np.where(
            u > 1.0,
            1.0,
            0.5 + 0.75 * u - 0.25 * u ** 3,
        ),
    )
    return cdf_vals


def init_dynamic_density(grid: np.ndarray, x_init: np.ndarray, h: float, omega: float) -> np.ndarray:
    t0 = x_init.size
    powers = np.arange(t0 - 1, -1, -1)
    weights = (1.0 - omega) * (omega ** powers)
    denom = 1.0 - (omega ** t0)
    if denom > EPS:
        weights = weights / denom
    u = (grid[None, :] - x_init[:, None]) / h
    kernels = np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u ** 2) / h, 0.0)
    density = np.sum(weights[:, None] * kernels, axis=0)
    return normalize_density(grid, density)


def init_dynamic_cdf(grid: np.ndarray, x_init: np.ndarray, h: float, omega: float) -> np.ndarray:
    """Initialize the dynamic CDF on the grid using the kernel primitive.

    F_t0(x) = sum_i  w_i * K_cdf((x - X_i) / h)
    with the same exponential weights as init_dynamic_density.
    """
    t0 = x_init.size
    powers = np.arange(t0 - 1, -1, -1)
    weights = (1.0 - omega) * (omega ** powers)
    denom = 1.0 - (omega ** t0)
    if denom > EPS:
        weights = weights / denom
    u = (grid[None, :] - x_init[:, None]) / h
    kernel_cdfs = np.where(
        u < -1.0,
        0.0,
        np.where(u > 1.0, 1.0, 0.5 + 0.75 * u - 0.25 * u ** 3),
    )
    cdf = np.sum(weights[:, None] * kernel_cdfs, axis=0)
    return np.clip(cdf, 0.0, 1.0)


def ks_uniform(z: np.ndarray) -> float:
    z = np.clip(np.asarray(z), 0.0, 1.0)
    n = z.size
    if n == 0:
        return np.inf
    zs = np.sort(z)
    i = np.arange(1, n + 1)
    d_plus = np.max(i / n - zs)
    d_minus = np.max(zs - (i - 1) / n)
    return float(max(d_plus, d_minus))


def ks_bivariate_lag(z: np.ndarray, tau: int, chunk: int = 128) -> float:
    if tau <= 0:
        raise ValueError("tau must be > 0")
    if tau >= z.size:
        return np.inf

    a = np.clip(z[:-tau], 0.0, 1.0)
    b = np.clip(z[tau:], 0.0, 1.0)
    n = a.size
    if n == 0:
        return np.inf

    anchor_idx = np.arange(n)

    best = 0.0
    for j0 in range(0, anchor_idx.size, chunk):
        ids = anchor_idx[j0 : j0 + chunk]
        aa = a[ids][None, :]
        bb = b[ids][None, :]
        empirical = ((a[:, None] <= aa) & (b[:, None] <= bb)).mean(axis=0)
        theoretical = a[ids] * b[ids]
        d = np.max(np.abs(theoretical - empirical))
        if d > best:
            best = float(d)
    return best


def d_nu_criterion(z: np.ndarray, nu: int) -> float:
    n = z.size
    if n <= max(10, nu + 2):
        return np.inf

    crit = np.sqrt(n) * ks_uniform(z)
    lag_max = min(nu, n - 2)
    for tau in range(1, lag_max + 1):
        k_tau = ks_bivariate_lag(z, tau=tau)
        crit = max(crit, np.sqrt(n - tau) * k_tau)
    return float(crit)


@dataclass
class ParamSelectionResult:
    h_star: float
    omega_star: float
    objective: float
    h_candidates: np.ndarray
    omega_candidates: np.ndarray
    objective_grid: np.ndarray


def compute_pit_series(
    x: np.ndarray,
    h: float,
    omega: float,
    t0: int,
    grid: np.ndarray,
) -> np.ndarray:
    """Compute PIT series using the paper's CDF recursion with kernel primitive.

    F_{t+1}(x) = omega * F_t(x) + (1 - omega) * K_cdf((x - X_{t+1}) / h)
    Z_t = F_{t-1}(X_t)
    """
    if t0 < 2 or t0 >= x.size:
        return np.array([], dtype=float)

    cdf = init_dynamic_cdf(grid=grid, x_init=x[:t0], h=h, omega=omega)
    pits = []

    for t in range(t0, x.size):
        z_t = float(np.interp(x[t], grid, cdf, left=0.0, right=1.0))
        pits.append(np.clip(z_t, 0.0, 1.0))

        # CDF recursion using kernel primitive (paper style)
        kernel_cdf = epanechnikov_kernel_cdf(grid, x[t], h)
        cdf = omega * cdf + (1.0 - omega) * kernel_cdf

    return np.asarray(pits)


def select_h_omega(
    x_calib: np.ndarray,
    t0: int,
    nu: int,
    grid: np.ndarray,
    h_grid_size: int,
    omega_grid_size: int,
    constrained: bool = True,
) -> ParamSelectionResult:
    std = float(np.std(x_calib))
    n = x_calib.size
    h_silverman = 1.06 * std * (n ** (-1.0 / 5.0))
    h_min = max(1e-3, 0.25 * h_silverman)
    h_max = max(h_min * 1.5, 2.5 * h_silverman)

    if constrained:
        omega_min = 1.0 - 1.0 / max(2, nu) + 1e-4
    else:
        omega_min = 1e-4
    omega_max = 0.999

    h_candidates = np.logspace(np.log10(h_min), np.log10(h_max), h_grid_size)
    omega_candidates = np.linspace(omega_min, omega_max, omega_grid_size)

    objective_grid = np.full((h_candidates.size, omega_candidates.size), np.inf)
    best = (np.inf, np.nan, np.nan)

    for i, h in enumerate(tqdm(h_candidates, desc="Selecting h,omega", leave=False)):
        for j, omega in enumerate(omega_candidates):
            z = compute_pit_series(x=x_calib, h=h, omega=omega, t0=t0, grid=grid)
            obj = d_nu_criterion(z, nu=nu)
            objective_grid[i, j] = obj
            if obj < best[0]:
                best = (obj, h, omega)

    return ParamSelectionResult(
        h_star=float(best[1]),
        omega_star=float(best[2]),
        objective=float(best[0]),
        h_candidates=h_candidates,
        omega_candidates=omega_candidates,
        objective_grid=objective_grid,
    )


def kl_divergence_continuous(grid: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    p_safe = np.maximum(p, EPS)
    q_safe = np.maximum(q, EPS)
    return float(np.trapezoid(p_safe * np.log(p_safe / q_safe), grid))


def first_day_hour_mask(
    times: pd.DatetimeIndex,
    start_hour: int = 9,
    end_hour: int = 10,
) -> np.ndarray:
    if end_hour <= start_hour:
        raise ValueError("reference-hour-end must be strictly greater than reference-hour-start.")
    if times.size == 0:
        return np.zeros(0, dtype=bool)

    first_day = times[0].date()
    in_first_day = (times.date == first_day)
    in_hour = (times.hour >= start_hour) & (times.hour < end_hour)
    return in_first_day & in_hour


def first_day_session_mask(
    times: pd.DatetimeIndex,
    start_hour: int = 9,
    start_minute: int = 0,
    end_hour: int = 17,
    end_minute: int = 30,
) -> np.ndarray:
    if times.size == 0:
        return np.zeros(0, dtype=bool)

    first_day = times[0].date()
    in_first_day = (times.date == first_day)
    minutes = times.hour * 60 + times.minute
    start_minutes = start_hour * 60 + start_minute
    end_minutes = end_hour * 60 + end_minute
    in_session = (minutes >= start_minutes) & (minutes <= end_minutes)
    return in_first_day & in_session


def static_kde_density(grid: np.ndarray, x_sample: np.ndarray, h: float) -> np.ndarray:
    if x_sample.size == 0:
        raise ValueError("Reference sample for static KDE is empty.")
    u = (grid[None, :] - x_sample[:, None]) / h
    kernels = np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u ** 2) / h, 0.0)
    density = np.mean(kernels, axis=0)
    return normalize_density(grid, density)


def rolling_dynamic_kl(
    x: np.ndarray,
    times: pd.DatetimeIndex,
    h: float,
    omega: float,
    t0: int,
    grid: np.ndarray,
    reference_density: np.ndarray,
    reference_time: pd.Timestamp,
    step_points: int,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    n = x.size
    if t0 < 2 or t0 >= n:
        raise ValueError("Invalid t0 for rolling KL.")

    density = init_dynamic_density(grid=grid, x_init=x[:t0], h=h, omega=omega)
    density = normalize_density(grid, density)

    reference_density = normalize_density(grid, reference_density)
    t_ref = int(np.searchsorted(times.values, reference_time.to_datetime64(), side="left"))
    t_ref = min(max(t_ref, t0 - 1), n - 1)

    kl_times: list[pd.Timestamp] = []
    kl_values: list[float] = []

    kl_times.append(times[t_ref])
    kl_values.append(kl_divergence_continuous(grid=grid, p=density, q=reference_density))

    for t in tqdm(range(t0, n), desc="Rolling KL", leave=False):
        kernel = epanechnikov_kernel_pdf(grid, x[t], h)
        density = omega * density + (1.0 - omega) * kernel
        density = normalize_density(grid, density)

        if (t >= t_ref) and ((t - t_ref) % max(1, step_points) == 0):
            kl = kl_divergence_continuous(grid=grid, p=density, q=reference_density)
            kl_times.append(times[t])
            kl_values.append(kl)

    if not kl_times:
        raise RuntimeError("No KL points were generated. Check step_points/reference_index.")

    return pd.DatetimeIndex(kl_times), np.asarray(kl_values)


def plot_kl_series(
    times: pd.DatetimeIndex,
    kl_values: np.ndarray,
    side: str,
    output: Path,
    title: str | None = None,
) -> None:
    plt.figure(figsize=(11, 4.2))

    plt.plot(times, kl_values, color="black", linewidth=1.4, label="vs reference density")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=times.tz))

    if title is None:
        title = f"Sanofi log_{side}volume1: KL(dynamic || reference)"
    plt.title(title)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.close()


def compute_density_snapshots(
    x: np.ndarray,
    times: pd.DatetimeIndex,
    grid: np.ndarray,
    h: float,
    omega: float,
    t0: int,
    target_indices: np.ndarray,
) -> dict[int, np.ndarray]:
    target_set = set(int(v) for v in np.unique(target_indices))
    target_set = {idx for idx in target_set if 0 <= idx < x.size}
    snapshots: dict[int, np.ndarray] = {}
    if not target_set:
        return snapshots

    pre_targets = sorted([idx for idx in target_set if idx < (t0 - 1)])
    for idx in tqdm(pre_targets, desc="Building pre-t0 snapshots", leave=False):
        snapshots[idx] = static_kde_density(grid=grid, x_sample=x[: idx + 1], h=h)

    post_targets = {idx for idx in target_set if idx >= (t0 - 1)}
    if not post_targets:
        return snapshots

    density = init_dynamic_density(grid=grid, x_init=x[:t0], h=h, omega=omega)
    density = normalize_density(grid, density)
    if (t0 - 1) in post_targets:
        snapshots[t0 - 1] = density.copy()

    max_target = max(post_targets)
    for t in tqdm(range(t0, max_target + 1), desc="Building PDF snapshots", leave=False):
        kernel = epanechnikov_kernel_pdf(grid, x[t], h)
        density = omega * density + (1.0 - omega) * kernel
        density = normalize_density(grid, density)
        if t in post_targets:
            snapshots[t] = density.copy()
    return snapshots


def first_day_target_indices(times: pd.DatetimeIndex, hours: list[int]) -> list[int]:
    if times.size == 0:
        return []
    first_day = times[0].date()
    day_mask = (times.date == first_day)
    day_idx = np.where(day_mask)[0]
    if day_idx.size == 0:
        return []

    day_times = times[day_mask]
    out = []
    for h in hours:
        ts = pd.Timestamp(first_day).tz_localize(times.tz) + pd.Timedelta(hours=int(h))
        pos = int(np.searchsorted(day_times.values, ts.to_datetime64(), side="left"))
        pos = min(max(pos, 0), day_times.size - 1)
        out.append(int(day_idx[pos]))
    return out


def plot_four_time_pdfs(
    grid: np.ndarray,
    densities: list[np.ndarray],
    labels: list[str],
    output: Path,
    side: str,
) -> None:
    plt.figure(figsize=(11, 4.2))
    colors = ["black", "#d62728", "#1f77b4", "#2ca02c"]
    for i, (dens, lbl) in enumerate(zip(densities, labels)):
        plt.plot(grid, dens, color=colors[i % len(colors)], linewidth=1.6, label=lbl)

    plt.title(f"Sanofi log_{side}volume1: dynamic PDFs at 10:00/12:00/14:00/16:00")
    plt.xlabel(f"log_{side}volume1")
    plt.ylabel("density")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.close()


def create_pdf_video_per_second(
    times: pd.DatetimeIndex,
    grid: np.ndarray,
    densities: np.ndarray,
    output: Path,
    side: str,
    max_frames: int = 0,
    fps: int = 30,
) -> None:
    if times.size == 0 or densities.size == 0:
        return

    n = min(times.size, int(densities.shape[0]))
    if max_frames > 0:
        n = min(n, max_frames)
    if n <= 1:
        return

    t_plot = times[:n]
    d_plot = densities[:n]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    line, = ax.plot([], [], color="black", linewidth=1.6, label="dynamic pdf")
    title = ax.set_title("")

    ax.set_xlim(grid[0], grid[-1])
    ymin = 0.0
    ymax = float(np.max(d_plot))
    if ymax <= ymin:
        ymax = ymin + 1.0
    margin = 0.08 * (ymax - ymin)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_xlabel(f"log_{side}volume1")
    ax.set_ylabel("density")
    ax.legend(loc="best", fontsize=9)

    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        with imageio.get_writer(str(output), fps=max(1, fps), codec="libx264", quality=7) as writer:
            for frame in tqdm(range(n), desc="Writing MP4", leave=False):
                line.set_data(grid, d_plot[frame])
                title.set_text(
                    f"Sanofi log_{side}volume1 PDF - {t_plot[frame].strftime('%Y-%m-%d %H:%M:%S')}"
                )
                fig.canvas.draw()
                rgba = np.asarray(fig.canvas.buffer_rgba())
                writer.append_data(rgba[:, :, :3])
    except Exception as exc:
        raise RuntimeError(
            "Failed to export MP4 video. Please check imageio-ffmpeg installation."
        ) from exc
    plt.close(fig)


def main() -> None:
    args = parse_args()

    times, x = load_log_volume_series(path=args.input, side=args.side, tz=args.tz)
    n = x.size

    # ---- Reference window: 9:00-10:00 (first day) ----
    ref_mask = first_day_hour_mask(
        times=times,
        start_hour=args.reference_hour_start,
        end_hour=args.reference_hour_end,
    )
    if not np.any(ref_mask):
        raise RuntimeError("No observations found in requested reference window (first day 9:00-10:00 by default).")
    ref_indices = np.where(ref_mask)[0]
    x_ref = x[ref_indices]
    ref_times = times[ref_mask]

    # ---- t0 = first index at or after 10:00 (end of reference window) ----
    t0 = int(ref_indices[-1]) + 1
    if t0 >= n:
        raise RuntimeError("No observations after reference window to run dynamic KDE.")
    print(f"Reference window: {ref_times[0].strftime('%H:%M:%S')} – {ref_times[-1].strftime('%H:%M:%S')}  "
          f"({ref_indices.size} points)")
    print(f"Dynamic recursion starts at t0={t0}  ({times[t0].strftime('%H:%M:%S')})")

    # ---- Calibration (h, omega) using 9:00-10:00 data only ----
    calib_t0 = min(max(20, int(x_ref.size * args.calib_init_frac)), x_ref.size - 5)
    calib_grid = make_x_grid(
        x=x_ref,
        grid_size=args.grid_size,
        q_low=args.grid_quantile_low,
        q_high=args.grid_quantile_high,
    )
    selection = select_h_omega(
        x_calib=x_ref,
        t0=calib_t0,
        nu=args.nu,
        grid=calib_grid,
        h_grid_size=args.h_grid_size,
        omega_grid_size=args.omega_grid_size,
        constrained=True,
    )
    print(f"Selected parameters: h*={selection.h_star:.6f}, omega*={selection.omega_star:.6f}")
    print(f"d_nu objective value: {selection.objective:.6f}")

    # ---- Full grid for downstream use ----
    full_grid = make_x_grid(
        x=x,
        grid_size=args.grid_size,
        q_low=args.grid_quantile_low,
        q_high=args.grid_quantile_high,
    )

    # ---- Reference density = static KDE on 9:00-10:00 ----
    reference_density = static_kde_density(full_grid, x_ref, selection.h_star)
    reference_time = ref_times[-1]

    # ---- Rolling KL (dynamic starts at 10:00) ----
    kl_times, kl_values = rolling_dynamic_kl(
        x=x,
        times=times,
        h=selection.h_star,
        omega=selection.omega_star,
        t0=t0,
        grid=full_grid,
        reference_density=reference_density,
        reference_time=reference_time,
        step_points=args.kl_step_points,
    )

    ref_date = ref_times[0].date()
    plot_title = (
        f"Sanofi log_{args.side}volume1: KL(dynamic || reference 9:00-10:00) - {ref_date}"
    )
    plot_kl_series(times=kl_times, kl_values=kl_values, side=args.side, output=args.output, title=plot_title)
    print(f"Saved KL plot to: {args.output}")

    # ---- Density snapshots: four-time PDFs + video (all >= t0, i.e. >= 10:00) ----
    target_hours = list(args.pdf_hours)
    target_indices = first_day_target_indices(times=times, hours=target_hours)
    target_for_snapshots = target_indices.copy()

    session_indices = np.array([], dtype=int)
    if not args.skip_video:
        session_mask = first_day_session_mask(times=times, start_hour=10, start_minute=0, end_hour=17, end_minute=30)
        session_indices = np.where(session_mask)[0]
        if session_indices.size == 0:
            raise RuntimeError("No observations found in first-day session 10:00-17:30.")
        if args.video_max_frames > 0:
            session_indices = session_indices[: args.video_max_frames]
        target_for_snapshots.extend(session_indices.tolist())

    snapshots = compute_density_snapshots(
        x=x,
        times=times,
        grid=full_grid,
        h=selection.h_star,
        omega=selection.omega_star,
        t0=t0,
        target_indices=np.asarray(target_for_snapshots, dtype=int),
    )

    if len(target_indices) == 4 and all(idx in snapshots for idx in target_indices):
        pdf_dens = [snapshots[idx] for idx in target_indices]
        pdf_labels = [times[idx].strftime("%H:%M") for idx in target_indices]
        plot_four_time_pdfs(
            grid=full_grid,
            densities=pdf_dens,
            labels=pdf_labels,
            output=args.four_pdf_output,
            side=args.side,
        )
        print(f"Saved 10/12/14/16 PDF figure to: {args.four_pdf_output}")
    else:
        print("Skipped 10/12/14/16 PDF figure: not enough snapshots available.")

    if not args.skip_video:
        needed_indices = session_indices
        missing = [idx for idx in needed_indices if idx not in snapshots]
        if missing:
            extra_snapshots = compute_density_snapshots(
                x=x,
                times=times,
                grid=full_grid,
                h=selection.h_star,
                omega=selection.omega_star,
                t0=t0,
                target_indices=np.asarray(missing, dtype=int),
            )
            snapshots.update(extra_snapshots)

        anim_indices = needed_indices
        anim_times = times[anim_indices]
        anim_densities = np.stack([snapshots[idx] for idx in anim_indices], axis=0)
        create_pdf_video_per_second(
            times=anim_times,
            grid=full_grid,
            densities=anim_densities,
            output=args.video_output,
            side=args.side,
            max_frames=args.video_max_frames,
            fps=args.video_fps,
        )
        print(f"Saved video to: {args.video_output}")


if __name__ == "__main__":
    main()
