"""
Time-varying KDE + rolling KL on 1-minute RLOP mean series.

This script mirrors the dynamic KDE workflow used in
log_volume/scripts/time_varying_kde_rolling_kl.py, but applies it to
causal_zovko/data/rlop_{side}_1min.csv where the input series is the
1-minute bucket mean of RLOP (`delta_mean`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RLOP_1MIN_KDE_DIR = SCRIPT_DIR.parent
EA_RECHERCHE_ROOT = RLOP_1MIN_KDE_DIR.parent
CAUSAL_ZOVKO_DATA_DIR = EA_RECHERCHE_ROOT / "causal_zovko" / "data"

if str(EA_RECHERCHE_ROOT) not in sys.path:
    sys.path.insert(0, str(EA_RECHERCHE_ROOT))

from log_volume.scripts.time_varying_kde_rolling_kl import (  # noqa: E402
    compute_density_snapshots,
    first_day_hour_mask,
    first_day_session_mask,
    first_day_target_indices,
    make_x_grid,
    rolling_dynamic_kl,
    select_h_omega,
    side_tagged_path,
    static_kde_density,
)


def default_input_for_side(side: str) -> Path:
    return CAUSAL_ZOVKO_DATA_DIR / f"rlop_{side}_1min.csv"


def parse_args() -> argparse.Namespace:
    results_dir = RLOP_1MIN_KDE_DIR / "results"
    parser = argparse.ArgumentParser(description="Time-varying KDE rolling KL on 1-minute RLOP mean.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional CSV path. Default: causal_zovko/data/rlop_{side}_1min.csv",
    )
    parser.add_argument(
        "--side",
        type=str,
        choices=["bid", "ask"],
        default="bid",
        help="Book side used to load rlop_{side}_1min.csv.",
    )
    parser.add_argument(
        "--value-col",
        type=str,
        default="delta_mean",
        help="Series column used for KDE / KL.",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default="time_paris",
        help="Timestamp column in the RLOP CSV.",
    )
    parser.add_argument(
        "--tz",
        type=str,
        default="Europe/Paris",
        help="Timezone used for plotting and window selection.",
    )
    parser.add_argument(
        "--nu",
        type=int,
        default=10,
        help="Max lag used in d_nu criterion. Lower than log_volume because the 1-minute reference window has 60 points.",
    )
    parser.add_argument(
        "--calib-init-frac",
        type=float,
        default=0.33,
        help="Fraction of 9:00-10:00 data used for initialization in calibration.",
    )
    parser.add_argument("--h-grid-size", type=int, default=12)
    parser.add_argument("--omega-grid-size", type=int, default=12)
    parser.add_argument(
        "--grid-size",
        type=int,
        default=256,
        help="Number of grid points used for density representation.",
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
        default=1,
        help="Sampling step in observations for KL curve points. Default 1 minute.",
    )
    parser.add_argument(
        "--reference-hour-start",
        type=int,
        default=9,
        help="Start hour (inclusive) for the first-day reference window.",
    )
    parser.add_argument(
        "--reference-hour-end",
        type=int,
        default=10,
        help="End hour (exclusive) for the first-day reference window.",
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        default=results_dir / "rlop_1min_video.mp4",
        help="MP4 output path for dynamic PDF video.",
    )
    parser.add_argument(
        "--video-max-frames",
        type=int,
        default=0,
        help="Maximum number of frames in video (0 = full 10:00-17:30 session).",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=15,
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
        default=results_dir / "rlop_1min_pdf_10_12_14_16.png",
        help="Output path for the 10:00/12:00/14:00/16:00 PDF comparison figure.",
    )
    parser.add_argument(
        "--pdf-hours",
        type=int,
        nargs=4,
        default=[10, 12, 14, 16],
        help="Four first-day hours used for the PDF overlay figure.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=results_dir / "rolling_kl_time_varying_kde_rlop_1min.png",
        help="KL plot output path.",
    )
    parser.add_argument(
        "--kl-csv-output",
        type=Path,
        default=results_dir / "rolling_kl_time_varying_kde_rlop_1min.csv",
        help="Rolling KL values CSV output path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=results_dir / "kde_selection_rlop_1min.csv",
        help="Selection summary CSV output path.",
    )
    return parser.parse_args()


def load_rlop_mean_series(
    path: Path,
    side: str,
    value_col: str,
    time_col: str,
    tz: str,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path, parse_dates=[time_col])
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in input file.")
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in input file.")

    idx = pd.DatetimeIndex(pd.to_datetime(df[time_col], errors="coerce"))
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    else:
        idx = idx.tz_convert(tz)

    values = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(values) & ~pd.isna(idx)
    idx = idx[mask]
    x = values[mask]

    order = np.argsort(idx.values)
    idx = idx[order]
    x = x[order]

    if x.size < 50:
        raise ValueError(f"Not enough valid observations for side={side}.")
    return idx, x


def effective_nu(requested_nu: int, n_calib: int, calib_t0: int) -> int:
    pit_n = max(0, n_calib - calib_t0)
    max_nu = max(1, pit_n - 3)
    return min(requested_nu, max_nu)


def plot_kl_series(
    times: pd.DatetimeIndex,
    kl_values: np.ndarray,
    output: Path,
    title: str,
) -> None:
    plt.figure(figsize=(11, 4.2))
    plt.plot(times, kl_values, color="black", linewidth=1.4, label="vs reference density")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=times.tz))

    plt.title(title)
    plt.ylabel("KL divergence")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.close()


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

    plt.title(f"Sanofi rlop_{side}_mean_1min: dynamic PDFs at 10:00/12:00/14:00/16:00")
    plt.xlabel(f"rlop_{side}_mean_1min")
    plt.ylabel("density")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.close()


def create_density_video(
    times: pd.DatetimeIndex,
    grid: np.ndarray,
    densities: np.ndarray,
    output: Path,
    side: str,
    max_frames: int,
    fps: int,
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
    ax.set_xlabel(f"rlop_{side}_mean_1min")
    ax.set_ylabel("density")
    ax.legend(loc="best", fontsize=9)

    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        with imageio.get_writer(str(output), fps=max(1, fps), codec="libx264", quality=7) as writer:
            for frame in range(n):
                line.set_data(grid, d_plot[frame])
                title.set_text(
                    f"Sanofi rlop_{side}_mean_1min PDF - {t_plot[frame].strftime('%Y-%m-%d %H:%M')}"
                )
                fig.canvas.draw()
                rgba = np.asarray(fig.canvas.buffer_rgba())
                writer.append_data(rgba[:, :, :3])
    except Exception as exc:
        raise RuntimeError(
            "Failed to export MP4 video. Please check imageio-ffmpeg installation."
        ) from exc
    finally:
        plt.close(fig)


def save_outputs(
    kl_times: pd.DatetimeIndex,
    kl_values: np.ndarray,
    kl_csv_output: Path,
    summary_output: Path,
    summary: dict[str, object],
) -> None:
    kl_df = pd.DataFrame(
        {
            "time_paris": kl_times,
            "kl_dynamic_vs_reference": kl_values,
        }
    )
    kl_csv_output.parent.mkdir(parents=True, exist_ok=True)
    kl_df.to_csv(kl_csv_output, index=False)

    summary_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(summary_output, index=False)


def main() -> None:
    args = parse_args()

    data_path = args.input if args.input is not None else default_input_for_side(args.side)
    args.output = side_tagged_path(args.output, args.side)
    args.kl_csv_output = side_tagged_path(args.kl_csv_output, args.side)
    args.summary_output = side_tagged_path(args.summary_output, args.side)
    args.four_pdf_output = side_tagged_path(args.four_pdf_output, args.side)
    args.video_output = side_tagged_path(args.video_output, args.side)

    times, x = load_rlop_mean_series(
        path=data_path,
        side=args.side,
        value_col=args.value_col,
        time_col=args.time_col,
        tz=args.tz,
    )

    ref_mask = first_day_hour_mask(
        times=times,
        start_hour=args.reference_hour_start,
        end_hour=args.reference_hour_end,
    )
    if not np.any(ref_mask):
        raise RuntimeError("No observations found in the requested reference window.")

    ref_indices = np.where(ref_mask)[0]
    x_ref = x[ref_indices]
    ref_times = times[ref_mask]
    t0 = int(ref_indices[-1]) + 1
    if t0 >= x.size:
        raise RuntimeError("No observations after reference window to run dynamic KDE.")

    calib_t0 = min(max(10, int(x_ref.size * args.calib_init_frac)), x_ref.size - 5)
    nu_eff = effective_nu(args.nu, x_ref.size, calib_t0)

    calib_grid = make_x_grid(
        x=x_ref,
        grid_size=args.grid_size,
        q_low=args.grid_quantile_low,
        q_high=args.grid_quantile_high,
    )
    selection = select_h_omega(
        x_calib=x_ref,
        t0=calib_t0,
        nu=nu_eff,
        grid=calib_grid,
        h_grid_size=args.h_grid_size,
        omega_grid_size=args.omega_grid_size,
        constrained=True,
    )

    full_grid = make_x_grid(
        x=x,
        grid_size=args.grid_size,
        q_low=args.grid_quantile_low,
        q_high=args.grid_quantile_high,
    )
    reference_density = static_kde_density(full_grid, x_ref, selection.h_star)
    reference_time = ref_times[-1]

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
        f"Sanofi rlop_{args.side}_mean_1min: KL(dynamic || reference 9:00-10:00) - {ref_date}"
    )
    plot_kl_series(
        times=kl_times,
        kl_values=kl_values,
        output=args.output,
        title=plot_title,
    )

    summary = {
        "side": args.side,
        "input_csv": str(data_path),
        "value_col": args.value_col,
        "n_total": int(x.size),
        "reference_points": int(x_ref.size),
        "reference_start": ref_times[0].isoformat(),
        "reference_end": ref_times[-1].isoformat(),
        "dynamic_start": times[t0].isoformat(),
        "h_star": selection.h_star,
        "omega_star": selection.omega_star,
        "objective_d_nu": selection.objective,
        "nu_requested": int(args.nu),
        "nu_effective": int(nu_eff),
        "calib_t0": int(calib_t0),
        "grid_size": int(args.grid_size),
        "h_grid_size": int(args.h_grid_size),
        "omega_grid_size": int(args.omega_grid_size),
        "kl_step_points": int(args.kl_step_points),
    }
    save_outputs(
        kl_times=kl_times,
        kl_values=kl_values,
        kl_csv_output=args.kl_csv_output,
        summary_output=args.summary_output,
        summary=summary,
    )

    print(f"Input: {data_path}")
    print(
        f"Reference window: {ref_times[0].strftime('%H:%M')} - "
        f"{ref_times[-1].strftime('%H:%M')} ({x_ref.size} points)"
    )
    print(f"Dynamic recursion starts at: {times[t0].strftime('%Y-%m-%d %H:%M')}")
    print(
        f"Selected parameters: h*={selection.h_star:.6f}, "
        f"omega*={selection.omega_star:.6f}, d_nu={selection.objective:.6f}"
    )
    if nu_eff != args.nu:
        print(f"Adjusted nu from {args.nu} to {nu_eff} for the available calibration sample.")
    print(f"Saved KL plot to: {args.output}")
    print(f"Saved KL csv to: {args.kl_csv_output}")
    print(f"Saved summary csv to: {args.summary_output}")

    target_indices = first_day_target_indices(times=times, hours=list(args.pdf_hours))
    target_for_snapshots = target_indices.copy()

    session_indices = np.array([], dtype=int)
    if not args.skip_video:
        session_mask = first_day_session_mask(
            times=times,
            start_hour=10,
            start_minute=0,
            end_hour=17,
            end_minute=30,
        )
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
        anim_times = times[session_indices]
        anim_densities = np.stack([snapshots[idx] for idx in session_indices], axis=0)
        create_density_video(
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
