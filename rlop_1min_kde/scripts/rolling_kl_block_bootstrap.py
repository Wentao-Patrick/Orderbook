from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RLOP_1MIN_KDE_DIR = SCRIPT_DIR.parent
EA_RECHERCHE_ROOT = RLOP_1MIN_KDE_DIR.parent
CAUSAL_ZOVKO_DATA_DIR = EA_RECHERCHE_ROOT / "causal_zovko" / "data"
RESULTS_DIR = RLOP_1MIN_KDE_DIR / "results"

if str(EA_RECHERCHE_ROOT) not in sys.path:
    sys.path.insert(0, str(EA_RECHERCHE_ROOT))

from log_volume.scripts.rolling_kl_block_bootstrap_core import (  # noqa: E402
    plot_kl_with_thresholds,
    run_block_bootstrap_analysis,
    save_bootstrap_outputs,
)
from log_volume.scripts.time_varying_kde_rolling_kl import side_tagged_path  # noqa: E402
from rlop_1min_kde.scripts.time_varying_kde_rolling_kl import (  # noqa: E402
    default_input_for_side,
    effective_nu,
    load_rlop_mean_series,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Block-bootstrap significance thresholds for rolling KL on 1-minute mean RLOP."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional CSV path. Default: causal_zovko/data/rlop_{side}_1min.csv",
    )
    parser.add_argument("--side", type=str, choices=["bid", "ask"], default="bid")
    parser.add_argument("--value-col", type=str, default="delta_mean")
    parser.add_argument("--time-col", type=str, default="time_paris")
    parser.add_argument("--tz", type=str, default="Europe/Paris")
    parser.add_argument("--nu", type=int, default=10)
    parser.add_argument("--calib-init-frac", type=float, default=0.33)
    parser.add_argument("--h-grid-size", type=int, default=12)
    parser.add_argument("--omega-grid-size", type=int, default=12)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--grid-quantile-low", type=float, default=0.002)
    parser.add_argument("--grid-quantile-high", type=float, default=0.998)
    parser.add_argument("--kl-step-points", type=int, default=1)
    parser.add_argument("--reference-hour-start", type=int, default=9)
    parser.add_argument("--reference-hour-end", type=int, default=10)
    parser.add_argument("--block-length", type=int, default=5, help="Bootstrap block length in minutes.")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--h-fixed", type=float, default=None, help="Optional fixed h* to skip parameter selection.")
    parser.add_argument("--omega-fixed", type=float, default=None, help="Optional fixed omega* to skip parameter selection.")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "rolling_kl_block_bootstrap_threshold_rlop_1min.png",
        help="PNG path for observed KL with bootstrap thresholds.",
    )
    parser.add_argument(
        "--threshold-csv-output",
        type=Path,
        default=RESULTS_DIR / "rolling_kl_block_bootstrap_threshold_rlop_1min.csv",
        help="CSV path for observed KL, thresholds, and p-values.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=RESULTS_DIR / "rolling_kl_block_bootstrap_summary_rlop_1min.csv",
        help="CSV path for bootstrap summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.input if args.input is not None else default_input_for_side(args.side)
    args.output = side_tagged_path(args.output, args.side)
    args.threshold_csv_output = side_tagged_path(args.threshold_csv_output, args.side)
    args.summary_output = side_tagged_path(args.summary_output, args.side)

    times, x = load_rlop_mean_series(
        path=data_path,
        side=args.side,
        value_col=args.value_col,
        time_col=args.time_col,
        tz=args.tz,
    )
    ref_size = int(((times.date == times[0].date()) & (times.hour >= args.reference_hour_start) & (times.hour < args.reference_hour_end)).sum())
    calib_t0 = min(max(10, int(ref_size * args.calib_init_frac)), ref_size - 5)
    nu_eff = effective_nu(args.nu, ref_size, calib_t0)

    result = run_block_bootstrap_analysis(
        times=times,
        x=x,
        nu_used=nu_eff,
        calib_t0=calib_t0,
        h_grid_size=args.h_grid_size,
        omega_grid_size=args.omega_grid_size,
        grid_size=args.grid_size,
        grid_quantile_low=args.grid_quantile_low,
        grid_quantile_high=args.grid_quantile_high,
        kl_step_points=args.kl_step_points,
        reference_hour_start=args.reference_hour_start,
        reference_hour_end=args.reference_hour_end,
        block_length=args.block_length,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
        seed=args.seed,
        h_fixed=args.h_fixed,
        omega_fixed=args.omega_fixed,
    )

    title = (
        f"Sanofi rlop_{args.side}_mean_1min: KL with block-bootstrap thresholds "
        f"({result.reference_start.date()})"
    )
    plot_kl_with_thresholds(
        times=result.times,
        observed_kl=result.observed_kl,
        pointwise_threshold=result.pointwise_threshold,
        global_threshold=result.global_threshold,
        output=args.output,
        title=title,
    )
    save_bootstrap_outputs(
        result=result,
        output_csv=args.threshold_csv_output,
        summary_csv=args.summary_output,
        extra_summary={
            "side": args.side,
            "input_file": str(data_path),
            "series_name": f"rlop_{args.side}_mean_1min",
            "value_col": args.value_col,
            "block_length": int(args.block_length),
            "alpha": float(args.alpha),
            "seed": int(args.seed),
            "kl_step_points": int(args.kl_step_points),
            "nu_requested": int(args.nu),
            "used_fixed_h_omega": bool(args.h_fixed is not None and args.omega_fixed is not None),
        },
    )

    print(f"Input: {data_path}")
    print(
        f"Selected parameters: h*={result.h_star:.6f}, "
        f"omega*={result.omega_star:.6f}, d_nu={result.objective:.6f}"
    )
    print(
        f"Global 95% threshold={result.global_threshold:.6f}, "
        f"observed max KL={result.observed_kl.max():.6f}, "
        f"global p-value={result.global_p_value:.4f}"
    )
    if nu_eff != args.nu:
        print(f"Adjusted nu from {args.nu} to {nu_eff}.")
    print(f"Saved plot to: {args.output}")
    print(f"Saved threshold csv to: {args.threshold_csv_output}")
    print(f"Saved summary csv to: {args.summary_output}")


if __name__ == "__main__":
    main()
