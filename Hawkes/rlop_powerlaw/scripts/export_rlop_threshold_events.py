# -*- coding: utf-8 -*-
"""
Export RLOP exceedance event times for Hawkes modeling.

Input
-----
- causal_zovko/data/rlop_events.csv

Output
------
- CSV with thresholded event times and metadata required by the
  power-law Hawkes fitting script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RLOP_POWERLAW_DIR = SCRIPT_DIR.parent
EA_RECHERCHE_ROOT = RLOP_POWERLAW_DIR.parent.parent
DEFAULT_INPUT_CSV = EA_RECHERCHE_ROOT / "causal_zovko" / "data" / "rlop_events.csv"
DEFAULT_DERIVED_DIR = RLOP_POWERLAW_DIR / "data" / "derived"


def side_to_code(side: str) -> int:
    side_l = side.strip().lower()
    if side_l == "bid":
        return 1
    if side_l == "ask":
        return 2
    raise ValueError("side must be 'bid' or 'ask'")


def threshold_tag(mode: str, value: float) -> str:
    if mode == "quantile":
        return f"q{int(round(value * 100))}"
    return f"abs_{value:.6f}".replace(".", "p")


def default_out_csv(side: str, mode: str, value: float) -> Path:
    tag = threshold_tag(mode, value)
    return DEFAULT_DERIVED_DIR / f"rlop_{side}_events_{tag}.csv"


def export_threshold_events(
    in_csv: str,
    side: str,
    threshold_mode: str,
    threshold_value: float,
    eps: float,
    out_csv: str,
) -> dict[str, object]:
    df = pd.read_csv(in_csv, parse_dates=["time_paris"])
    side_code = side_to_code(side)

    side_df = df[df["side"] == side_code].copy()
    side_df = side_df.dropna(subset=["time_paris", "delta"]).sort_values("time_paris").reset_index(drop=True)
    if side_df.empty:
        raise ValueError(f"No RLOP rows found for side={side}.")

    if threshold_mode == "quantile":
        threshold_actual = float(side_df["delta"].quantile(threshold_value))
    elif threshold_mode == "absolute":
        threshold_actual = float(threshold_value)
    else:
        raise ValueError("threshold_mode must be 'quantile' or 'absolute'.")

    events = side_df[side_df["delta"] >= threshold_actual].copy()
    if events.empty:
        raise ValueError("No exceedance events found. Lower the threshold.")

    time_origin = side_df["time_paris"].iloc[0]
    window_start = side_df["time_paris"].iloc[0]
    window_end = side_df["time_paris"].iloc[-1]
    observation_T = float((window_end - window_start).total_seconds())

    events["t_seconds"] = (events["time_paris"] - time_origin).dt.total_seconds()
    events["dup_rank"] = events.groupby("time_paris").cumcount()
    events["t_seconds_jitter"] = events["t_seconds"] + eps * events["dup_rank"]

    events["side_label"] = side
    events["threshold_mode"] = threshold_mode
    events["threshold_value_input"] = float(threshold_value)
    events["threshold_actual"] = threshold_actual
    events["time_origin_paris"] = time_origin.isoformat()
    events["window_start_paris"] = window_start.isoformat()
    events["window_end_paris"] = window_end.isoformat()
    events["T_observation_seconds"] = observation_T

    cols = [
        "time_paris",
        "t_ns",
        "side",
        "side_label",
        "delta",
        "bid",
        "ask",
        "price",
        "t_seconds",
        "dup_rank",
        "t_seconds_jitter",
        "threshold_mode",
        "threshold_value_input",
        "threshold_actual",
        "time_origin_paris",
        "window_start_paris",
        "window_end_paris",
        "T_observation_seconds",
    ]
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    events[cols].to_csv(out_path, index=False)

    duration_exceed = float(events["t_seconds_jitter"].iloc[-1] - events["t_seconds_jitter"].iloc[0]) if len(events) > 1 else 0.0
    summary = {
        "side": side,
        "side_code": side_code,
        "input_csv": str(in_csv),
        "output_csv": str(out_path),
        "threshold_mode": threshold_mode,
        "threshold_value_input": float(threshold_value),
        "threshold_actual": threshold_actual,
        "n_side_rows": int(len(side_df)),
        "n_events": int(len(events)),
        "window_start_paris": window_start.isoformat(),
        "window_end_paris": window_end.isoformat(),
        "T_observation_seconds": observation_T,
        "event_span_seconds": duration_exceed,
        "mean_delta_events": float(events["delta"].mean()),
        "median_delta_events": float(events["delta"].median()),
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default=str(DEFAULT_INPUT_CSV))
    ap.add_argument("--side", choices=["bid", "ask"], default="bid")
    ap.add_argument("--threshold_mode", choices=["quantile", "absolute"], default="quantile")
    ap.add_argument("--threshold_value", type=float, default=0.99)
    ap.add_argument("--eps", type=float, default=1e-9, help="Jitter size in seconds for duplicated timestamps.")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    out_csv = args.out_csv or str(default_out_csv(args.side, args.threshold_mode, args.threshold_value))
    summary = export_threshold_events(
        in_csv=args.in_csv,
        side=args.side,
        threshold_mode=args.threshold_mode,
        threshold_value=args.threshold_value,
        eps=args.eps,
        out_csv=out_csv,
    )

    print("[OK] saved:", summary["output_csv"])
    print(
        "[INFO] side={side} threshold={thr:.6f} n_events={n} T={T:.3f}s".format(
            side=summary["side"],
            thr=summary["threshold_actual"],
            n=summary["n_events"],
            T=summary["T_observation_seconds"],
        )
    )


if __name__ == "__main__":
    main()
