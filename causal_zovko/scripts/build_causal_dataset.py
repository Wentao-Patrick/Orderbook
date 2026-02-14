"""
Create lag-augmented causal dataset from bucket-level features.

Inputs/outputs are inside causal_zovko/data by default:
- input: features_1min.csv
- output: causal_dataset_1min.csv

Use relative defaults so the script can run from any working directory
inside the EA_recherche workspace.
"""

import argparse
from pathlib import Path
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CAUSAL_ZOVKO_DIR = SCRIPT_DIR.parent


def add_lags(df: pd.DataFrame, cols, lags):
    out = df.copy()
    for col in cols:
        for L in lags:
            out[f"{col}_lag{L}"] = out[col].shift(L)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default=str(CAUSAL_ZOVKO_DIR / "data" / "features_1min.csv"))
    ap.add_argument("--out_csv", default=str(CAUSAL_ZOVKO_DIR / "data" / "causal_dataset_1min.csv"))
    ap.add_argument("--lags", default="1,2,5")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv, parse_dates=["bucket"]).set_index("bucket")

    # keep key columns
    base_cols = [
        "rlop_ask_mean","rlop_bid_mean","vol_bid_mean","vol_ask_mean","spread_mean","imbalance_ob_mean",
        "bid_depth_mean","ask_depth_mean","imb_of",
        "limit_new_bid_count","limit_new_ask_count","cancel_bid_count","cancel_ask_count","trade_count",
        "signed_volume","abs_signed_volume",
    ]
    # only keep those that exist
    base_cols = [c for c in base_cols if c in df.columns]
    df = df[base_cols]

    lags = [int(x) for x in args.lags.split(",") if x.strip()]
    df_lag = add_lags(df, base_cols, lags)

    df_lag = df_lag.dropna()
    df_lag.to_csv(args.out_csv)
    print("saved:", args.out_csv, "rows:", len(df_lag))


if __name__ == "__main__":
    main()
