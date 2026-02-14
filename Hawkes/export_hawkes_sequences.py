# -*- coding: utf-8 -*-
"""
Export Scheme-A Hawkes event time series (buyer-initiated vs seller-initiated trades)
from a decoded Euronext 'decoded_full_trade_information.csv'.

Inputs
------
- decoded_full_trade_information.csv (your decoded output)

Outputs
-------
- sanofi_hawkes_buy_times.csv   (sign=+1)
- sanofi_hawkes_sell_times.csv  (sign=-1)

Notes
-----
- Filters to keep a clean microstructure sample:
  * MMTMarketMechanism == 1  (Central Limit Order Book)
  * MMTTradingMode == "2"    (Continuous Trading)
  * sign in {+1, -1}         (direction inferred)
- Creates:
  * t_seconds: seconds since first retained event (common clock for multivariate Hawkes)
  * t_seconds_jitter: adds tiny jitter to break ties for identical timestamps
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def export_hawkes_sequences(
    in_csv: str,
    out_buy: str,
    out_sell: str,
    venue: str | None = None,
    eps: float = 1e-6,
) -> None:
    df = pd.read_csv(in_csv)

    # Parse timestamps (timezone-aware strings in the CSV)
    df["trade_time_paris"] = pd.to_datetime(df["trade_time_paris"], errors="coerce")
    df = df.dropna(subset=["trade_time_paris"]).copy()

    # Clean filters: CLOB + continuous trading + known sign
    df = df[
        (df["MMTMarketMechanism"] == 1)
        & (df["MMTTradingMode"].astype(str) == "2")
        & (df["sign"].isin([1, -1]))
    ].copy()

    # Optional venue filter (e.g. "XPAR")
    if venue is not None:
        df = df[df["Venue"].astype(str) == venue].copy()

    if df.empty:
        raise ValueError("No rows left after filtering. Check MMTTradingMode / venue / sign logic.")

    # Sort and build common time axis (seconds since first retained event)
    df = df.sort_values("trade_time_paris").reset_index(drop=True)
    t0 = df["trade_time_paris"].iloc[0]
    df["t_seconds"] = (df["trade_time_paris"] - t0).dt.total_seconds()

    # Jitter identical timestamps (many Hawkes libs require strict increasing times)
    df["dup_rank"] = df.groupby("trade_time_paris").cumcount()
    df["t_seconds_jitter"] = df["t_seconds"] + eps * df["dup_rank"]

    buy = df[df["sign"] == 1].copy()
    sell = df[df["sign"] == -1].copy()

    cols = ["trade_time_paris", "t_seconds", "t_seconds_jitter", "MiFIDPrice", "MiFIDQuantity", "Venue"]
    buy[cols].to_csv(out_buy, index=False)
    sell[cols].to_csv(out_sell, index=False)

    print(f"[OK] saved: {out_buy} (n={len(buy)})")
    print(f"[OK] saved: {out_sell} (n={len(sell)})")
    print(f"[INFO] t0 (common origin) = {t0}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default=r"C:\Users\Wentao\Desktop\EA_recherche\decoded_full_trade_information.csv")
    ap.add_argument("--out_buy", default="sanofi_hawkes_buy_times.csv")
    ap.add_argument("--out_sell", default="sanofi_hawkes_sell_times.csv")
    ap.add_argument("--venue", default=None, help='Optional filter, e.g. "XPAR"')
    ap.add_argument("--eps", type=float, default=1e-6, help="Jitter size in seconds (default 1e-6)")
    args = ap.parse_args()

    export_hawkes_sequences(
        in_csv=args.in_csv,
        out_buy=args.out_buy,
        out_sell=args.out_sell,
        venue=args.venue,
        eps=args.eps,
    )


if __name__ == "__main__":
    main()
