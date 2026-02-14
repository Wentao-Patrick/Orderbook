# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0) Parameters
# ============================================================
INPUT_TRADE_CSV = r"C:\Users\Wentao\Desktop\EA_recherche\euronextparis\EuronextParis\EuronextParis_20191001_FR0000120578\FR0000120578\FullTradeInformation_20191001_FR0000120578.csv"  # change to your local path
TZ_LOCAL = "Europe/Paris"

OUT_FIG_TIMEBAR = "sanofi_signed_flow_3min.png"
OUT_FIG_COUNTBAR = "sanofi_signed_flow_countbar.png"
OUT_FIG_TICK_HIST = "sanofi_price_jump_tick_hist.png"

# Tick size depends on liquidity band and price range (ESMA MiFID II table).
# Set the correct liquidity band for the instrument (1..6).
LIQUIDITY_BAND = 6  # TODO: confirm for FR0000120578 (Sanofi)


# ============================================================
# 1) Column schema (from the PDF: Euronext Paris V2, from 2018)
#    Note: Trade Qualifier is a sequence like [1,{...}]
#    It *looks* like a single CSV field, but contains commas, so we must
#    extract it first before splitting by ','.
# ============================================================
FIELDS = [
    # --- header ---
    "MarketDataSequenceNumber",     # 0
    "MessageType",                  # 1
    "RebroadcastIndicator",         # 2
    "EMM",                          # 3
    "EventTimeNs",                  # 4 (epoch ns UTC)
    "SymbolIndex",                  # 5
    "TradingDateTimeUtc",           # 6 (ISO string, UTC)
    "PublicationDateTimeUtc",       # 7 (ISO string, UTC)
    # --- MiFID ---
    "TradeType",                    # 8
    "MiFIDInstrumentIdType",        # 9  (always ISIN)
    "MiFIDInstrumentId",            # 10
    "MiFIDExecutionId",             # 11
    "MiFIDPrice",                   # 12
    "MiFIDQuantity",                # 13
    "MiFIDPriceNotation",           # 14
    "MiFIDCurrency",                # 15
    "MiFIDQtyInMeasurementUnitNotation",  # 16 (often empty)
    "MiFIDQuantityMeasurementUnit",       # 17 (often empty)
    "MiFIDNotionalAmount",          # 18 (optional)
    "NotionalCurrency",             # 19
    "MiFIDClearingFlag",            # 20 (often '-' / empty)
    # --- MMT Level fields ---
    "MMTMarketMechanism",           # 21
    "MMTTradingMode",               # 22
    "MMTTransactionCategory",       # 23
    "MMTNegotiationIndicator",      # 24
    "MMTAgencyCrossTradeIndicator", # 25
    "MMTModificationIndicator",     # 26
    "MMTBenchmarkIndicator",        # 27
    "MMTSpecialDividendIndicator",  # 28
    "MMTOffBookAutomatedIndicator", # 29
    "MMTContributionToPrice",       # 30
    "MMTAlgorithmicIndicator",      # 31
    "MMTPublicationMode",           # 32
    "MMTPostTradeDeferral",         # 33
    "MMTDuplicativeIndicator",      # 34
    # --- sequence ---
    "TradeQualifierSeq",            # 35 (sequence like [1,{...}])
    # --- post qualifiers ---
    "TransactionType",              # 36
    "EffectiveDateIndicator",       # 37
    "BlockTradeCode",               # 38
    "TradeReference",               # 39
    "OriginalReportTimestampNs",    # 40
    "TransparencyIndicator",        # 41
    "CurrencyCoefficient",          # 42
    "PriceMultiplier",              # 43
    "PriceMultiplierDecimals",      # 44
    "Venue",                        # 45 (MIC or BIC)
    "StartTimeVwap",                # 46 (optional)
    "EndTimeVwap",                  # 47 (optional)
    "MiFIDEmissionAllowanceType",   # 48 (optional)
    "MarketOfReferenceMIC",         # 49 (optional)
]

_BRACKET_RE = re.compile(r"\[[^\]]*\]")  # match one [...]


def _to_int(x: str) -> Optional[int]:
    x = x.strip()
    if x == "" or x == "-" or x is None:
        return None
    # Some fields might look like "1.0"
    return int(float(x))


def _to_float(x: str) -> Optional[float]:
    x = x.strip()
    if x == "" or x == "-" or x is None:
        return None
    return float(x)


# ============================================================
# Tick size (ESMA MiFID II tick size regime)
# ============================================================
_TICK_SIZE_TABLE = [
    (0.0, 0.1,   [0.0005, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001]),
    (0.1, 0.2,   [0.001,  0.0005, 0.0002, 0.0001, 0.0001, 0.0001]),
    (0.2, 0.5,   [0.002,  0.001,  0.0005, 0.0002, 0.0001, 0.0001]),
    (0.5, 1.0,   [0.005,  0.002,  0.001,  0.0005, 0.0002, 0.0001]),
    (1.0, 2.0,   [0.01,   0.005,  0.002,  0.001,  0.0005, 0.0002]),
    (2.0, 5.0,   [0.02,   0.01,   0.005,  0.002,  0.001,  0.0005]),
    (5.0, 10.0,  [0.05,   0.02,   0.01,   0.005,  0.002,  0.001]),
    (10.0, 20.0, [0.1,    0.05,   0.02,   0.01,   0.005,  0.002]),
    (20.0, 50.0, [0.2,    0.1,    0.05,   0.02,   0.01,   0.005]),
    (50.0, 100.0,[0.5,    0.2,    0.1,    0.05,   0.02,   0.01]),
    (100.0, 200.0,[1.0,   0.5,    0.2,    0.1,    0.05,   0.02]),
    (200.0, 500.0,[2.0,   1.0,    0.5,    0.2,    0.1,    0.05]),
    (500.0, 1000.0,[5.0,  2.0,    1.0,    0.5,    0.2,    0.1]),
    (1000.0, 2000.0,[10.0, 5.0,   2.0,    1.0,    0.5,    0.2]),
    (2000.0, 5000.0,[20.0, 10.0,  5.0,    2.0,    1.0,    0.5]),
    (5000.0, 10000.0,[50.0, 20.0, 10.0,   5.0,    2.0,    1.0]),
    (10000.0, 20000.0,[100.0, 50.0, 20.0, 10.0,   5.0,    2.0]),
    (20000.0, 50000.0,[200.0, 100.0, 50.0, 20.0, 10.0,    5.0]),
    (50000.0, float("inf"), [500.0, 200.0, 100.0, 50.0, 20.0, 10.0]),
]


def tick_size_from_price(price: float, liquidity_band: int) -> float:
    if price is None or np.isnan(price):
        return np.nan
    if liquidity_band < 1 or liquidity_band > 6:
        raise ValueError("liquidity_band must be in 1..6")
    lb_idx = liquidity_band - 1
    for low, high, ticks in _TICK_SIZE_TABLE:
        if low <= price < high:
            return ticks[lb_idx]
    return np.nan


# ============================================================
# 2) Parse Trade Qualifier & infer signed volume direction
#    PDF: 8-bit meaning (as used here):
#      q3 passive buy
#      q4 aggressive buy
#    We use:
#      (q4 == 1) => buyer initiated => +qty
#      (q3 == 1) => seller initiated => -qty
# ============================================================
_BRACE_RE = re.compile(r"\{([^}]*)\}")


def parse_trade_qualifier_8bits(seq: str) -> Dict[str, Optional[int]]:
    """
    seq example: [1,{1,1,0,0,0,0,0,0}]
    returns q1..q8
    """
    qualifier_keys = [
        "q_UncrossingTrade",
        "q_FirstTradePrice",
        "q_PassiveBuyOrder",
        "q_AggressiveBuyOrder",
        "q_MarketOperations",
        "q_NAVTrade_bps",
        "q_NAVTrade_currency",
        "q_DeferredPublication",
    ]
    out = {key: None for key in qualifier_keys}
    if not seq:
        return out
    m = _BRACE_RE.search(seq)
    if not m:
        return out
    parts = [p.strip() for p in m.group(1).split(",")]
    parts += [""] * (8 - len(parts))
    parts = parts[:8]
    for i in range(8):
        out[qualifier_keys[i]] = _to_int(parts[i]) if parts[i] != "" else None
    return out


def infer_sign_from_qualifier(q: Dict[str, Optional[int]]) -> int:
    # aggressive buy => buyer initiated => +
    if q.get("q_AggressiveBuyOrder") == 1:
        return +1
    # passive buy => buy side passive => seller initiated => -
    if q.get("q_PassiveBuyOrder") == 1:
        return -1
    return 0  # cannot infer (auction/special trades/etc.)


# ============================================================
# 3) Decoder: decode FullTradeInformation using the PDF schema
# ============================================================
def decode_full_trade_information(path: str, tz_local: str = "Europe/Paris") -> pd.DataFrame:
    rows: List[Dict] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue

            # Extract TradeQualifier sequence first to avoid commas inside breaking split(",")
            m = _BRACKET_RE.search(line)
            seq = m.group(0) if m else ""
            safe = line
            if m:
                safe = line[:m.start()] + "@@SEQ@@" + line[m.end():]

            parts = safe.split(",")

            # We need at least up to Venue (index 45)
            if len(parts) < 46:
                continue

            # Restore sequence field
            parts = [seq if p == "@@SEQ@@" else p for p in parts]

            # Truncate/align to schema length (ignore trailing empty columns)
            parts = parts[:len(FIELDS)]

            rec = {FIELDS[i]: parts[i] if i < len(parts) else "" for i in range(len(FIELDS))}
            if rec["MessageType"] != "FullTradeInformation":
                continue

            # Convert key fields
            rec["MarketDataSequenceNumber"] = _to_int(rec["MarketDataSequenceNumber"])
            rec["RebroadcastIndicator"] = _to_int(rec["RebroadcastIndicator"])
            rec["EMM"] = _to_int(rec["EMM"])
            rec["EventTimeNs"] = _to_int(rec["EventTimeNs"])
            rec["SymbolIndex"] = _to_int(rec["SymbolIndex"])
            rec["TradeType"] = _to_int(rec["TradeType"])
            rec["MiFIDPrice"] = _to_float(rec["MiFIDPrice"])
            rec["MiFIDQuantity"] = _to_float(rec["MiFIDQuantity"])
            rec["TransparencyIndicator"] = _to_int(rec["TransparencyIndicator"])
            rec["OriginalReportTimestampNs"] = _to_int(rec["OriginalReportTimestampNs"])
            rec["PriceMultiplier"] = _to_float(rec["PriceMultiplier"])
            rec["PriceMultiplierDecimals"] = _to_int(rec["PriceMultiplierDecimals"])
            rec["CurrencyCoefficient"] = _to_float(rec["CurrencyCoefficient"])

            # qualifier -> sign -> signed qty
            q = parse_trade_qualifier_8bits(rec.get("TradeQualifierSeq", ""))
            sign = infer_sign_from_qualifier(q)
            qty = rec["MiFIDQuantity"] if rec["MiFIDQuantity"] is not None else 0.0
            rec.update(q)
            rec["sign"] = sign
            rec["signed_qty"] = float(sign) * float(qty)

            rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # TradingDateTimeUtc is ISO UTC string
    dt_utc = pd.to_datetime(df["TradingDateTimeUtc"], utc=True, errors="coerce")
    df["trade_time_paris"] = dt_utc.dt.tz_convert(tz_local)

    # Filter out data after 17:30
    df = df[df["trade_time_paris"].dt.time <= pd.to_datetime("17:30").time()]

    df = df.sort_values("trade_time_paris").reset_index(drop=True)
    return df


# ============================================================
# 4) Aggregation A: time bars (Paris local time)
# ============================================================
def aggregate_by_timebar(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    d = df.dropna(subset=["trade_time_paris"]).copy()
    d = d.set_index("trade_time_paris")

    # Calculate imbalance
    sum_signed_qty = d["signed_qty"].resample(freq).sum()
    sum_abs_signed_qty = d["signed_qty"].abs().resample(freq).sum()
    imbalance = (sum_signed_qty / sum_abs_signed_qty).rename("imbalance")

    out = pd.DataFrame({
        "x_t_signed_volume": sum_signed_qty,
        "total_qty": d["MiFIDQuantity"].resample(freq).sum(),
        "known_qty": d.loc[d["sign"] != 0, "MiFIDQuantity"].resample(freq).sum(),
    })
    out = pd.concat([out, imbalance], axis=1)
    out["known_ratio"] = np.where(out["total_qty"] > 0, out["known_qty"] / out["total_qty"], np.nan)
    return out


# ============================================================
# 5) Aggregation B: trade-count bars with N trades per block
#    Matches the paper: x_t = sum_{j=1..N} v_{N(t-1)+j}
# ============================================================
def aggregate_by_trade_count(df: pd.DataFrame, N: int) -> pd.DataFrame:
    d = df.dropna(subset=["trade_time_paris"]).copy().reset_index(drop=True)

    m = len(d)
    T = m // N  # floor(M/N)
    if T <= 0:
        raise ValueError(f"Not enough trades (M={m}) for N={N}")

    # Keep only the first T*N trades to match the definition exactly
    d = d.iloc[: T * N].copy()
    d["block_id"] = np.arange(len(d)) // N

    agg = d.groupby("block_id").agg(
        x_t_signed_volume=("signed_qty", "sum"),
        total_qty=("MiFIDQuantity", "sum"),
        known_qty=("MiFIDQuantity", lambda s: s[d.loc[s.index, "sign"] != 0].sum()),
        start_time=("trade_time_paris", "first"),
        end_time=("trade_time_paris", "last"),
        last_price=("MiFIDPrice", "last"),
    )
    agg["known_ratio"] = np.where(agg["total_qty"] > 0, agg["known_qty"] / agg["total_qty"], np.nan)
    return agg.reset_index()


# ============================================================
# 6) Plotting
# ============================================================
def plot_timebar_curve(agg: pd.DataFrame, out_path: Optional[str] = None):
    plt.figure(figsize=(12, 4))
    plt.plot(agg.index, agg["x_t_signed_volume"])
    plt.grid(alpha=0.3)
    plt.xlabel("Time (Europe/Paris)")
    plt.ylabel("Aggregated signed volume (3-min sum)")
    plt.title("Sanofi — 3-minute aggregated signed order flow")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=160)
    plt.show()


def plot_countbar_curve(agg: pd.DataFrame, out_path: Optional[str] = None, x_axis: str = "block_id"):
    plt.figure(figsize=(12, 4))
    if x_axis == "end_time":
        x = agg["end_time"]
        xlabel = "End time of block (Europe/Paris)"
    else:
        x = agg["block_id"]
        xlabel = "Block index (each block has N trades)"

    plt.plot(x, agg["x_t_signed_volume"])
    plt.grid(alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel("Aggregated signed volume (sum over N trades)")
    plt.title("Sanofi — trade-count aggregated signed order flow")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=160)
    plt.show()


def compute_tick_jumps_continuous(
    df: pd.DataFrame,
    liquidity_band: int = 6,
    trading_mode_continuous: str = "2",
) -> pd.Series:
    """
    Compute transaction price jumps (in ticks) for continuous trading only.
    Returns a Series of integer tick jumps.
    """
    d = df.dropna(subset=["MiFIDPrice"]).copy()
    d = d.sort_values("trade_time_paris").reset_index(drop=True)

    mode = d["MMTTradingMode"].astype(str).str.strip()
    d = d[mode == trading_mode_continuous].copy()
    if d.empty:
        return pd.Series([], dtype=int)

    prices = d["MiFIDPrice"].astype(float)
    diffs = prices.diff()
    prev_price = prices.shift(1)
    tick_sizes = prev_price.apply(lambda p: tick_size_from_price(p, liquidity_band))

    tick_jumps = diffs / tick_sizes
    tick_jumps = tick_jumps.dropna()
    tick_jumps = np.rint(tick_jumps).astype(int)
    return tick_jumps


def plot_tick_jump_hist(
    tick_jumps: pd.Series,
    out_path: Optional[str] = None,
    max_abs_ticks: Optional[int] = 50,
):
    if tick_jumps.empty:
        print("No tick jumps to plot (continuous trading only).")
        return

    if max_abs_ticks is not None:
        tick_jumps = tick_jumps[(tick_jumps >= -max_abs_ticks) & (tick_jumps <= max_abs_ticks)]

    min_tick = int(tick_jumps.min())
    max_tick = int(tick_jumps.max())
    bins = np.arange(min_tick - 0.5, max_tick + 1.5, 1)

    plt.figure(figsize=(12, 5))
    n, bins, patches = plt.hist(tick_jumps, bins=bins, edgecolor="black", alpha=0.7)
    
    # Set x-axis ticks to integers
    tick_min = int(np.floor(tick_jumps.min()))
    tick_max = int(np.ceil(tick_jumps.max()))
    plt.xticks(np.arange(tick_min, tick_max + 1, 1))

    plt.grid(axis="y", alpha=0.3)
    plt.xlabel("Transaction price jump (ticks, continuous trading only)")
    plt.ylabel("Count")
    plt.title("Sanofi transaction price jumps (tick units)")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=160)
    plt.show()


# ============================================================
# 7) Main
# ============================================================
 
if __name__ == "__main__":
    df = decode_full_trade_information(INPUT_TRADE_CSV, tz_local=TZ_LOCAL)

    df[[
        "trade_time_paris", "MiFIDPrice", "MiFIDQuantity",
        "q_UncrossingTrade", "q_FirstTradePrice", "q_PassiveBuyOrder", "q_AggressiveBuyOrder", "sign", "signed_qty",
        "TradeType", "MMTMarketMechanism", "MMTTradingMode", "Venue"
    ]].to_csv("decoded_full_trade_information.csv", index=False)
    print("saved: decoded_full_trade_information.csv")

    # A) 3-minute aggregation
    agg3 = aggregate_by_timebar(df, freq="3min")
    plot_timebar_curve(agg3, out_path=OUT_FIG_TIMEBAR)
    print("saved:", OUT_FIG_TIMEBAR)

    # B) 1-minute aggregation and export
    agg1 = aggregate_by_timebar(df, freq="1min")
    out_csv_1min = "sanofi_1min_aggregation.csv"
    agg1.to_csv(out_csv_1min)
    print("saved:", out_csv_1min)

    # C) Transaction price jump histogram (continuous trading only)
    tick_jumps = compute_tick_jumps_continuous(
        df,
        liquidity_band=LIQUIDITY_BAND,
        trading_mode_continuous="2",
    )
    tick_jumps.to_csv("sanofi_tick_jumps_continuous.csv", index=False)
    print("saved: sanofi_tick_jumps_continuous.csv")
    plot_tick_jump_hist(tick_jumps, out_path=OUT_FIG_TICK_HIST, max_abs_ticks=50)
    print("saved:", OUT_FIG_TICK_HIST)
