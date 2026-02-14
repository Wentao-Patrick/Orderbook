"""
Build multivariate bucket-level features for causal discovery.

External inputs (relative to EA_recherche root):
- euronextparis/EuronextParis/EuronextParis_20191001_FR0000120578/FR0000120578/OrderUpdate_20191001_FR0000120578.csv
- msc_decoded/merged_intervals.csv
- decoded_full_trade_information.csv

Internal inputs/outputs (under causal_zovko/data):
- input: rlop_events.csv, vol_events.csv
- output: features_{freq}.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import re
from dataclasses import dataclass
from typing import Optional, Iterable, Dict

TZ_LOCAL = "Europe/Paris"
SCRIPT_DIR = Path(__file__).resolve().parent
CAUSAL_ZOVKO_DIR = SCRIPT_DIR.parent
EA_RECHERCHE_ROOT = CAUSAL_ZOVKO_DIR.parent

DEFAULT_ORDERUPDATE = EA_RECHERCHE_ROOT / "euronextparis" / "EuronextParis" / "EuronextParis_20191001_FR0000120578" / "FR0000120578" / "OrderUpdate_20191001_FR0000120578.csv"
DEFAULT_INTERVALS = EA_RECHERCHE_ROOT / "msc_decoded" / "merged_intervals.csv"
DEFAULT_DECODED_TRADES = EA_RECHERCHE_ROOT / "decoded_full_trade_information.csv"
DEFAULT_FREQS = ["1min", "2min"]

# ---- OrderUpdate parser ----
_LINE_RE = re.compile(r"^([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),(\[.*\])\s*$")
_SEQ_RE  = re.compile(r"^\[(\d+),(.*)\]$")
_BRACE_RE = re.compile(r"\{([^}]*)\}")

@dataclass
class Update:
    event_time: int
    rebroadcast: int
    emm: int
    symbol_index: int
    action: int
    priority: Optional[int]
    prev_priority: Optional[int]
    order_type: Optional[int]
    price: Optional[float]
    side: Optional[int]
    qty: Optional[float]
    peg_offset: Optional[float]


def _to_int(x: str) -> Optional[int]:
    x = x.strip()
    if x == "" or x == "-":
        return None
    return int(float(x))


def _to_float(x: str) -> Optional[float]:
    x = x.strip()
    if x == "" or x == "-":
        return None
    return float(x)


def iter_orderupdate_file(path: str) -> Iterable[Update]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = _LINE_RE.match(line)
            if not m:
                continue
            md_seq, msg_type, reb, emm, event_time, seq = m.groups()
            if msg_type != "OrderUpdate":
                continue
            reb = int(reb)
            emm = int(emm)
            event_time = int(event_time)
            sm = _SEQ_RE.match(seq)
            if not sm:
                continue
            body = sm.group(2)
            groups = _BRACE_RE.findall(body)
            for g in groups:
                fields = g.split(",")
                if len(fields) < 9:
                    fields = fields + [""] * (9 - len(fields))
                fields = fields[:9]
                symbol_index = _to_int(fields[0])
                action       = _to_int(fields[1])
                priority     = _to_int(fields[2])
                prev_priority= _to_int(fields[3])
                order_type   = _to_int(fields[4])
                price        = _to_float(fields[5])
                side         = _to_int(fields[6])
                qty          = _to_float(fields[7])
                peg_offset   = _to_float(fields[8])
                if symbol_index is None or action is None:
                    continue
                yield Update(
                    event_time=event_time,
                    rebroadcast=reb,
                    emm=emm,
                    symbol_index=symbol_index,
                    action=action,
                    priority=priority,
                    prev_priority=prev_priority,
                    order_type=order_type,
                    price=price,
                    side=side,
                    qty=qty,
                    peg_offset=peg_offset,
                )

# ---- OrderBook ----
@dataclass
class Order:
    side: int
    price: float
    qty: float
    order_type: Optional[int]

class OrderBook:
    def __init__(self):
        self.orders: Dict[int, Order] = {}
        self._in_retx = False

    def clear(self, side: Optional[int] = None) -> None:
        if side is None:
            self.orders.clear()
        else:
            to_del = [pid for pid, o in self.orders.items() if o.side == side]
            for pid in to_del:
                del self.orders[pid]

    def best_bid_ask(self):
        bid = None
        ask = None
        for o in self.orders.values():
            if o.qty <= 0 or o.price is None or o.price <= 0:
                continue
            if o.side == 1:
                bid = o.price if bid is None else max(bid, o.price)
            elif o.side == 2:
                ask = o.price if ask is None else min(ask, o.price)
        return bid, ask

    def depth_at_best(self):
        bid, ask = self.best_bid_ask()
        bid_qty = 0.0
        ask_qty = 0.0
        if bid is None or ask is None:
            return bid, ask, np.nan, np.nan
        for o in self.orders.values():
            if o.qty <= 0 or o.price is None or o.price <= 0:
                continue
            if o.side == 1 and o.price == bid:
                bid_qty += o.qty
            elif o.side == 2 and o.price == ask:
                ask_qty += o.qty
        return bid, ask, bid_qty, ask_qty

    def apply(self, u: Update):
        a = u.action
        if a == 5 and not self._in_retx:
            self.clear(side=None)
            self._in_retx = True
        if a != 5 and self._in_retx:
            self._in_retx = False

        if a == 1 or a == 5:
            if u.priority is None or u.side is None or u.qty is None:
                return
            if u.price is None or u.price <= 0:
                return
            self.orders[int(u.priority)] = Order(int(u.side), float(u.price), float(u.qty), u.order_type)
        elif a == 2:
            if u.prev_priority is None:
                return
            self.orders.pop(int(u.prev_priority), None)
        elif a == 3:
            if u.side is None:
                self.clear(side=None)
            else:
                self.clear(side=int(u.side))
        elif a == 4:
            if u.priority is None:
                return
            pid = int(u.priority)
            if u.price is None or u.price <= 0 or u.side is None or u.qty is None:
                self.orders.pop(pid, None)
                return
            self.orders[pid] = Order(int(u.side), float(u.price), float(u.qty), u.order_type)
        elif a == 6:
            if u.prev_priority is not None:
                self.orders.pop(int(u.prev_priority), None)
            if u.priority is None or u.side is None or u.qty is None:
                return
            if u.price is None or u.price <= 0:
                return
            self.orders[int(u.priority)] = Order(int(u.side), float(u.price), float(u.qty), u.order_type)

# ---- intervals ----

def load_continuous_session(intervals_csv: str):
    df = pd.read_csv(intervals_csv)
    continuous = df[df["book_state"].str.contains("Continuous", na=False)]
    if continuous.empty:
        return None
    row = continuous.loc[continuous["duration_s"].idxmax()]
    return int(row["start_ns"]), int(row["end_ns"])

# ---- feature build ----

def build_features(orderupdate_csv: str, intervals_csv: str, rlop_path: str, vol_path: str, decoded_trades: str, freq: str, out_path: str):
    freq = re.sub(r"\s+", "", freq).lower()
    
    def to_bucket(ts):
        return ts.dt.floor(freq)

    # 1. RLOP / vol buckets, separated by side
    rlop = pd.read_csv(rlop_path, parse_dates=["time_paris"])
    vol = pd.read_csv(vol_path, parse_dates=["time_paris"])

    rlop["bucket"] = to_bucket(rlop["time_paris"])
    vol["bucket"] = to_bucket(vol["time_paris"])

    # --- RLOP Aggregation ---
    rlop_bid = rlop[rlop["side"] == 1]
    rlop_ask = rlop[rlop["side"] == 2]
    
    rlop_bid_agg = rlop_bid.groupby("bucket").agg(
        rlop_bid_mean=("delta", "mean"),
        rlop_bid_p90=("delta", lambda s: s.quantile(0.9)),
        rlop_bid_count=("delta", "count"),
    )
    rlop_ask_agg = rlop_ask.groupby("bucket").agg(
        rlop_ask_mean=("delta", "mean"),
        rlop_ask_p90=("delta", lambda s: s.quantile(0.9)),
        rlop_ask_count=("delta", "count"),
    )

    # --- Volatility Aggregation ---
    vol_bid = vol[vol["side"] == 1]
    vol_ask = vol[vol["side"] == 2]

    vol_bid_agg = vol_bid.groupby("bucket").agg(
        vol_bid_mean=("vol", "mean"),
        vol_bid_p90=("vol", lambda s: s.quantile(0.9)),
        vol_bid_count=("vol", "count"),
    )
    vol_ask_agg = vol_ask.groupby("bucket").agg(
        vol_ask_mean=("vol", "mean"),
        vol_ask_p90=("vol", lambda s: s.quantile(0.9)),
        vol_ask_count=("vol", "count"),
    )

    # 2. Trades Aggregation
    trades = pd.read_csv(decoded_trades, parse_dates=["trade_time_paris"])
    trades["bucket"] = to_bucket(trades["trade_time_paris"])
    trades_agg = trades.groupby("bucket").agg(
        trade_count=("MiFIDPrice", "count"),
        signed_volume=("signed_qty", "sum"),
        abs_signed_volume=("signed_qty", lambda s: np.abs(s).sum()),
        imb_of=("signed_qty", lambda s: s.sum() / np.abs(s).sum() if np.abs(s).sum() > 0 else np.nan),
    )

    # 3. OrderUpdate-derived microstructure features
    sess = load_continuous_session(intervals_csv)
    cont_start, cont_end = sess if sess else (None, None)

    book = OrderBook()
    rows = []
    for u in iter_orderupdate_file(orderupdate_csv):
        t = u.event_time
        if cont_start is not None and (t < cont_start or t > cont_end):
            continue

        time_paris = pd.to_datetime(t, unit="ns", utc=True).tz_convert(TZ_LOCAL)
        bucket = time_paris.floor(freq)

        # Snapshot BEFORE applying the update
        bid, ask, bid_qty, ask_qty = book.depth_at_best()
        spread = (ask - bid) if (bid is not None and ask is not None) else np.nan
        mid = (bid + ask) / 2 if (bid is not None and ask is not None) else np.nan
        imb_ob = (bid_qty - ask_qty) / (bid_qty + ask_qty) if (bid_qty and ask_qty and (bid_qty + ask_qty) > 0) else np.nan

        # Classify event by side
        limit_new_bid = 0
        limit_new_ask = 0
        if u.action == 1 and u.order_type == 2 and u.price is not None and u.price > 0:
            if u.side == 1:
                limit_new_bid = 1
            elif u.side == 2:
                limit_new_ask = 1

        cancel_bid = 0
        cancel_ask = 0
        cancel_bid_qty = 0.0
        cancel_ask_qty = 0.0
        
        # Get side of cancelled order
        cancelled_order_side = None
        if u.action in (2, 6) and u.prev_priority is not None:
            old = book.orders.get(int(u.prev_priority))
            if old: cancelled_order_side = old.side
        elif u.action == 4 and u.priority is not None:
            old = book.orders.get(int(u.priority))
            if old and u.qty is not None and old.qty > u.qty:
                cancelled_order_side = old.side
                
        if cancelled_order_side == 1:
            cancel_bid = 1
        elif cancelled_order_side == 2:
            cancel_ask = 1

        rows.append({
            "bucket": bucket,
            "spread": spread,
            "mid": mid,
            "bid_depth": bid_qty,
            "ask_depth": ask_qty,
            "imbalance_ob": imb_ob,
            "limit_new_bid": limit_new_bid,
            "limit_new_ask": limit_new_ask,
            "cancel_bid": cancel_bid,
            "cancel_ask": cancel_ask,
        })

        book.apply(u)

    micro = pd.DataFrame(rows)
    micro_agg = micro.groupby("bucket").agg(
        spread_mean=("spread", "mean"),
        mid_mean=("mid", "mean"),
        bid_depth_mean=("bid_depth", "mean"),
        ask_depth_mean=("ask_depth", "mean"),
        imbalance_ob_mean=("imbalance_ob", "mean"),
        limit_new_bid_count=("limit_new_bid", "sum"),
        limit_new_ask_count=("limit_new_ask", "sum"),
        cancel_bid_count=("cancel_bid", "sum"),
        cancel_ask_count=("cancel_ask", "sum"),
    )

    # 4. Merge all features
    df = pd.concat([
        rlop_bid_agg, rlop_ask_agg,
        vol_bid_agg, vol_ask_agg,
        trades_agg,
        micro_agg
    ], axis=1)
    df = df.sort_index()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    print("saved:", out_path)


def parse_freqs(freqs_arg: str):
    tokens = re.findall(r"\d+\s*[A-Za-z]+", freqs_arg)
    freqs = [re.sub(r"\s+", "", t).lower() for t in tokens]
    seen = set()
    out = []
    for f in freqs:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def resolve_out_path(out_arg: str, freq: str, multi: bool):
    if "{freq}" in out_arg:
        return out_arg.format(freq=freq)

    out_path = Path(out_arg)
    if not multi:
        return str(out_path)

    if out_path.suffix:
        return str(out_path.with_name(f"{out_path.stem}_{freq}{out_path.suffix}"))

    return str(out_path / f"features_{freq}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orderupdate", default=str(DEFAULT_ORDERUPDATE))
    ap.add_argument("--intervals", default=str(DEFAULT_INTERVALS))
    ap.add_argument("--rlop", default=str(CAUSAL_ZOVKO_DIR / "data" / "rlop_events.csv"))
    ap.add_argument("--vol", default=str(CAUSAL_ZOVKO_DIR / "data" / "vol_events.csv"))
    ap.add_argument("--trades", default=str(DEFAULT_DECODED_TRADES))
    ap.add_argument("--freq", default=None, help="single frequency (overrides --freqs)")
    ap.add_argument("--freqs", default="1min,2min,5min,10min")
    ap.add_argument("--out", default=str(CAUSAL_ZOVKO_DIR / "data" / "features_{freq}.csv"))
    args = ap.parse_args()

    if args.freq:
        freqs = [re.sub(r"\s+", "", args.freq).lower()]
    else:
        freqs = parse_freqs(args.freqs) or DEFAULT_FREQS

    multi = len(freqs) > 1
    for freq in freqs:
        out_path = resolve_out_path(args.out, freq, multi)
        build_features(args.orderupdate, args.intervals, args.rlop, args.vol, args.trades, freq, out_path)


if __name__ == "__main__":
    main()
