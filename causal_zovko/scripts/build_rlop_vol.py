"""
Build RLOP and volatility event/bucket datasets for the causal_zovko pipeline.

External inputs (relative to EA_recherche root):
- euronextparis/EuronextParis/EuronextParis_20191001_FR0000120578/FR0000120578/OrderUpdate_20191001_FR0000120578.csv
- msc_decoded/merged_intervals.csv

Main outputs (under causal_zovko/data):
- rlop_events.csv, vol_events.csv
- rlop_{bid|ask}_{1min|2min|5min|10min}.csv
- vol_{bid|ask}_{1min|2min|5min|10min}.csv
"""

import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# --------- Paths ---------
SCRIPT_DIR = Path(__file__).resolve().parent
CAUSAL_ZOVKO_DIR = SCRIPT_DIR.parent
EA_RECHERCHE_ROOT = CAUSAL_ZOVKO_DIR.parent

DEFAULT_ORDERUPDATE = EA_RECHERCHE_ROOT / "euronextparis" / "EuronextParis" / "EuronextParis_20191001_FR0000120578" / "FR0000120578" / "OrderUpdate_20191001_FR0000120578.csv"
DEFAULT_INTERVALS = EA_RECHERCHE_ROOT / "msc_decoded" / "merged_intervals.csv"

TZ_LOCAL = "Europe/Paris"

# --------- Parsers (OrderUpdate) ---------
import re
from dataclasses import dataclass
from typing import Optional, Iterable, Dict

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

# --------- LOB State ---------
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

    def apply(self, u: Update):
        a = u.action

        # retransmission burst
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

# --------- Continuous Session ---------

def load_continuous_session(intervals_csv: str):
    df = pd.read_csv(intervals_csv)
    continuous = df[df["book_state"].str.contains("Continuous", na=False)]
    if continuous.empty:
        return None
    row = continuous.loc[continuous["duration_s"].idxmax()]
    return int(row["start_ns"]), int(row["end_ns"])

# --------- Feature Extraction ---------

def extract_rlop_and_vol(orderupdate_csv: str, intervals_csv: str | None = None):
    book = OrderBook()

    cont_start, cont_end = None, None
    if intervals_csv and Path(intervals_csv).exists():
        sess = load_continuous_session(intervals_csv)
        if sess is not None:
            cont_start, cont_end = sess

    rlop_rows = []
    vol_rows = []

    last_bid = None
    last_ask = None

    for u in iter_orderupdate_file(orderupdate_csv):
        t = u.event_time
        if cont_start is not None and (t < cont_start or t > cont_end):
            # still need to update book? in practice we skip to keep continuous only
            continue

        # take snapshot BEFORE applying update for RLOP of new limit orders
        if u.action == 1 and u.order_type == 2 and u.price is not None and u.side in (1, 2):
            bid, ask = book.best_bid_ask()
            if bid is not None and ask is not None:
                if u.side == 1:
                    delta = bid - float(u.price)
                    if delta > 0:
                        rlop_rows.append({"t_ns": t, "side": 1, "delta": delta, "bid": bid, "ask": ask, "price": float(u.price)})
                elif u.side == 2:
                    delta = float(u.price) - ask
                    if delta > 0:
                        rlop_rows.append({"t_ns": t, "side": 2, "delta": delta, "bid": bid, "ask": ask, "price": float(u.price)})

        # apply update
        book.apply(u)

        # volatility from best prices after update
        bid, ask = book.best_bid_ask()
        if bid is not None:
            if last_bid is not None and bid > 0 and last_bid > 0:
                vol_rows.append({"t_ns": t, "side": 1, "vol": abs(np.log(bid / last_bid)), "best": bid})
            last_bid = bid
        if ask is not None:
            if last_ask is not None and ask > 0 and last_ask > 0:
                vol_rows.append({"t_ns": t, "side": 2, "vol": abs(np.log(ask / last_ask)), "best": ask})
            last_ask = ask

    rlop = pd.DataFrame(rlop_rows)
    vol = pd.DataFrame(vol_rows)

    for df in (rlop, vol):
        if not df.empty:
            df["time_paris"] = pd.to_datetime(df["t_ns"], unit="ns", utc=True).dt.tz_convert(TZ_LOCAL)

    return rlop, vol


def bucket_time(df: pd.DataFrame, freq: str, value_col: str):
    d = df.dropna(subset=["time_paris"]).copy()
    d = d.set_index("time_paris")
    out = d[value_col].resample(freq).mean().to_frame(name=value_col + "_mean")
    out["count"] = d[value_col].resample(freq).count()
    return out


def bucket_events(df: pd.DataFrame, value_col: str, events_per_bucket: int):
    d = df.sort_values("t_ns").reset_index(drop=True)
    if d.empty:
        return pd.DataFrame()
    d["bucket_id"] = np.arange(len(d)) // events_per_bucket
    out = d.groupby("bucket_id")[value_col].mean().to_frame(name=value_col + "_mean")
    out["count"] = d.groupby("bucket_id")[value_col].count()
    return out


# --------- Main ---------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orderupdate", default=str(DEFAULT_ORDERUPDATE))
    ap.add_argument("--intervals", default=str(DEFAULT_INTERVALS))
    ap.add_argument("--out_dir", default=str(CAUSAL_ZOVKO_DIR / "data"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rlop, vol = extract_rlop_and_vol(args.orderupdate, args.intervals)

    rlop_path = out_dir / "rlop_events.csv"
    vol_path = out_dir / "vol_events.csv"
    rlop.to_csv(rlop_path, index=False)
    vol.to_csv(vol_path, index=False)

    # time buckets
    for freq in ["10min", "5min", "2min", "1min"]:
        if not rlop.empty:
            rlop_bid = rlop[rlop["side"] == 1]
            rlop_ask = rlop[rlop["side"] == 2]
            if not rlop_bid.empty:
                bucket_time(rlop_bid, freq, "delta").to_csv(out_dir / f"rlop_bid_{freq}.csv")
            if not rlop_ask.empty:
                bucket_time(rlop_ask, freq, "delta").to_csv(out_dir / f"rlop_ask_{freq}.csv")

        if not vol.empty:
            vol_bid = vol[vol["side"] == 1]
            vol_ask = vol[vol["side"] == 2]
            if not vol_bid.empty:
                bucket_time(vol_bid, freq, "vol").to_csv(out_dir / f"vol_bid_{freq}.csv")
            if not vol_ask.empty:
                bucket_time(vol_ask, freq, "vol").to_csv(out_dir / f"vol_ask_{freq}.csv")

    # # event buckets
    # if not rlop.empty:
    #     bucket_events(rlop, "delta", 60).to_csv(out_dir / "rlop_60events.csv")
    # if not vol.empty:
    #     bucket_events(vol, "vol", 60).to_csv(out_dir / "vol_60events.csv")

    print("saved:", rlop_path, vol_path)


if __name__ == "__main__":
    main()
