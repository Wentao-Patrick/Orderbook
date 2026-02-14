# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Iterable, List, Tuple

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TZ_LOCAL = "Europe/Paris"

# =========================
# 0) Paths (EDIT HERE)
# =========================
ORDERUPDATE_CSV = r"C:\Users\Wentao\Desktop\EA_recherche\euronextparis\EuronextParis\EuronextParis_20191001_FR0000120578\FR0000120578\OrderUpdate_20191001_FR0000120578.csv"
TRADE_CSV = r"C:\Users\Wentao\Desktop\EA_recherche\euronextparis\EuronextParis\EuronextParis_20191001_FR0000120578\FR0000120578\FullTradeInformation_20191001_FR0000120578.csv"

OUT_FIG = "trade_alignment_decrement.png"


# =========================
# 1) Parse OrderUpdate (your style)
# =========================
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


# =========================
# 2) Minimal decode FullTradeInformation (robust: only split early columns)
#    Trick: split with maxsplit=14 so qty field is isolated even if later columns contain [...] with commas.
# =========================
@dataclass
class Trade:
    event_time: int  # ns UTC epoch
    price: Optional[float]
    qty: Optional[float]

def iter_trades_minimal(path: str) -> Iterable[Trade]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # We only need up to MiFIDQuantity (index 13).
            # Use maxsplit=14 so parts[13] is ONLY qty, parts[14] is the rest.
            parts = line.split(",", maxsplit=14)
            if len(parts) < 14:
                continue
            msg_type = parts[1]
            if msg_type != "FullTradeInformation":
                continue
            event_time = _to_int(parts[4])
            price = _to_float(parts[12])
            qty = _to_float(parts[13])
            if event_time is None:
                continue
            yield Trade(event_time=int(event_time), price=price, qty=qty)

def load_trades_df(path: str) -> pd.DataFrame:
    rows = []
    for tr in iter_trades_minimal(path):
        rows.append({"t_ns": tr.event_time, "price": tr.price, "qty": tr.qty})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time_paris"] = pd.to_datetime(df["t_ns"], unit="ns", utc=True).dt.tz_convert(TZ_LOCAL)
    df = df.sort_values("t_ns").reset_index(drop=True)
    return df


# =========================
# 3) Build "decrement events" from OrderUpdate (NO matching, just observe book deltas)
# =========================
@dataclass
class RestingOrder:
    side: int
    price: float
    qty: float

@dataclass
class DecrementEvent:
    t_ns: int
    side: int
    price: float
    delta_qty: float
    kind: str       # "delete" / "qty_down"
    action: int
    prio: Optional[int]
    prev: Optional[int]
    order_type: Optional[int]

def build_decrement_events(orderupdate_csv: str, ignore_cross_side: bool = True) -> pd.DataFrame:
    book: Dict[int, RestingOrder] = {}
    in_retx = False
    events: List[DecrementEvent] = []

    def safe_side(u: Update) -> bool:
        if u.side is None:
            return True
        return u.side in (1, 2)

    for u in iter_orderupdate_file(orderupdate_csv):
        a = u.action

        # retransmission blocks: reset book when the burst starts
        if a == 5 and not in_retx:
            book.clear()
            in_retx = True
        if a != 5 and in_retx:
            in_retx = False

        if ignore_cross_side and (u.side is not None) and (u.side not in (1,2)):
            # Cross or other side: ignore
            continue

        # Helper: remove by priority and create a delete event (if we know old)
        def remove_priority(t_ns: int, pid: int, u: Update, action: int, kind: str = "delete"):
            old = book.pop(pid, None)
            if old is None:
                return
            if old.qty > 0 and old.price > 0:
                events.append(DecrementEvent(
                    t_ns=t_ns,
                    side=old.side,
                    price=old.price,
                    delta_qty=float(old.qty),
                    kind=kind,
                    action=action,
                    prio=u.priority,
                    prev=u.prev_priority,
                    order_type=u.order_type,
                ))

        # Apply logic
        if a in (1, 5):  # New insert or retransmission insert
            if u.priority is None or u.side is None or u.qty is None:
                continue
            # if price missing, can't place; skip (it won't be a resting visible LOB level)
            if u.price is None or u.price <= 0:
                continue
            book[int(u.priority)] = RestingOrder(side=int(u.side), price=float(u.price), qty=float(u.qty))

        elif a == 2:  # Delete by Previous Priority
            if u.prev_priority is None:
                continue
            remove_priority(u.event_time, int(u.prev_priority), u, action=2, kind="delete")

        elif a == 3:  # Delete all
            # Hard to attribute deltas reliably; skip logging here, but clear the book
            if u.side is None:
                book.clear()
            else:
                s = int(u.side)
                for pid in [pid for pid,o in book.items() if o.side == s]:
                    book.pop(pid, None)

        elif a == 4:  # Modify without loss of priority
            if u.priority is None:
                continue
            pid = int(u.priority)
            old = book.get(pid)

            # If old exists and new qty < old qty -> decrement event
            # If price missing, we treat as remove (can't keep it on priced book)
            if u.price is None or u.price <= 0 or u.side is None or u.qty is None:
                if old is not None:
                    remove_priority(u.event_time, pid, u, action=4, kind="delete")
                continue

            new_side = int(u.side)
            new_price = float(u.price)
            new_qty = float(u.qty)

            if old is not None:
                if old.qty > new_qty and old.price > 0:
                    events.append(DecrementEvent(
                        t_ns=u.event_time,
                        side=old.side,
                        price=old.price,
                        delta_qty=float(old.qty - new_qty),
                        kind="qty_down",
                        action=4,
                        prio=u.priority,
                        prev=u.prev_priority,
                        order_type=u.order_type,
                    ))
            # update/insert
            book[pid] = RestingOrder(side=new_side, price=new_price, qty=new_qty)

        elif a == 6:  # Modify with loss of priority: delete prev then add new
            if u.prev_priority is not None:
                remove_priority(u.event_time, int(u.prev_priority), u, action=6, kind="delete")

            if u.priority is None or u.side is None or u.qty is None:
                continue
            if u.price is None or u.price <= 0:
                continue
            book[int(u.priority)] = RestingOrder(side=int(u.side), price=float(u.price), qty=float(u.qty))

        else:
            continue

    # to df
    df = pd.DataFrame([e.__dict__ for e in events])
    if df.empty:
        return df
    df = df.sort_values("t_ns").reset_index(drop=True)
    df["time_paris"] = pd.to_datetime(df["t_ns"], unit="ns", utc=True).dt.tz_convert(TZ_LOCAL)
    return df


# =========================
# 4) Alignment analysis
# =========================
def window_sums(event_times: np.ndarray, event_values: np.ndarray, centers: np.ndarray, window_ns: int) -> np.ndarray:
    """
    Sum event_values within [center-window, center+window] for each center.
    event_times must be sorted.
    """
    left = np.searchsorted(event_times, centers - window_ns, side="left")
    right = np.searchsorted(event_times, centers + window_ns, side="right")
    out = np.empty(len(centers), dtype=float)
    for i, (l, r) in enumerate(zip(left, right)):
        out[i] = float(event_values[l:r].sum())
    return out

def pick_random_centers(min_ns: int, max_ns: int, n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(low=min_ns, high=max_ns, size=n, dtype=np.int64)

def verify_alignment(
    trades_df: pd.DataFrame,
    dec_df: pd.DataFrame,
    window_ns: int = 5_000_000,   # 5 ms
    sample_trades: int = 5000,
    seed: int = 42,
):
    """
    Compare decrement mass near trades vs near random times.
    """
    if trades_df.empty:
        raise RuntimeError("Trades DF is empty.")
    if dec_df.empty:
        raise RuntimeError("Decrement-events DF is empty.")

    # sample trades to keep runtime reasonable
    if len(trades_df) > sample_trades:
        trades_s = trades_df.sample(sample_trades, random_state=seed).sort_values("t_ns").reset_index(drop=True)
    else:
        trades_s = trades_df.copy()

    # arrays
    t_trade = trades_s["t_ns"].to_numpy(dtype=np.int64)
    t_event = dec_df["t_ns"].to_numpy(dtype=np.int64)
    dq_event = dec_df["delta_qty"].to_numpy(dtype=float)

    # sums near trade
    sum_trade = window_sums(t_event, dq_event, t_trade, window_ns)

    # baseline: random centers (same count) within full day range of trades
    min_ns = int(trades_df["t_ns"].min())
    max_ns = int(trades_df["t_ns"].max())
    t_rand = pick_random_centers(min_ns, max_ns, len(t_trade), seed=seed)
    sum_rand = window_sums(t_event, dq_event, t_rand, window_ns)

    # summary stats
    def stats(x: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "p90": float(np.quantile(x, 0.90)),
            "p99": float(np.quantile(x, 0.99)),
            "zero_ratio": float(np.mean(x == 0.0)),
        }

    st_trade = stats(sum_trade)
    st_rand = stats(sum_rand)

    print("=== Window:", window_ns/1e6, "ms ===")
    print("[trade-near] ", st_trade)
    print("[random]     ", st_rand)

    # plot
    plt.figure(figsize=(12, 5))

    # Use log1p to compare heavy tails
    plt.hist(np.log1p(sum_rand), bins=80, alpha=0.5, label="Random times (log1p sum decrement)")
    plt.hist(np.log1p(sum_trade), bins=80, alpha=0.5, label="Trade times (log1p sum decrement)")

    plt.grid(alpha=0.3)
    plt.xlabel("log(1 + sum(decrement qty) within window)")
    plt.ylabel("Count")
    plt.title("Decrement activity near trades vs random times (evidence that OrderUpdate reflects executions)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=160)
    plt.show()

    # show a few example trades with local context
    print("\n=== Examples (first 5 sampled trades) ===")
    for i in range(min(5, len(trades_s))):
        t0 = int(trades_s.loc[i, "t_ns"])
        time_str = trades_s.loc[i, "time_paris"]
        qty = trades_s.loc[i, "qty"]
        px = trades_s.loc[i, "price"]
        l = np.searchsorted(t_event, t0 - window_ns, side="left")
        r = np.searchsorted(t_event, t0 + window_ns, side="right")
        ctx = dec_df.iloc[l:r].copy()
        print(f"\nTrade @ {time_str}  price={px} qty={qty}  window_sum={sum_trade[i]:.2f}  events={len(ctx)}")
        print(ctx[["time_paris","kind","action","side","price","delta_qty","prio","prev","order_type"]].head(25).to_string(index=False))

    return sum_trade, sum_rand


# =========================
# 5) Main
# =========================
if __name__ == "__main__":
    print("[1/3] Loading trades...")
    trades = load_trades_df(TRADE_CSV)
    print("Trades:", len(trades), "from", trades["time_paris"].min(), "to", trades["time_paris"].max())

    print("[2/3] Building decrement events from OrderUpdate (one pass)...")
    dec = build_decrement_events(ORDERUPDATE_CSV)
    print("Decrement events:", len(dec), "from", dec["time_paris"].min(), "to", dec["time_paris"].max())

    print("[3/3] Align trades with decrement events...")
    # you can try different windows: 1ms, 5ms, 20ms, 100ms
    verify_alignment(trades, dec, window_ns=5_000_000, sample_trades=5000)
    print("Saved figure:", OUT_FIG)
