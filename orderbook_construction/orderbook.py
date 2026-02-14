from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Iterable
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from tqdm import tqdm
mpl.rcParams["animation.ffmpeg_path"] = r"C:\\Users\\Wentao\\anaconda3\\Library\\bin\\ffmpeg.exe" 
mpl.rcParams["animation.writer"] = "ffmpeg"

# =========================
# 1) Decode EUROFIDAI OrderUpdate lines
# =========================
# Definition（Euronext Paris V2, from 2018）One OrderUpdate message's repeating group field order is: :contentReference[oaicite:3]{index=3}
# { SymbolIndex, ActionType, OrderPriority, PreviousPriority, OrderType, OrderPrice, OrderSide, OrderQuantity, PegOffset }

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
    return None if x == "" else int(x)

def _to_float(x: str) -> Optional[float]:
    x = x.strip()
    return None if x == "" else float(x)

def iter_orderupdate_file(path: str) -> Iterable[Update]:
    """
    Streaming parser:
    yields one Update per {...} group in each CSV line.
    """
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

            no_updates = int(sm.group(1))  # can be used for checks
            body = sm.group(2)

            groups = _BRACE_RE.findall(body)
            for g in groups:
                fields = g.split(",")
                # pad to 9 fields (missing optional fields appear empty)
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
# 2) Order (maintain L3 by priority, then aggregate into L2 depth)
# =========================
@dataclass
class Order:
    side: int        # 1 Buy, 2 Sell 
    price: float
    qty: float
    order_type: Optional[int]  # 1 Market, 2 Limit, 6 Market-to-limit, etc.

class OrderBook:
    def __init__(self, include_market_in_depth: bool = False):
        # key: Order Priority -> Order
        self.orders: Dict[int, Order] = {}
        self.include_market_in_depth = include_market_in_depth
        self.market_bucket_qty_buy = 0.0
        self.market_bucket_qty_sell = 0.0

        # retransmission state
        self._in_retransmission = False

    def clear(self, side: Optional[int] = None) -> None:
        if side is None:
            self.orders.clear()
            self.market_bucket_qty_buy = 0.0
            self.market_bucket_qty_sell = 0.0
        else:
            to_del = [pid for pid, o in self.orders.items() if o.side == side]
            for pid in to_del:
                del self.orders[pid]

    def _is_depth_eligible(self, u: Update) -> bool:
        """
        Decide whether this order should appear in price-level depth.
        - Market orders (OrderType=1) often have empty/0 price, cannot be placed on x=price curve.
        - Pegged types may also have missing price; without best-price feed we can't reprice.
        Policy:
          - If price is present and > 0 -> eligible.
          - Else:
              * if include_market_in_depth=True, we allow price=0 as a special level (not recommended for plotting).
              * otherwise exclude from depth but optionally count in market buckets.
        """
        if u.price is not None and u.price > 0:
            return True
        if self.include_market_in_depth:
            return True
        return False

    def _add_market_bucket(self, u: Update) -> None:
        # Only for accounting (not a real book level)
        if u.qty is None or u.side is None:
            return
        if u.side == 1:
            self.market_bucket_qty_buy += u.qty
        elif u.side == 2:
            self.market_bucket_qty_sell += u.qty

    def apply(self, u: Update) -> None:
        """
        Apply one update according to Market Data Action 
          1 New
          2 Delete by Previous Priority
          3 Delete all (side-filtered)
          4 Modify without loss of priority
          5 Retransmission of all orders
          6 Modify with loss of priority (delete prev, add new)
        """
        a = u.action

        # --- Retransmission handling (Action=5 is a full book retransmission) :contentReference[oaicite:8]{index=8}
        if a == 5 and not self._in_retransmission:
            # first retransmission row -> reset book, then rebuild from retransmitted orders
            self.clear(side=None)
            self._in_retransmission = True
        if a != 5 and self._in_retransmission:
            # retransmission burst ended
            self._in_retransmission = False

        # --- Apply actions
        if a == 1:
            # New order: add if we have a priority
            if u.priority is None:
                return
            if u.side is None or u.qty is None:
                return

            # market vs limit handling
            if not self._is_depth_eligible(u):
                # we do not put it on depth curve; just count it (optional)
                if u.order_type == 1:
                    self._add_market_bucket(u)
                return

            price = 0.0 if (u.price is None) else float(u.price)
            self.orders[int(u.priority)] = Order(int(u.side), price, float(u.qty), u.order_type)

        elif a == 2:
            # Delete by Previous Priority
            if u.prev_priority is None:
                return
            self.orders.pop(int(u.prev_priority), None)

        elif a == 3:
            # Delete all orders for instrument (optionally by side)
            if u.side is None:
                self.clear(side=None)
            else:
                self.clear(side=int(u.side))

        elif a == 4:
            # Modify without loss of priority: same priority updated
            if u.priority is None:
                return
            if u.side is None or u.qty is None:
                return

            if not self._is_depth_eligible(u):
                # if it becomes a market-type with no price, remove it from depth
                self.orders.pop(int(u.priority), None)
                if u.order_type == 1:
                    self._add_market_bucket(u)
                return

            price = 0.0 if (u.price is None) else float(u.price)
            self.orders[int(u.priority)] = Order(int(u.side), price, float(u.qty), u.order_type)

        elif a == 5:
            # Retransmission rows are essentially "New orders" in the reconstructed snapshot
            if u.priority is None:
                return
            if u.side is None or u.qty is None:
                return

            if not self._is_depth_eligible(u):
                if u.order_type == 1:
                    self._add_market_bucket(u)
                return

            price = 0.0 if (u.price is None) else float(u.price)
            self.orders[int(u.priority)] = Order(int(u.side), price, float(u.qty), u.order_type)

        elif a == 6:
            # Modify with loss of priority: delete prev, add new
            if u.prev_priority is not None:
                self.orders.pop(int(u.prev_priority), None)

            if u.priority is None:
                return
            if u.side is None or u.qty is None:
                return

            if not self._is_depth_eligible(u):
                if u.order_type == 1:
                    self._add_market_bucket(u)
                return

            price = 0.0 if (u.price is None) else float(u.price)
            self.orders[int(u.priority)] = Order(int(u.side), price, float(u.qty), u.order_type)

        else:
            # RFQ types etc. ignore by default
            pass

    def depth_cumulative(self, max_levels: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build cumulative depth curves:
          bids: prices desc, cumqty asc
          asks: prices asc, cumqty asc
        """
        bid_levels: Dict[float, float] = {}
        ask_levels: Dict[float, float] = {}

        for o in self.orders.values():
            if o.qty <= 0:
                continue
            # ignore price=0 in plotting unless user explicitly wants it
            if (o.price <= 0) and (not self.include_market_in_depth):
                continue

            if o.side == 1:
                bid_levels[o.price] = bid_levels.get(o.price, 0.0) + o.qty
            elif o.side == 2:
                ask_levels[o.price] = ask_levels.get(o.price, 0.0) + o.qty

        bid_prices = np.array(sorted(bid_levels.keys(), reverse=True), dtype=float)
        ask_prices = np.array(sorted(ask_levels.keys()), dtype=float)

        bid_qty = np.array([bid_levels[p] for p in bid_prices], dtype=float)
        ask_qty = np.array([ask_levels[p] for p in ask_prices], dtype=float)

        bid_cum = np.cumsum(bid_qty)
        ask_cum = np.cumsum(ask_qty)

        if max_levels is not None:
            bid_prices, bid_cum = bid_prices[:max_levels], bid_cum[:max_levels]
            ask_prices, ask_cum = ask_prices[:max_levels], ask_cum[:max_levels]

        return bid_prices, bid_cum, ask_prices, ask_cum

    def calculate_imbalance(self, max_levels: int = 1) -> Optional[float]:
        """
        Calculate order book imbalance based on the top N levels.
        Formula: (V_bid - V_ask) / (V_bid + V_ask)
        where V is the volume at the best price levels.
        Returns a float between -1.0 and 1.0, or None if book is empty.
        """
        bid_p, bid_q, ask_p, ask_q = self.depth_cumulative(max_levels=max_levels)

        # We need the non-cumulative quantity at the best levels
        bid_levels: Dict[float, float] = {}
        ask_levels: Dict[float, float] = {}
        for o in self.orders.values():
            if o.qty > 0 and o.price > 0:
                if o.side == 1:
                    bid_levels[o.price] = bid_levels.get(o.price, 0.0) + o.qty
                elif o.side == 2:
                    ask_levels[o.price] = ask_levels.get(o.price, 0.0) + o.qty
        
        if not bid_levels or not ask_levels:
            return None

        best_bid_price = max(bid_levels.keys())
        best_ask_price = min(ask_levels.keys())

        vol_bid = bid_levels.get(best_bid_price, 0.0)
        vol_ask = ask_levels.get(best_ask_price, 0.0)

        if vol_bid + vol_ask == 0:
            return 0.0
        
        imbalance = (vol_bid - vol_ask) / (vol_bid + vol_ask)
        return imbalance



# =========================
# 3) Replayer：snapshot_at(t) + Animation
# =========================
class OrderBookReplayer:
    def __init__(self, path: str, include_market_in_depth: bool = False):
        self.path = path
        self.book = OrderBook(include_market_in_depth=include_market_in_depth)
        self._it = iter_orderupdate_file(path)
        self._next: Optional[Update] = None
        self._exhausted = False

        # prime first
        self._advance()

    def _advance(self):
        if self._exhausted:
            self._next = None
            return
        try:
            self._next = next(self._it)
        except StopIteration:
            self._exhausted = True
            self._next = None

    def reset(self):
        self.book = OrderBook(include_market_in_depth=self.book.include_market_in_depth)
        self._it = iter_orderupdate_file(self.path)
        self._next = None
        self._exhausted = False
        self._advance()

    def replay_until(self, t_ns: int):
        """
        Apply all updates with event_time <= t_ns.
        """
        t_ns = int(t_ns)
        while self._next is not None and self._next.event_time <= t_ns:
            self.book.apply(self._next)
            self._advance()

    def snapshot_at(self, t_ns: int) -> OrderBook:
        """
        Return the book snapshot at time t_ns (advances internal state).
        """
        self.replay_until(t_ns)
        return self.book

    def plot_depth(self, t_ns: int, ax=None, title: str = "", max_levels: int = 50):
        """
        Replay until t_ns and plot the book snapshot at that time.
        """
        # Replay until the specified timestamp
        self.replay_until(t_ns)

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 4))

        bid_p, bid_c, ask_p, ask_c = self.book.depth_cumulative(max_levels=max_levels)
        ax.clear()

        if len(bid_p) > 0:
            ax.plot(bid_p, bid_c, label="Bids (cum qty)")
        if len(ask_p) > 0:
            ax.plot(ask_p, ask_c, label="Asks (cum qty)")

        # Format the timestamp for the title
        time_str = pd.to_datetime(t_ns, unit="ns", utc=True).tz_convert("Europe/Paris").strftime('%Y-%m-%d %H:%M:%S')
        
        ax.set_xlabel("Price")
        ax.set_ylabel("Cumulative Quantity")
        # Use the formatted time in the title
        ax.set_title(f"{title} | {time_str}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")


    def animate_day_mp4_with_ffmpeg(
        self,
        out_path: str = "sanofi_depth.mp4",
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        step_ns: int = 5_000_000_000,   # 5 seconds per frame
        max_levels: int = 50,
        fps: int = 20,
        dpi: int = 140,
        ffmpeg_path: str = r"C:\Users\Wentao\anaconda3\Library\bin\ffmpeg.exe",
    ):
        """
        Robust MP4 export on Windows:
        1) Render frames as PNG
        2) Call ffmpeg.exe directly to encode MP4
        This bypasses Matplotlib's writer detection entirely.
        """
        import os
        import tempfile
        import subprocess
        from tqdm import tqdm
        import matplotlib.pyplot as plt

        # ---- sanity check ----
        if not os.path.isfile(ffmpeg_path):
            raise FileNotFoundError(f"ffmpeg.exe not found at: {ffmpeg_path}")

        self.reset()

        # infer start/end if needed (stream once to get last timestamp)
        if start_ns is None or end_ns is None:
            first = self._next.event_time if self._next is not None else None
            last = first
            for u in iter_orderupdate_file(self.path):
                last = u.event_time
            if start_ns is None:
                start_ns = first
            if end_ns is None:
                end_ns = last
            self.reset()

        start_ns = int(start_ns)
        end_ns = int(end_ns)
        step_ns = int(step_ns)

        frame_times = list(range(start_ns, end_ns + 1, step_ns))
        if not frame_times:
            raise RuntimeError("No frames (check start/end/step).")

        # create temp directory for frames
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots(figsize=(9, 4))

            # ---- render frames ----
            for i, t in enumerate(tqdm(frame_times, desc="Rendering frames", unit="frame")):
                self.replay_until(t)
                
                # # Format the timestamp for the title
                # time_str = pd.to_datetime(t, unit="ns", utc=True).tz_convert("Europe/Paris").strftime('%Y-%m-%d %H:%M:%S')
                
                self.plot_depth(
                    t_ns=t, # Pass the timestamp
                    ax=ax,
                    title=f"Sanofi (FR0000120578)", # Base title without time
                    max_levels=max_levels,
                )
                frame_path = os.path.join(tmpdir, f"frame_{i:06d}.png")
                fig.savefig(frame_path, dpi=dpi, bbox_inches="tight")

            plt.close(fig)

            # ---- encode mp4 with ffmpeg ----
            # input pattern: frame_000000.png, frame_000001.png, ...
            input_pattern = os.path.join(tmpdir, "frame_%06d.png")

            # ffmpeg command (H.264)
            cmd = [
                ffmpeg_path,
                "-y",
                "-framerate", str(fps),
                "-i", input_pattern,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                out_path,
            ]

            print("[info] Encoding MP4 with ffmpeg...")
            # ffmpeg prints progress to stderr; we just run it
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if proc.returncode != 0:
                # show last part of stderr for debugging
                tail = proc.stderr[-2000:]
                raise RuntimeError("ffmpeg encoding failed. Last stderr:\n" + tail)

            print(f"[done] Saved: {out_path}")

# =========================
# 4) main function
# =========================
import sys, shutil
print("python:", sys.executable)
print("ffmpeg:", shutil.which("ffmpeg"))

def best_bid_ask(book: OrderBook):
    bid = None
    ask = None
    for o in book.orders.values():
        if o.qty <= 0:
            continue
        if o.price <= 0:
            continue
        if o.side == 1:
            bid = o.price if bid is None else max(bid, o.price)
        elif o.side == 2:
            ask = o.price if ask is None else min(ask, o.price)
    return bid, ask

def top_n_levels(book: OrderBook, n: int = 5):
    bid_levels: Dict[float, float] = {}
    ask_levels: Dict[float, float] = {}

    for o in book.orders.values():
        if o.qty <= 0:
            continue
        if o.price is None or o.price <= 0:
            continue
        if o.side == 1:
            bid_levels[o.price] = bid_levels.get(o.price, 0.0) + o.qty
        elif o.side == 2:
            ask_levels[o.price] = ask_levels.get(o.price, 0.0) + o.qty

    bid_prices = sorted(bid_levels.keys(), reverse=True)
    ask_prices = sorted(ask_levels.keys())

    def _pad(prices: List[float], levels: Dict[float, float]):
        out_p: List[float] = []
        out_q: List[float] = []
        for i in range(n):
            if i < len(prices):
                p = prices[i]
                out_p.append(p)
                out_q.append(levels[p])
            else:
                out_p.append(np.nan)
                out_q.append(np.nan)
        return out_p, out_q

    bid_p, bid_q = _pad(bid_prices, bid_levels)
    ask_p, ask_q = _pad(ask_prices, ask_levels)
    return bid_p, bid_q, ask_p, ask_q

def scan_negative_spread(rep: OrderBookReplayer, step_ns: int = 10_000_000_000, max_checks: int = 2000):
    import pandas as pd

    rep.reset()

    # get start/end (same method as animate)
    first = rep._next.event_time if rep._next is not None else None
    last = first
    for u in iter_orderupdate_file(rep.path):
        last = u.event_time
    rep.reset()

    times = list(range(int(first), int(last)+1, int(step_ns)))[:max_checks]
    bad = []

    for t in times:
        rep.replay_until(t)
        bid, ask = best_bid_ask(rep.book)
        if bid is not None and ask is not None and (bid - ask) > 0:
            # spread negative => bid > ask
            # here (bid-ask)>0 is equivalent to spread = ask-bid < 0
            bad.append((t, bid, ask, ask - bid))

    print(f"negative spread frames: {len(bad)}")
    for t, bid, ask, spr in bad[:50]:
        dt = pd.to_datetime(t, unit="ns", utc=True).tz_convert("Europe/Paris")
        print(dt, "bid", bid, "ask", ask, "spread", spr)


def plot_imbalance_over_time(
    rep: OrderBookReplayer,
    intervals_path: str,
    step_ns: int = 10_000_000_000, # 10 seconds
):
    """
    Calculates and plots the order book imbalance over the continuous trading session.
    """
    print("[info] Plotting imbalance over time...")
    # 1. Read intervals to find the continuous trading session
    try:
        intervals_df = pd.read_csv(intervals_path)
        continuous_session = intervals_df[intervals_df["book_state"].str.contains("Continuous", na=False)]
        
        # Find the main continuous session (longest duration)
        continuous_session = continuous_session.loc[continuous_session['duration_s'].idxmax()]
        
        start_ns = int(continuous_session["start_ns"])
        end_ns = int(continuous_session["end_ns"])

        print(f"[info] Found continuous session from {pd.to_datetime(start_ns, unit='ns', utc=True).tz_convert('Europe/Paris')} to {pd.to_datetime(end_ns, unit='ns', utc=True).tz_convert('Europe/Paris')}")

    except (FileNotFoundError, KeyError, IndexError) as e:
        print(f"[error] Could not determine continuous session from {intervals_path}: {e}")
        print("[info] Falling back to full day scan for imbalance.")
        # Fallback to full range if file is not available
        rep.reset()
        start_ns = rep._next.event_time if rep._next is not None else 0
        last = start_ns
        for u in iter_orderupdate_file(rep.path):
            last = u.event_time
        end_ns = last
        rep.reset()

    # 2. Sample points in time and calculate imbalance
    rep.reset()
    sample_times = list(range(start_ns, end_ns, step_ns))
    imbalances = []
    timestamps = []

    from tqdm import tqdm
    for t in tqdm(sample_times, desc="Calculating imbalance", unit="sample"):
        rep.replay_until(t)
        imb = rep.book.calculate_imbalance()
        if imb is not None:
            imbalances.append(imb)
            timestamps.append(pd.to_datetime(t, unit='ns', utc=True).tz_convert('Europe/Paris'))

    # 3. Plot the results
    if not timestamps:
        print("[warning] No imbalance data was generated. Cannot plot.")
        return

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(timestamps, imbalances, label="Order Book Imbalance", color='teal')
    
    ax.set_xlabel("Time (Europe/Paris)")
    ax.set_ylabel("Imbalance")
    ax.set_title("Order Book Imbalance during Continuous Trading")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add a horizontal line at 0 for reference
    ax.axhline(0, color='r', linestyle='--', linewidth=0.8, label='Balanced')
    
    plt.tight_layout()
    plt.show()

def export_orderbook_snapshots(
    rep: OrderBookReplayer,
    intervals_path: str,
    event_out_path: str,
    snapshot_out_path: str,
    snapshot_freq_s: int = 1,
):
    """
    Replays the order book and exports its state.

    1. Exports the state of the book AFTER every single order update event.
    2. Exports a snapshot of the book at a regular time interval (e.g., every 1 second).

    Args:
        rep: The OrderBookReplayer instance.
        intervals_path: Path to the market session intervals file.
        event_out_path: Output path for the event-by-event data (CSV or Parquet).
        snapshot_out_path: Output path for the time-based snapshots (CSV or Parquet).
        snapshot_freq_s: The frequency in seconds for taking snapshots.
    """
    print("[info] Starting order book state export...")
    # 1. Find continuous trading session
    try:
        intervals_df = pd.read_csv(intervals_path)
        session = intervals_df[intervals_df["book_state"].str.contains("Continuous", na=False)]
        session = session.loc[session['duration_s'].idxmax()]
        start_ns = int(session["start_ns"])
        end_ns = int(session["end_ns"])
        print(f"[info] Continuous session: {pd.to_datetime(start_ns, unit='ns', utc=True).tz_convert('Europe/Paris')} to {pd.to_datetime(end_ns, unit='ns', utc=True).tz_convert('Europe/Paris')}")
    except Exception as e:
        print(f"[error] Could not determine continuous session from {intervals_path}: {e}")
        return

    # 2. Replay and collect data
    rep.reset()
    event_rows = []
    snapshot_rows = []
    
    # For tqdm progress bar, we need the total number of updates
    print("[info] Counting total updates for progress bar...")
    total_updates = sum(1 for _ in iter_orderupdate_file(rep.path))
    rep.reset()

    # Find first event time to align snapshots
    if rep._next is None:
        print("[warning] No updates found in the file.")
        return
    
    # Align snapshot times to the start of the session
    snapshot_interval_ns = snapshot_freq_s * 1_000_000_000
    next_snapshot_time_ns = start_ns

    with tqdm(total=total_updates, desc="Exporting book states", unit="update") as pbar:
        while rep._next is not None:
            update = rep._next
            t_ns = update.event_time
            
            # Advance replayer by one step
            rep.book.apply(update)
            pbar.update(1)
            rep._advance() # Move to next update

            # Skip events outside the continuous session
            if not (start_ns <= t_ns <= end_ns):
                continue

            # --- A. Capture time-based snapshots ---
            if t_ns >= next_snapshot_time_ns:
                # Take snapshot
                bid_p, bid_q, ask_p, ask_q = top_n_levels(rep.book, n=5)
                best_bid = bid_p[0]
                best_ask = ask_p[0]
                if np.isfinite(best_bid) and np.isfinite(best_ask):
                    spread = best_ask - best_bid
                    mid_price = (best_ask + best_bid) / 2
                else:
                    spread = np.nan
                    mid_price = np.nan

                row = {
                    "timestamp": pd.to_datetime(next_snapshot_time_ns, unit='ns', utc=True).tz_convert('Europe/Paris'),
                    "imbalance": rep.book.calculate_imbalance(max_levels=1),
                    "spread": spread,
                    "mid_price": mid_price,
                }
                for i in range(5):
                    row[f"bid{i+1}"] = bid_p[i]
                    row[f"bidvolume{i+1}"] = bid_q[i]
                    row[f"ask{i+1}"] = ask_p[i]
                    row[f"askvolume{i+1}"] = ask_q[i]

                snapshot_rows.append(row)
                # Advance to the next snapshot time
                next_snapshot_time_ns += snapshot_interval_ns

            # --- B. Capture event-based changes ---
            bid, ask = best_bid_ask(rep.book)
            event_rows.append({
                "timestamp": pd.to_datetime(t_ns, unit='ns', utc=True).tz_convert('Europe/Paris'),
                "action": update.action,
                "best_bid": bid,
                "best_ask": ask,
                "spread": (ask - bid) if bid and ask else None,
                "mid_price": (ask + bid) / 2 if bid and ask else None,
                "imbalance": rep.book.calculate_imbalance(max_levels=1),
            })

    # 3. Create DataFrames and save
    print("\n[info] Saving exported data...")
    if event_rows:
        df_events = pd.DataFrame(event_rows).set_index("timestamp")
        if event_out_path.endswith(".csv"):
            df_events.to_csv(event_out_path)
        elif event_out_path.endswith(".parquet"):
            df_events.to_parquet(event_out_path)
        print(f"[done] Saved {len(df_events)} event-based states to: {event_out_path}")
    else:
        print("[warning] No event-based data was generated.")

    if snapshot_rows:
        df_snapshots = pd.DataFrame(snapshot_rows).set_index("timestamp")
        if snapshot_out_path.endswith(".csv"):
            df_snapshots.to_csv(snapshot_out_path)
        elif snapshot_out_path.endswith(".parquet"):
            df_snapshots.to_parquet(snapshot_out_path)
        print(f"[done] Saved {len(df_snapshots)} snapshots to: {snapshot_out_path}")
    else:
        print("[warning] No snapshot data was generated.")

if __name__ == "__main__":
    order_update_path = "C:\\Users\\Wentao\\Desktop\\EA_recherche\\euronextparis\\EuronextParis\\EuronextParis_20191001_FR0000120578\\FR0000120578\\OrderUpdate_20191001_FR0000120578.csv"
    intervals_path = "C:\\Users\\Wentao\\Desktop\\EA_recherche\\msc_decoded\\merged_intervals.csv"

    # 1) build replayer
    rep = OrderBookReplayer(order_update_path, include_market_in_depth=False)

    # 2) Export book states (new feature)
    # Using parquet for efficiency, but .csv is also supported.
    export_orderbook_snapshots(
        rep,
        intervals_path=intervals_path,
        event_out_path="sanofi_book_events.parquet",
        snapshot_out_path="sanofi_book_snapshots_1s.parquet",
        snapshot_freq_s=1,
    )

    # 3) Find continuous trading session for animation and plotting
    try:
        intervals_df = pd.read_csv(intervals_path)
        continuous_session = intervals_df[intervals_df["book_state"].str.contains("Continuous", na=False)]
        continuous_session = continuous_session.loc[continuous_session['duration_s'].idxmax()]
        start_ns_cont = int(continuous_session["start_ns"])
        end_ns_cont = int(continuous_session["end_ns"])
        print(f"[info] Main continuous session identified: {pd.to_datetime(start_ns_cont, unit='ns', utc=True).tz_convert('Europe/Paris')} to {pd.to_datetime(end_ns_cont, unit='ns', utc=True).tz_convert('Europe/Paris')}")
    except Exception as e:
        print(f"[warning] Could not read continuous session from {intervals_path}, will animate full day. Error: {e}")
        start_ns_cont, end_ns_cont = None, None


    # 4) Plot imbalance over the continuous session
    plot_imbalance_over_time(rep, intervals_path=intervals_path, step_ns=60_000_000_000) # 1 minute steps

    # 5) Animate only the continuous trading session
    print("\n[info] Starting animation for the continuous trading session...")
    rep.animate_day_mp4_with_ffmpeg(
        out_path="sanofi_depth_continuous.mp4",
        start_ns=start_ns_cont,
        end_ns=end_ns_cont,
        step_ns=20_000_000_000,  # 20 seconds per frame
        max_levels=50,
        fps=20,
    )

