# decode_msc_with_intervals.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, List, Tuple
import re
import os
import pandas as pd

# =========================
# A) 你只需要改这里：输入/输出路径
# =========================
INPUT_CSV = r"C:\Users\Wentao\Desktop\EA_recherche\euronextparis\EuronextParis\EuronextParis_20191001_FR0000120578\FR0000120578\MarketStatusChange_20191001_FR0000120578.csv"
OUTPUT_DIR = r"C:\Users\Wentao\Desktop\EA_recherche\msc_decoded"   # 输出文件夹（不存在会自动创建）
TIMEZONE_LOCAL = "Europe/Paris"                                    # 本地时区显示

# 输出文件名（会自动写到 OUTPUT_DIR 下）
OUT_EVENTS_CSV = "decoded_events.csv"
OUT_INTERVALS_CSV = "merged_intervals.csv"

# =========================
# B) 枚举映射（按说明书）
# =========================
BOOK_STATE = {
    1: "Inaccessible（不可访问）",
    2: "Closed（关闭）",
    3: "Call（集合竞价/叫价阶段）",
    4: "Uncrossing（撮合/解交叉）",
    5: "Continuous（连续交易）",
    6: "Halted（临时停牌/暂停撮合）",
    8: "Suspended（停牌）",
    9: "Reserved（保留）",
}
STATUS_REASON = {
    0: "Scheduled（按计划）",
    4: "Collars Breach（触发价格保护带）",
    7: "Automatic Reopening（自动重新开市）",
    8: "No Liquidity Provider（无流动性提供者）",
    11: "Knock-In by Issuer",
    12: "Knock-Out by Exchange",
    13: "Knock-Out by Issuer",
    15: "Action by Market Operations（市场运营操作）",
    20: "New Listing（新上市）",
    21: "Due to Underlying（由于标的）",
    22: "Outside of LP quotes（超出做市商报价）",
}
PHASE_QUALIFIER = {
    0: "No Qualifier（无）",
    1: "Call BBO Only",
    2: "Trading At Last",
    3: "Random Uncrossing（随机撮合）",
    5: "Trading At Last + Random Uncrossing",
    6: "Stressed Market Conditions",
    7: "Exceptional Market Conditions",
    9: "Random Uncrossing + Stressed",
    11: "Trading At Last + Random Uncrossing + Stressed",
}
TRADING_PERIOD = {
    1: "Opening（开盘段）",
    2: "Standard（盘中）",
    3: "Closing（收盘段）",
}
TRADING_SIDE = {
    1: "Bid Only（仅买）",
    2: "Offer Only（仅卖）",
    3: "PAKO",
    4: "Both Sides（双边）",
}
ORDER_ENTRY_QUALIFIER = {
    0: "Disabled（禁止下单/撤单/改单）",
    1: "Enabled（允许下单/撤单/改单）",
    3: "Cancel Only（只允许撤单）",
}
SCHEDULED_EVENT = {
    0: "Cancel Previously Scheduled Event（取消预告）",
    1: "Reopening（重新开市）",
    3: "Resumption of trading（恢复交易）",
    12: "Suspension（停牌）",
    13: "Collars Normal",
    14: "Collars Wide",
    15: "Pre-Expiry",
    16: "Closing Price",
}
MD_CHANGE_TYPE = {
    0: "Status Change(s)（状态变化）",
    1: "Scheduled Event Notification（预告事件）",
    2: "Status Change(s) + Scheduled Event（状态+预告）",
}

# =========================
# C) 解析器：读取 MarketStatusChange CSV
# =========================
_LINE_RE  = re.compile(r"^([^,]+),([^,]+),([^,]+),([^,]+),(\[.*\])\s*$")
_SEQ_RE   = re.compile(r"^\[(\d+),(.*)\]$")
_BRACE_RE = re.compile(r"\{([^}]*)\}")

def _to_int(x: str) -> Optional[int]:
    x = x.strip()
    return None if x == "" else int(x)

def _dt_str(ns: Optional[int], tz: str) -> str:
    if ns is None or ns == 0:
        return ""
    dt = pd.to_datetime(int(ns), unit="ns", utc=True).tz_convert(tz)
    return str(dt)

def _map(d: Dict[int, str], x: Optional[int]) -> str:
    if x is None:
        return ""
    return d.get(x, f"Unknown({x})")

@dataclass
class MarketStatusChangeRow:
    md_seq: int
    rebroadcast: int
    emm: int
    no_changes: int

    md_change_type: int
    symbol_index: int
    event_time_ns: int

    book_state: Optional[int]
    status_reason: Optional[int]
    phase_qualifier: int
    trading_period: Optional[int]
    trading_side: Optional[int]

    price_limits: Optional[str]
    quote_spread_multiplier: Optional[str]

    order_entry_qualifier: Optional[int]
    session: int
    scheduled_event: Optional[int]
    scheduled_event_time_ns: Optional[int]

def iter_market_status_change(path: str) -> Iterable[MarketStatusChangeRow]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = _LINE_RE.match(line)
            if not m:
                continue

            md_seq, msg_type, reb, emm, seq = m.groups()
            if msg_type != "MarketStatusChange":
                continue

            md_seq = int(md_seq)
            reb = int(reb)
            emm = int(emm)

            sm = _SEQ_RE.match(seq)
            if not sm:
                continue
            no_changes = int(sm.group(1))
            body = sm.group(2)

            groups = _BRACE_RE.findall(body)
            for g in groups:
                fields = g.split(",")

                # 2019：每个 group = 14 个字段（Instrument State 2020 才有）
                if len(fields) < 14:
                    fields += [""] * (14 - len(fields))
                fields = fields[:14]

                md_change_type = _to_int(fields[0]) or 0
                symbol_index = _to_int(fields[1]) or 0
                event_time_ns = _to_int(fields[2]) or 0

                book_state = _to_int(fields[3])
                status_reason = _to_int(fields[4])
                phase_qualifier = _to_int(fields[5]) or 0
                trading_period = _to_int(fields[6])
                trading_side = _to_int(fields[7])

                price_limits = fields[8].strip() or None
                quote_spread_multiplier = fields[9].strip() or None

                order_entry_qualifier = _to_int(fields[10])
                session = _to_int(fields[11]) or 1
                scheduled_event = _to_int(fields[12])
                scheduled_event_time_ns = _to_int(fields[13])

                yield MarketStatusChangeRow(
                    md_seq=md_seq,
                    rebroadcast=reb,
                    emm=emm,
                    no_changes=no_changes,
                    md_change_type=md_change_type,
                    symbol_index=symbol_index,
                    event_time_ns=event_time_ns,
                    book_state=book_state,
                    status_reason=status_reason,
                    phase_qualifier=phase_qualifier,
                    trading_period=trading_period,
                    trading_side=trading_side,
                    price_limits=price_limits,
                    quote_spread_multiplier=quote_spread_multiplier,
                    order_entry_qualifier=order_entry_qualifier,
                    session=session,
                    scheduled_event=scheduled_event,
                    scheduled_event_time_ns=scheduled_event_time_ns,
                )

# =========================
# D) 解码为事件表（逐条更清楚）
# =========================
def decode_events(path: str, tz_local: str) -> pd.DataFrame:
    rows = []
    for m in iter_market_status_change(path):
        rows.append({
            "event_time_ns": m.event_time_ns,
            "event_time_utc": _dt_str(m.event_time_ns, "UTC"),
            "event_time_local": _dt_str(m.event_time_ns, tz_local),

            "md_seq": m.md_seq,
            "rebroadcast": m.rebroadcast,
            "emm": m.emm,
            "no_changes": m.no_changes,

            "md_change_type_raw": m.md_change_type,
            "md_change_type": _map(MD_CHANGE_TYPE, m.md_change_type),

            "symbol_index": m.symbol_index,

            "book_state_raw": m.book_state,
            "book_state": _map(BOOK_STATE, m.book_state),

            "status_reason_raw": m.status_reason,
            "status_reason": _map(STATUS_REASON, m.status_reason),

            "phase_qualifier_raw": m.phase_qualifier,
            "phase_qualifier": _map(PHASE_QUALIFIER, m.phase_qualifier),

            "trading_period_raw": m.trading_period,
            "trading_period": _map(TRADING_PERIOD, m.trading_period),

            "trading_side_raw": m.trading_side,
            "trading_side": _map(TRADING_SIDE, m.trading_side),

            "order_entry_qualifier_raw": m.order_entry_qualifier,
            "order_entry_qualifier": _map(ORDER_ENTRY_QUALIFIER, m.order_entry_qualifier),

            "session": m.session,

            "scheduled_event_raw": m.scheduled_event,
            "scheduled_event": _map(SCHEDULED_EVENT, m.scheduled_event),

            "scheduled_event_time_local": _dt_str(m.scheduled_event_time_ns, tz_local),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("event_time_ns").reset_index(drop=True)
    return df

# =========================
# E) 合并区间（更清楚的“状态段”表）
# =========================
def build_intervals(events_df: pd.DataFrame, tz_local: str) -> pd.DataFrame:
    """
    每条事件定义一个状态，从 event_time 开始到下一条 event_time 结束。
    合并相邻且状态字段完全相同的区间。
    """
    if events_df.empty:
        return events_df

    # 定义“状态等价”用哪些列（你也可以自行删减/增加）
    key_cols = [
        "book_state_raw",
        "status_reason_raw",
        "phase_qualifier_raw",
        "trading_period_raw",
        "trading_side_raw",
        "order_entry_qualifier_raw",
    ]

    starts = events_df["event_time_ns"].to_list()
    ends = starts[1:] + [None]  # 最后一段没有明确结束，先留空

    intervals = []
    for i in range(len(events_df)):
        row = events_df.iloc[i]
        intervals.append({
            "start_ns": int(row["event_time_ns"]),
            "end_ns": int(ends[i]) if ends[i] is not None else None,

            "start_local": _dt_str(int(row["event_time_ns"]), tz_local),
            "end_local": _dt_str(int(ends[i]), tz_local) if ends[i] is not None else "",

            "book_state": row["book_state"],
            "status_reason": row["status_reason"],
            "phase_qualifier": row["phase_qualifier"],
            "trading_period": row["trading_period"],
            "trading_side": row["trading_side"],
            "order_entry_qualifier": row["order_entry_qualifier"],

            # 原始值也保留一份，方便你之后做过滤/对齐
            **{k: row[k] for k in key_cols},
        })

    # 合并相邻且 key 相同的区间
    merged: List[dict] = []
    for itv in intervals:
        if not merged:
            merged.append(itv)
            continue

        prev = merged[-1]
        same = all(prev[k] == itv[k] for k in key_cols)

        # 只有在 prev.end == itv.start 且 key 相同的情况下才合并
        if same and prev["end_ns"] == itv["start_ns"]:
            prev["end_ns"] = itv["end_ns"]
            prev["end_local"] = itv["end_local"]
        else:
            merged.append(itv)

    # 计算持续时间
    for r in merged:
        if r["end_ns"] is None:
            r["duration_s"] = ""
        else:
            r["duration_s"] = (r["end_ns"] - r["start_ns"]) / 1e9

    out = pd.DataFrame(merged)

    # 更清楚的列顺序
    cols = [
        "start_local", "end_local", "duration_s",
        "book_state", "trading_period", "phase_qualifier",
        "status_reason", "trading_side", "order_entry_qualifier",
        "start_ns", "end_ns",
        # raw keys
        "book_state_raw", "status_reason_raw", "phase_qualifier_raw",
        "trading_period_raw", "trading_side_raw", "order_entry_qualifier_raw",
    ]
    cols = [c for c in cols if c in out.columns]
    out = out[cols]
    return out

# =========================
# F) 主程序：自动保存两张表
# =========================
def main():
    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(f"Input not found: {INPUT_CSV}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    events_df = decode_events(INPUT_CSV, TIMEZONE_LOCAL)
    if events_df.empty:
        print("[warn] No MarketStatusChange rows parsed. Check file format / content.")
        return

    intervals_df = build_intervals(events_df, TIMEZONE_LOCAL)

    out_events = os.path.join(OUTPUT_DIR, OUT_EVENTS_CSV)
    out_intervals = os.path.join(OUTPUT_DIR, OUT_INTERVALS_CSV)

    events_df.to_csv(out_events, index=False, encoding="utf-8")
    intervals_df.to_csv(out_intervals, index=False, encoding="utf-8")

    print(f"[done] decoded events -> {out_events}  (rows={len(events_df)})")
    print(f"[done] merged intervals -> {out_intervals}  (rows={len(intervals_df)})")

    # 额外：在终端打印前几行让你确认
    print("\n=== Preview: merged intervals (first 20) ===")
    print(intervals_df.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
