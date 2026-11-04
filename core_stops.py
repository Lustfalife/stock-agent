"""
core_stops.py - two ideas on the core strategy (pullback only + stronger candle):

IDEA 2  Stop with breathing room: stop 0%, 0.5%, 1%, 1.5% BELOW the candle's stop level.
        The target moves out so it stays 1:1.5. Trades whose stop would be wider than 7.5%
        are skipped.
IDEA 4  Weekly trend filter: weekly close above the weekly 9 EMA, or above the weekly 20 EMA.
        Only weeks that were already complete on the signal day are used (no peeking).

Base signals = pullback setup + candle range >= 1.5x the last-5 average + close in top 80%.
Entry = break of the signal candle's high within 3 days. Target 1:1.5. Hold up to 30 days.
If entry and stop are both possible on the same day, the worse outcome (stop) is counted.
"""
import sys

import numpy as np
import pandas as pd

import swing
import swing_backtest as B
import ema20_backtest as E
import core_backtest as C
from intraday import load_universe, safe_send

BUFFERS = [0.0, 0.5, 1.0, 1.5]
MIN_RANGE_X = 1.5
MIN_CLOSE_POS = 0.8


def collect_signals(data):
    """Pullback signals that pass the stronger-candle rule. Cooldown is applied to ALL core
    signals first (exactly as in the earlier backtests), then the filters."""
    out = []
    for sym, df, a in data:
        last = -10 ** 9
        h, l, c = a["h"], a["l"], a["c"]
        for i in range(C.EVAL_FROM, len(df)):
            st, sig = C.core_stage(a, i)
            if sig is None or i - last < C.COOLDOWN:
                continue
            last = i
            if sig["mode"] != "PB":
                continue
            if sig["range_x"] < MIN_RANGE_X or (c[i] - l[i]) / (h[i] - l[i]) < MIN_CLOSE_POS:
                continue
            out.append((sym, df, a, i, sig))
    return out


def weekly_flags(df):
    """(weekly close > weekly 9 EMA, weekly close > weekly 20 EMA), labelled by week-end Friday."""
    wc = df["Close"].resample("W-FRI").last().dropna()
    e9 = wc.ewm(span=9, adjust=False).mean()
    e20 = wc.ewm(span=20, adjust=False).mean()
    return wc > e9, wc > e20


def flag_on(series, day):
    """Last completed week on or before `day` (week label is the Friday)."""
    v = series.asof(pd.Timestamp(day))
    return bool(v) if pd.notna(v) else False


def run_variant(signals, weekly, buffer):
    counts = {"CORE": {"risk_ok": 0, "res_ok": 0}}
    rows = []
    with C.settings(7.5):
        for sym, df, a, i, sig in signals:
            sl_c = sig["sl_c"] * (1 - buffer / 100)
            t = E.try_trade("CORE", a, df, sym, i,
                            {"trigger": sig["trigger"], "candle": sig["mode"]}, sl_c, counts)
            if t:
                w9, w20 = weekly[sym]
                t["wk9"] = flag_on(w9, t["signal_date"])
                t["wk20"] = flag_on(w20, t["signal_date"])
                t["buffer"] = buffer
                rows.append(t)
    return pd.DataFrame(rows)


def main():
    uni = load_universe()
    frames = B.download_long(uni["ticker"].tolist())
    if not frames:
        safe_send("❌ Core stops: could not download price data.")
        sys.exit(1)
    data = [(s.replace(".NS", ""), df, E.arrays(df)) for s, df in frames.items()]
    weekly = {s.replace(".NS", ""): weekly_flags(df) for s, df in frames.items()}
    signals = collect_signals(data)
    first = min(df.index[C.EVAL_FROM].date() for df in frames.values())
    last = max(df.index[-1].date() for df in frames.values())
    L = [f"🧪 CORE STOPS + WEEKLY TREND - Nifty 500 ({first} to {last})",
         "Pullback only + candle 1.5x last-5 avg + close in top 80%. Target 1:1.5, hold up to 30 days.",
         "Stop buffer = extra room below the candle stop; trades with stop wider than 7.5% are skipped.",
         "Format: trades, win, avg (net of 0.3% cost) | 22-24: n / avg | 25-26: n / avg",
         f"Signals tested: {len(signals)}", ""]
    all_rows = []
    for buf in BUFFERS:
        t = run_variant(signals, weekly, buf)
        name = "Reference: stop at the candle level" if buf == 0 else f"Stop {buf}% below the candle level"
        L.append(name)
        if len(t) == 0:
            L.append("   no trades")
        else:
            L.append(f"   no weekly filter: {C.fmt(t)}")
            L.append(f"   weekly close > 9 EMA: {C.fmt(t[t.wk9])}")
            L.append(f"   weekly close > 20 EMA: {C.fmt(t[t.wk20])}")
            all_rows.append(t)
        L.append("")
    L += ["Keep a change only if it helps in BOTH periods and after costs.",
          "Trade list: download core_stops.csv from the run page (Artifacts)."]
    if all_rows:
        pd.concat(all_rows).to_csv("core_stops.csv", index=False)
    safe_send("\n".join(L))


if __name__ == "__main__":
    main()
