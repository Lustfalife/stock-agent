"""
pullback_signals.py - evening signal list for the "Pullback to the 20 EMA" strategy.
Signal only: no Google Sheet logging, no live monitor.

STRATEGY (daily chart, Nifty 500, buy only)
 1. Close > 20 EMA > 200 EMA, 20 EMA rising, 200 EMA not falling.
 2. Pullback: in the last 10 days the stock was 4%+ above its 20 EMA, then came back
    down to it; 6 of the last 8 closes were above the 20 EMA; the pullback low is a
    higher low than the lowest low of days 30 to 5 back.
 3. Signal candle: green, low touches/goes below the 20 EMA, closes above it, closes in the
    top 60% of its range, range >= 1.2x the average range of the previous 5 candles.
 4. Entry: within the next 3 trading days, price breaks above the signal candle's HIGH
    (skip if already more than 3% above it; cancel if a close falls below the stop first).
 5. Stop: lower of the signal candle's low and the previous candle's low, at least 1% below
    the entry; skip if more than 7.5% away.
 6. Target 1.5 x risk. Exit at stop or target; otherwise at the close of day 30.
"""
import datetime as dt
import sys

import swing
import swing_backtest as B
import ema20_backtest as E
import core_backtest as C
from intraday import now_ist, safe_send, load_universe

RR = 1.5
MIN_RISK_PCT = 1.0
MAX_RISK_PCT = 7.5
EXTEND_PCT = 3.0
WATCH_DAYS = 3
HOLD_DAYS = 30
COOLDOWN = 5
MAX_LISTED = 40


def plan(entry, sl_c):
    """(stop, target, risk %) or None if the stop is too wide."""
    sl = min(sl_c, entry * (1 - MIN_RISK_PCT / 100))
    risk = entry - sl
    risk_pct = risk / entry * 100
    if risk <= 0 or risk_pct > MAX_RISK_PCT:
        return None
    return sl, entry + RR * risk, risk_pct


def pullback_signal(a, i):
    """The pullback signal on session i, or None (repeat signals within 5 sessions are hidden)."""
    st, sig = C.core_stage(a, i)
    if sig is None or sig["mode"] != "PB":
        return None
    for j in range(i - COOLDOWN + 1, i):
        if C.core_stage(a, j)[1] is not None:
            return None
    return sig


def track(a, sig, i, last):
    """What happened to a signal from session i, looking at sessions up to `last`."""
    o, h, l, c = a["o"], a["h"], a["l"], a["c"]
    trig, sl_c = sig["trigger"], sig["sl_c"]
    p0 = plan(trig, sl_c)
    if p0 is None:
        return {"state": "skip"}
    sl0 = p0[0]
    for k in range(i + 1, min(i + WATCH_DAYS, last) + 1):
        if k > i + 1 and (c[i + 1:k] < sl0).any():
            return {"state": "cancelled"}
        if h[k] >= trig:
            entry = max(o[k], trig)
            if entry > trig * (1 + EXTEND_PCT / 100):
                return {"state": "missed"}
            p = plan(entry, sl_c)
            if p is None:
                return {"state": "skip"}
            sl, tgt, rp = p
            res = B.exit_trade(o, h, l, c, k, k, sl, tgt, same_day=True)
            return {"state": "entered", "k": k, "entry": entry, "sl": sl, "tgt": tgt,
                    "risk": rp, "res": res}
    if last - i >= WATCH_DAYS:
        return {"state": "expired"}
    return {"state": "waiting", "sl": p0[0], "tgt": p0[1], "risk": p0[2]}


def scan(frames, sector_of):
    """Returns (new, waiting, entered, skipped_wide, last_date)."""
    swing.MAX_HOLD_DAYS = HOLD_DAYS
    new, waiting, entered, skipped = [], [], [], 0
    last_date = max(df.index[-1].date() for df in frames.values())
    for t, df in frames.items():
        if df.index[-1].date() != last_date or len(df) < 270:
            continue
        a = E.arrays(df)
        last = len(df) - 1
        for d in range(0, WATCH_DAYS):
            i = last - d
            sig = pullback_signal(a, i)
            if sig is None:
                continue
            item = {"sym": t.replace(".NS", ""), "sector": sector_of.get(t, "?"), "sig": sig,
                    "date": df.index[i].date(), "df": df, "i": i, "a": a}
            if d == 0:
                p = plan(sig["trigger"], sig["sl_c"])
                if p is None:
                    skipped += 1
                    continue
                item.update(sl=p[0], tgt=p[1], risk=p[2])
                new.append(item)
            else:
                tr = track(a, sig, i, last)
                item.update(tr)
                if tr["state"] == "waiting":
                    waiting.append(item)
                elif tr["state"] == "entered":
                    entered.append(item)
    new.sort(key=lambda x: -x["sig"]["range_x"])
    return new, waiting, entered, skipped, last_date


def build_message(new, waiting, entered, skipped, last_date, partial_note=""):
    day = last_date.strftime("%d %b %Y")
    L = [f"📈 PULLBACK SIGNALS - after close {day}{partial_note}",
         "Pullback to the 20 EMA (daily, buy only). Buy only if price breaks above the signal high "
         "within the next 3 trading days.", ""]
    L.append(f"NEW SIGNALS TODAY ({len(new)})")
    if not new:
        L.append("   None today.")
    for x in new[:MAX_LISTED]:
        s = x["sig"]
        df, i = x["df"], x["i"]
        L.append(f"{x['sym']} ({x['sector']})")
        L.append(f"   Candle: high {s['trigger']:.2f} / low {df['Low'].iloc[i]:.2f} / close {df['Close'].iloc[i]:.2f}"
                 f" | 20 EMA {x['a']['e20'][i]:.2f} | size {s['range_x']:.1f}x last-5 avg")
        L.append(f"   BUY above {s['trigger']:.2f} (skip if above {s['trigger'] * (1 + EXTEND_PCT / 100):.2f}) "
                 f"within 3 trading days")
        L.append(f"   SL {x['sl']:.2f} ({x['risk']:.1f}% risk) | Target {x['tgt']:.2f} | Reward:Risk 1:{RR}")
    if len(new) > MAX_LISTED:
        L.append(f"   ... and {len(new) - MAX_LISTED} more")
    if skipped:
        L.append(f"   ({skipped} more signal(s) skipped because the stop would be wider than {MAX_RISK_PCT}%)")
    L += ["", f"STILL WAITING - entry window open ({len(waiting)})"]
    if not waiting:
        L.append("   None.")
    for x in waiting[:MAX_LISTED]:
        L.append(f"{x['sym']} ({x['sector']}) - signal {x['date'].strftime('%d %b')}: buy above "
                 f"{x['sig']['trigger']:.2f} | SL {x['sl']:.2f} | Target {x['tgt']:.2f}")
    L += ["", f"ENTERED in the last 3 days ({len(entered)})"]
    if not entered:
        L.append("   None.")
    for x in entered[:MAX_LISTED]:
        df = x["df"]
        res = x["res"]
        status = "still open" if res is None else f"closed: {res[0]} at {res[1]:.2f}"
        L.append(f"{x['sym']} ({x['sector']}) - bought {df.index[x['k']].strftime('%d %b')} at {x['entry']:.2f} | "
                 f"SL {x['sl']:.2f} | Target {x['tgt']:.2f} | {status}")
    L += ["", "Exit at stop or target; otherwise at the close of day 30. Signal tool only - "
              "check the chart before trading."]
    return "\n".join(L)


def main():
    uni = load_universe()
    sector_of = dict(zip(uni["ticker"], uni["sector"]))
    frames = swing.download_daily(uni["ticker"].tolist())
    if not frames:
        safe_send("❌ Pullback signals: could not download price data.")
        sys.exit(1)
    note = ""
    now = now_ist()
    if now.time() < dt.time(16, 0):            # market may still be open: ignore today's unfinished candle
        before = sum(1 for df in frames.values() if df.index[-1].date() == now.date())
        if before:
            frames = {t: (df.iloc[:-1] if df.index[-1].date() == now.date() else df) for t, df in frames.items()}
            note = " (market still open - last completed day used)"
    new, waiting, entered, skipped, last_date = scan(frames, sector_of)
    safe_send(build_message(new, waiting, entered, skipped, last_date, note))


if __name__ == "__main__":
    main()
