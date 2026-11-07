"""
live_core.py - shared code for the two LIVE swing strategies (daily chart, Nifty 500, buy only).

PB  Pullback to the 20 EMA (20/200 EMA retracement): see pullback_signals.py for the rules.
CE  Compression -> Expansion near the 52-week high:
    close above 20 EMA and 200 EMA, 20 EMA above 200 EMA, close within 5% of the 52-week high,
    volume compressed in the last 10 days, today's volume >= 1.5x the 20-day average, a GREEN
    candle closing in the top half of its range, and the move is not already done:
    today's gain <= 3%, close <= 8% above the 20 EMA, up <= 10% in the last 5 days.

Both: entry when price breaks above the signal candle's HIGH within the next 3 trading days
(skip if it gaps more than 3% above). SL = swing low (PB: lower of signal/previous candle low;
CE: lowest low of the last 5 candles), at least 1% and at most 7.5% below the entry.
Target 1 : 1.5. Exit at stop or target, otherwise at the close of day 30.
"""
import swing
import ce_backtest as X
import pullback_signals as P

STRATS = {
    "PB": {"title": "PULLBACK TO THE 20 EMA (20/200 EMA retracement)", "tab": "Pullback",
           "name": "Pullback 20/200 EMA"},
    "CE": {"title": "COMPRESSION -> EXPANSION near the 52-week high", "tab": "CE",
           "name": "Compression-Expansion"},
}

# "the move is already done" rules for CE (set CE_USE_MOVED_RULES = False to switch them off)
CE_USE_MOVED_RULES = True
CE_MAX_DAY_GAIN = 3.0
CE_MAX_ABOVE_EMA20 = 8.0
CE_MAX_GAIN_5D = 10.0
MAX_LISTED = 30


def prepare_items(frames):
    """(ticker, daily table, arrays) for every stock with enough history."""
    return [(t, df, X.arrays_ce(df)) for t, df in frames.items() if len(df) >= 270]


def pb_signal(a, i):
    return P.pullback_signal(a, i)


def ce_signal(a, i):
    sig = X.ce_signal(a, i)
    if sig is None or not (sig["ema_order"] and sig["green_top"]):
        return None
    if CE_USE_MOVED_RULES and not (sig["day_gain"] <= CE_MAX_DAY_GAIN
                                   and sig["above20"] <= CE_MAX_ABOVE_EMA20
                                   and sig["gain5"] <= CE_MAX_GAIN_5D):
        return None
    for j in range(i - X.COOLDOWN + 1, i):          # repeat signals within 5 sessions are hidden
        if X.ce_signal(a, j) is not None:
            return None
    return sig


SIGNAL_FUNCS = {"PB": pb_signal, "CE": ce_signal}


# ---------------- evening scan ----------------
def scan(items, sector_of, key):
    """Signals of one strategy from the last 3 sessions. Returns (new, waiting, entered, skipped, last_date)."""
    swing.MAX_HOLD_DAYS = P.HOLD_DAYS
    fn = SIGNAL_FUNCS[key]
    last_date = max(df.index[-1].date() for _, df, _ in items)
    new, waiting, entered, skipped = [], [], [], 0
    for t, df, a in items:
        if df.index[-1].date() != last_date:
            continue
        last = len(df) - 1
        for d in range(P.WATCH_DAYS):
            i = last - d
            sig = fn(a, i)
            if sig is None:
                continue
            item = {"key": key, "t": t, "sym": t.replace(".NS", ""), "sector": sector_of.get(t, "?"),
                    "sig": sig, "date": df.index[i].date(), "df": df, "i": i, "a": a}
            if d == 0:
                p = P.plan(sig["trigger"], sig["sl_c"])
                if p is None:
                    skipped += 1
                    continue
                item.update(sl=p[0], tgt=p[1], risk=p[2])
                new.append(item)
            else:
                tr = P.track(a, sig, i, last)
                item.update(tr)
                if tr["state"] == "waiting":
                    waiting.append(item)
                elif tr["state"] == "entered":
                    entered.append(item)
    new.sort(key=lambda x: -(x["sig"].get("range_x") or x["sig"].get("vol_x") or 0))
    return new, waiting, entered, skipped, last_date


def section_lines(key, new, waiting, entered, skipped):
    L = [f"=== {STRATS[key]['title']} ===", f"NEW SIGNALS TODAY ({len(new)})"]
    if not new:
        L.append("   None today.")
    for x in new[:MAX_LISTED]:
        s, df, i = x["sig"], x["df"], x["i"]
        L.append(f"{x['sym']} ({x['sector']})")
        if key == "PB":
            more = f"size {s['range_x']:.1f}x last-5 avg"
        else:
            more = f"volume {s['vol_x']:.1f}x | gain today {s['day_gain']:+.1f}% | {s['dist52']:.1f}% below 52w high"
        L.append(f"   Candle: high {s['trigger']:.2f} / low {df['Low'].iloc[i]:.2f} / close {df['Close'].iloc[i]:.2f}"
                 f" | 20 EMA {x['a']['e20'][i]:.2f} | {more}")
        L.append(f"   BUY above {s['trigger']:.2f} (skip if above {s['trigger'] * (1 + P.EXTEND_PCT / 100):.2f}) "
                 f"within 3 trading days")
        L.append(f"   SL {x['sl']:.2f} ({x['risk']:.1f}% risk) | Target {x['tgt']:.2f} | Reward:Risk 1:{P.RR}")
    if len(new) > MAX_LISTED:
        L.append(f"   ... and {len(new) - MAX_LISTED} more")
    if skipped:
        L.append(f"   ({skipped} more skipped: stop wider than {P.MAX_RISK_PCT}%)")
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
        res = x["res"]
        status = "still open" if res is None else f"closed: {res[0]} at {res[1]:.2f}"
        L.append(f"{x['sym']} ({x['sector']}) - bought {x['df'].index[x['k']].strftime('%d %b')} at "
                 f"{x['entry']:.2f} | SL {x['sl']:.2f} | Target {x['tgt']:.2f} | {status}")
    return L


# ---------------- live monitor helpers ----------------
def build_watchlist(items, sector_of):
    """Signals from the last 3 COMPLETED sessions whose entry window is still open (no break yet)."""
    swing.MAX_HOLD_DAYS = P.HOLD_DAYS
    out = []
    for t, df, a in items:
        last = len(df) - 1
        for key, fn in SIGNAL_FUNCS.items():
            for d in range(P.WATCH_DAYS):
                j = last - d
                sig = fn(a, j)
                if sig is None:
                    continue
                tr = P.track(a, sig, j, last)
                if tr["state"] == "waiting":
                    out.append({"key": key, "t": t, "sym": t.replace(".NS", ""),
                                "sector": sector_of.get(t, "?"), "sig": sig,
                                "date": df.index[j].date()})
    return out


def find_break(d5, today, trigger):
    """First completed 5-minute candle today whose high reaches the trigger.
    Returns (candle time, entry price) or None. A gap above the trigger enters at the open."""
    d = d5[d5.index.date == today]
    if d.empty:
        return None
    hit = d[d["High"] >= trigger]
    if hit.empty:
        return None
    return hit.index[0], max(float(d["Open"].iloc[0]), trigger)


def handle_break(w, d5, today, ts, entry):
    """Applies the entry rules. Returns (alert text, sheet row) or None if the trade is skipped."""
    trig, sl_c = w["sig"]["trigger"], w["sig"]["sl_c"]
    if entry > trig * (1 + P.EXTEND_PCT / 100):
        return None                                     # gapped too far above the high
    p = P.plan(entry, sl_c)
    if p is None:
        return None                                     # stop wider than 7.5%
    sl, tgt, risk_pct = p
    after = d5[(d5.index.date == today) & (d5.index >= ts)]
    if float(after["Low"].min()) <= sl:
        return None                                     # already hit the stop after breaking out
    st = STRATS[w["key"]]
    gap = " (gapped up - entry at the open)" if entry > trig else ""
    text = (f"🟢 {st['name'].upper()} ENTRY - {w['sym']} ({w['sector']})\n"
            f"   Price broke above the signal high {trig:.2f} (signal day {w['date'].strftime('%d %b')}){gap}\n"
            f"   Entry {entry:.2f} | SL {sl:.2f} | Target {tgt:.2f}\n"
            f"   Risk {risk_pct:.1f}% | Reward:Risk 1:{P.RR} | Time exit after {P.HOLD_DAYS} trading days")
    row = {"ID": f"{w['date']}|{w['sym']}|{w['key']}", "Date": today.strftime("%Y-%m-%d"),
           "Time": ts.strftime("%H:%M"), "Strategy": st["name"], "Stock": w["sym"], "Sector": w["sector"],
           "Side": "BUY", "Entry Type": "BREAK_OF_HIGH", "Entry": round(entry, 2), "SL": round(sl, 2),
           "Target": round(tgt, 2), "Status": "OPEN", "Exit Price": "", "Exit Time": "", "R": "",
           "Notes": f"Signal day {w['date']}"}
    return text, row
