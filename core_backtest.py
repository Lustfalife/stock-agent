"""
core_backtest.py - ONE combined 20-EMA strategy, built from your charts.

SIGNAL CANDLE (daily): green, its LOW touches/goes below the 20 EMA, it CLOSES above
the 20 EMA, closes in the top 60% of its range, and its range is at least 1.2x the
average range of the last 5 candles.
SETUP (one of two):
  PB  pullback: close > 20 EMA > 200 EMA, 20 EMA rising, stock ran up (4%+ above the EMA)
      and pulled back, the pullback low is a HIGHER LOW, and it did not climb from below.
  BX  box: 20 EMA flat (within +/-2% in 10 days), a tight 5-day box (<= 6%) with quiet
      volume, and the signal candle closes ABOVE the box.
ENTRY: break of the signal candle's high within 3 days.
SL: PB = lower of signal and previous candle low; BX = box low (at least 1% below entry).
Max stop 5% and 7.5% are both tested. Target 1:1.5. Hold up to 30 days.
Filters tested on/off afterwards: volume >= 1.2x, resistance space, Nifty > 200 EMA,
max 3 signals a day. Also checks whether YOUR example trades are found.
"""
import sys
import contextlib

import numpy as np
import pandas as pd
import yfinance as yf

import swing
import swing_backtest as B
import ema20_backtest as E
from intraday import load_universe, safe_send

# ===================== SETTINGS YOU CAN CHANGE =====================
EVAL_FROM = 260
COOLDOWN = 5
FLAT_PCT = 2               # 20 EMA counts as flat if it moved less than 2% in 10 days
MIN_RUNUP_PCT = 4          # PB: recent high at least 4% above the 20 EMA
BOX_DAYS = 5
BOX_MAX_RANGE_PCT = 6
BOX_VOL_COMP = 0.9         # BX: box volume below 0.9 x the 20 days before it
CLOSE_POS = 0.6            # close in the top 60% of the candle's range
SIZE_MULT = 1.2            # candle range >= 1.2 x average of last 5 candles
VOL_MIN = 1.2              # volume filter (tested on/off)
COST_PCT = 0.3             # shown as "net" (assumed cost per trade)
SPLIT = pd.Timestamp("2025-01-01")
KNOWN = [("NYKAA", "2026-04-02"), ("FINCABLES", "2026-09-03"), ("GLAND", "2026-06-26"),
         ("GLENMARK", "2026-08-19"), ("GODAVARIB", "2026-03-27")]
# ====================================================================

STAGE_TEXT = ["close not above 20 EMA, or 20 EMA not above 200 EMA",
              "20 EMA neither rising nor flat",
              "no valid setup (pullback: ran up + higher low; box: tight and quiet)",
              "candle not green / low not touching 20 EMA / close not above it / weak close "
              "(box: close not above the box)",
              "candle smaller than 1.2x the average of the last 5 candles",
              "signal found"]
FUNNEL_NAMES = ["Close > 20 EMA > 200 EMA", "20 EMA rising or flat", "Valid pullback or box setup",
                "Green candle touching 20 EMA, closing above", "Candle >= 1.2x last-5 average"]


@contextlib.contextmanager
def settings(max_stop):
    old = (swing.MAX_HOLD_DAYS, E.WATCH_DAYS, E.MAX_RISK_PCT)
    swing.MAX_HOLD_DAYS, E.WATCH_DAYS, E.MAX_RISK_PCT = 30, 3, max_stop
    try:
        yield
    finally:
        swing.MAX_HOLD_DAYS, E.WATCH_DAYS, E.MAX_RISK_PCT = old


def core_stage(a, i):
    """Returns (rules passed 0-5, signal dict or None)."""
    o, h, l, c, v, e20, e200 = a["o"], a["h"], a["l"], a["c"], a["v"], a["e20"], a["e200"]
    if not (c[i] > e20[i] > e200[i]):
        return 0, None
    rising = e20[i] > e20[i - 5] and e200[i] >= e200[i - 20] * 0.999
    flat = abs(e20[i] / e20[i - 10] - 1) * 100 <= FLAT_PCT and e200[i] >= e200[i - 20] * 0.995
    if not (rising or flat):
        return 1, None
    mode, bh, bl = None, None, None
    if rising:
        higher_low = l[i - 2:i + 1].min() > l[i - 30:i - 5].min()
        ran_up = (h[i - 10:i].max() / e20[i] - 1) * 100 >= MIN_RUNUP_PCT
        from_above = (c[i - 8:i] > e20[i - 8:i]).sum() >= 6
        if higher_low and ran_up and from_above:
            mode = "PB"
    if mode is None and flat:
        bx = slice(i - BOX_DAYS, i)
        bh, bl = h[bx].max(), l[bx].min()
        if (bh - bl) / bl * 100 <= BOX_MAX_RANGE_PCT:
            prior = v[i - BOX_DAYS - 20:i - BOX_DAYS].mean()
            if prior > 0 and v[bx].mean() < BOX_VOL_COMP * prior:
                mode = "BX"
    if mode is None:
        return 2, None
    rng = h[i] - l[i]
    if rng <= 0 or not (c[i] > o[i] and l[i] <= e20[i] and c[i] > e20[i]
                        and (c[i] - l[i]) / rng >= CLOSE_POS):
        return 3, None
    if mode == "BX" and not c[i] > bh:
        return 3, None
    avg5 = (h[i - 5:i] - l[i - 5:i]).mean()
    if avg5 <= 0 or rng < SIZE_MULT * avg5:
        return 4, None
    sl_c = min(l[i], l[i - 1]) if mode == "PB" else min(bl, l[i])
    avg20 = v[i - 20:i].mean()
    vol_x = v[i] / avg20 if avg20 > 0 else 0.0
    return 5, {"mode": mode, "trigger": h[i], "sl_c": sl_c, "vol_x": vol_x, "range_x": rng / avg5}


def explain(a, i):
    o, h, l, c, e20 = a["o"], a["h"], a["l"], a["c"], a["e20"]
    rng = h[i] - l[i]
    avg5 = (h[i - 5:i] - l[i - 5:i]).mean()
    return (f"20 EMA slope {((e20[i] / e20[i - 10]) - 1) * 100:+.1f}%/10d, "
            f"green={bool(c[i] > o[i])}, low<=20EMA={bool(l[i] <= e20[i])}, "
            f"close>20EMA={bool(c[i] > e20[i])}, size {rng / avg5 if avg5 > 0 else 0:.2f}x last-5 avg")


def run_core(data, max_stop):
    counts = {"CORE": {"risk_ok": 0, "res_ok": 0}}
    funnel = np.zeros(6, dtype=np.int64)
    signals, rows = 0, []
    with settings(max_stop):
        for sym, df, a in data:
            last = -10 ** 9
            for i in range(EVAL_FROM, len(df)):
                st, sig = core_stage(a, i)
                funnel[:st + 1] += 1
                if sig is None or i - last < COOLDOWN:
                    continue
                last = i
                signals += 1
                t = E.try_trade("CORE", a, df, sym, i,
                                {"trigger": sig["trigger"], "candle": sig["mode"]}, sig["sl_c"], counts)
                if t:
                    high52 = a["h"][max(0, i - 251):i + 1].max()
                    t["mode"] = sig["mode"]
                    t["vol_x"] = round(sig["vol_x"], 2)
                    t["vol_ok"] = bool(sig["vol_x"] >= VOL_MIN)
                    t["score"] = round(sig["range_x"] * sig["vol_x"], 2)
                    t["dist52"] = round((high52 - a["c"][i]) / high52 * 100, 2)
                    rows.append(t)
    return funnel, signals, counts, pd.DataFrame(rows)


def nifty_ok():
    try:
        d = yf.download("^NSEI", period="5y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = swing._fix_daily_index(d.dropna(subset=["Close"]).copy())
        s = d["Close"] > d["Close"].ewm(span=200, adjust=False).mean()

        def ok(day):
            v = s.asof(pd.Timestamp(day))
            return bool(v) if pd.notna(v) else False
        return ok
    except Exception as e:
        print("Nifty data unavailable:", e)
        return None


def fmt(t):
    if t is None or len(t) == 0:
        return "no trades"
    d = pd.to_datetime(t.entry_date)

    def part(g):
        return f"{len(g)} / {g.ret_pct.mean():+.2f}%" if len(g) else "0"
    return (f"{len(t)} trades, win {(t.ret_pct > 0).mean() * 100:.0f}%, avg {t.ret_pct.mean():+.2f}% "
            f"(net {t.ret_pct.mean() - COST_PCT:+.2f}%) | 22-24: {part(t[d < SPLIT])} "
            f"| 25-26: {part(t[d >= SPLIT])}")


def result_lines(t, ok, full=True):
    if len(t) == 0:
        return ["   no trades"]
    cap = lambda g: g.sort_values("score", ascending=False).groupby("signal_date").head(3) if len(g) else g
    mk = (lambda g: g[g.signal_date.map(ok)]) if ok else (lambda g: g.iloc[0:0])
    L = [f"   0 Core rules only: {fmt(t)}",
         f"   1 + volume >= 1.2x: {fmt(t[t.vol_ok])}"]
    if full:
        L += [f"   2 + resistance filter: {fmt(t[t.res_ok])}"]
    L += [f"   3 + Nifty > 200 EMA: {fmt(mk(t)) if ok else 'Nifty data unavailable'}",
          f"   4 + max 3 per day: {fmt(cap(t))}"]
    if full:
        L += [f"   5 + volume + Nifty: {fmt(mk(t[t.vol_ok])) if ok else 'n/a'}",
              f"   6 ALL (volume+resistance+Nifty+cap): "
              f"{fmt(cap(mk(t[t.vol_ok & t.res_ok]))) if ok else 'n/a'}",
              f"   Pullback only: {fmt(t[t['mode'] == 'PB'])}",
              f"   Box only: {fmt(t[t['mode'] == 'BX'])}"]
    return L


def check_known(sym_dates):
    """Does the strategy find YOUR trades? Window: 6 sessions before to 2 after the date."""
    out = []
    syms = [s + ".NS" for s, _ in sym_dates]
    frames = B.download_long(syms)
    data = [(k.replace(".NS", ""), df, E.arrays(df)) for k, df in frames.items()]
    _, _, _, trades = run_core(data, 7.5)
    for sym, day in sym_dates:
        key = sym + ".NS"
        if key not in frames:
            out.append(f"{sym} ({day}): no price data found - check the symbol")
            continue
        df = frames[key]
        a = next(x[2] for x in data if x[0] == sym)
        p = int(df.index.searchsorted(pd.Timestamp(day)))
        best, best_i, found = -1, None, []
        for i in range(max(EVAL_FROM, p - 6), min(len(df) - 1, p + 2) + 1):
            st, sig = core_stage(a, i)
            if sig:
                found.append((i, sig["mode"]))
            if st > best:
                best, best_i = st, i
        if found:
            i, mode = found[0]
            sd = df.index[i].date()
            tr = trades[(trades.symbol == sym) & (pd.to_datetime(trades.signal_date) == pd.Timestamp(sd))] \
                if len(trades) else trades
            if len(tr):
                r = tr.iloc[0]
                out.append(f"{sym} ({day}): FOUND [{mode}] signal {sd} -> entry {r.entry:.2f}, SL {r.sl:.2f}, "
                           f"exit {r.reason} {r.ret_pct:+.1f}%")
            else:
                out.append(f"{sym} ({day}): signal [{mode}] on {sd}, but NO TRADE (stop too wide, or no "
                           f"break of the high within 3 days)")
        else:
            out.append(f"{sym} ({day}): NOT FOUND. Closest day {df.index[best_i].date()} stopped at: "
                       f"{STAGE_TEXT[best]} [{explain(a, best_i)}]")
    return out


def main():
    uni = load_universe()
    frames = B.download_long(uni["ticker"].tolist())
    if not frames:
        safe_send("❌ Core backtest: could not download price data.")
        sys.exit(1)
    data = [(s.replace(".NS", ""), df, E.arrays(df)) for s, df in frames.items()]
    ok = nifty_ok()
    first = min(df.index[EVAL_FROM].date() for df in frames.values())
    last = max(df.index[-1].date() for df in frames.values())
    L = [f"🧪 CORE STRATEGY BACKTEST - Nifty 500 ({first} to {last})",
         "Entry = break of signal-candle high (3 days). Target 1:1.5. Hold up to 30 days.",
         "'net' = after an assumed 0.3% cost per trade. Format: trades, win, avg | 22-24: n / avg | 25-26: n / avg",
         "", "DO YOUR TRADES GET FOUND?"]
    L += check_known(KNOWN)
    all_rows = []
    for stop in (7.5, 5.0):
        funnel, signals, counts, trades = run_core(data, stop)
        if stop == 7.5:
            L += ["", f"FUNNEL ({funnel[0]:,} stock-days)"]
            for s, nm in enumerate(FUNNEL_NAMES, start=1):
                L.append(f"{s}. {nm}: {funnel[s]:,} ({funnel[s] / max(funnel[s - 1], 1) * 100:.1f}% of previous)")
            L.append(f"Signals: {signals}")
        L += ["", f"RESULTS - MAX STOP {stop}% (signals with stop inside limit: {counts['CORE']['risk_ok']})"]
        L += result_lines(trades, ok, full=(stop == 7.5))
        if len(trades):
            all_rows.append(trades.assign(max_stop=stop))
    L += ["", "Keep a rule only if it helps in BOTH periods. Same caveats as before "
              "(today's Nifty 500 list, mostly rising market).",
          "Trade list: download core_trades.csv from the run page (Artifacts)."]
    if all_rows:
        pd.concat(all_rows).to_csv("core_trades.csv", index=False)
    safe_send("\n".join(L))


if __name__ == "__main__":
    main()
