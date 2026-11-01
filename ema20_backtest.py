"""
ema20_backtest.py - backtest of two 20-EMA strategies on ~4 years of daily data.

S1  Pullback to the 20 EMA: uptrend (close > 20 EMA > 200 EMA, both rising), the
    stock ran up and pulled back to the 20 EMA (it did NOT climb up from below),
    and a bullish candle forms there (strong green, hammer/doji with lower wick,
    or engulfing, closing in the top 60% of its range).
S2  Box near the 20 EMA: at least 5 quiet-volume days in a tight box near the
    20 EMA, then a green candle closing above the box with volume >= 1.3x.
    Two stop-loss versions: below the signal candle ("candle") or the box ("box").

Entry: next 1-3 sessions when price breaks above the signal candle's HIGH.
SL: S1 = lower of signal-candle and previous-candle low; S2 as above (at least
1% below entry). Target 1:1.5. Time exit after 15 trading days.
Resistance filter: nearest resistance above the entry (52-week high or a
confirmed swing high of the last year) must be at least 1.5 x risk away.
"""
import sys

import numpy as np
import pandas as pd

import swing
import swing_backtest as B
from intraday import load_universe, safe_send

# ===================== SETTINGS YOU CAN CHANGE =====================
RR = 1.5
BOX_DAYS = 5
BOX_MAX_RANGE_PCT = 5
VOL_COMPRESSION = 0.8
VOL_EXPANSION = 1.3
MIN_RUNUP_PCT = 4          # S1: recent high at least 4% above the 20 EMA
TOUCH_PCT = 1              # S1: candle low within 1% above the 20 EMA (or below it)
MIN_RANGE_ATR = 0.6        # signal candle range at least 0.6 x average daily range
MIN_RISK_PCT = 1.0
MAX_RISK_PCT = 6
MAX_EXTEND_PCT = 3
WATCH_DAYS = 3
COOLDOWN = 5
EVAL_FROM = 260
SPLIT = pd.Timestamp("2025-01-01")
# ====================================================================

TITLES = {"S1": "S1 Pullback to 20 EMA",
          "S2-candle": "S2 Box near 20 EMA, SL = signal candle low",
          "S2-box": "S2 Box near 20 EMA, SL = box low"}
S1_STAGES = ["Close > 20 EMA > 200 EMA", "Both EMAs rising", "Ran up, then pulled back (not from below)",
             "Candle touches the 20 EMA and closes above", "Bullish candle type", "Candle not tiny"]
S2_STAGES = ["Close > 20 EMA > 200 EMA", "Both EMAs rising", "Tight 5-day box (<= 5%)",
             "Box near the 20 EMA", "Volume compressed in box", "Green candle closes above box",
             "Volume >= 1.3x"]


def arrays(df):
    df = swing.add_emas(df)
    o, h, l, c, v = (df[x].values for x in ("Open", "High", "Low", "Close", "Volume"))
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).rolling(14).mean().values
    return {"o": o, "h": h, "l": l, "c": c, "v": v, "e20": df["EMA20"].values,
            "e200": df["EMA200"].values, "atr": atr}


def candle_type(a, i):
    o, h, l, c = a["o"], a["h"], a["l"], a["c"]
    rng = h[i] - l[i]
    if rng <= 0 or (c[i] - l[i]) / rng < 0.6:
        return None
    body = abs(c[i] - o[i])
    lower = min(o[i], c[i]) - l[i]
    if c[i] > o[i] and body >= 0.6 * rng:
        return "STRONG_GREEN"
    if lower >= 2 * body and lower >= 0.5 * rng:
        return "HAMMER"
    if c[i - 1] < o[i - 1] and c[i] > o[i] and c[i] > o[i - 1] and o[i] <= c[i - 1]:
        return "ENGULFING"
    return None


def trend_stage(a, i):
    c, e20, e200 = a["c"], a["e20"], a["e200"]
    if not (c[i] > e20[i] > e200[i]):
        return 0
    if not (e20[i] > e20[i - 5] and e200[i] > e200[i - 20]):
        return 1
    return 2


def s1_stage(a, i):
    st = trend_stage(a, i)
    if st < 2:
        return st, None
    o, h, l, c, e20, atr = a["o"], a["h"], a["l"], a["c"], a["e20"], a["atr"]
    if ((h[i - 10:i].max() / e20[i] - 1) * 100 < MIN_RUNUP_PCT
            or (c[i - 8:i] > e20[i - 8:i]).sum() < 6):
        return 2, None
    if not (l[i] <= e20[i] * (1 + TOUCH_PCT / 100) and c[i] >= e20[i] and l[i] >= e20[i] * 0.97):
        return 3, None
    ct = candle_type(a, i)
    if ct is None:
        return 4, None
    if h[i] - l[i] < MIN_RANGE_ATR * atr[i]:
        return 5, None
    return 6, {"candle": ct, "trigger": h[i], "sls": {"S1": min(l[i], l[i - 1])}}


def s2_stage(a, i):
    st = trend_stage(a, i)
    if st < 2:
        return st, None
    o, h, l, c, v, e20 = a["o"], a["h"], a["l"], a["c"], a["v"], a["e20"]
    bx = slice(i - BOX_DAYS, i)
    bh, bl = h[bx].max(), l[bx].min()
    if (bh - bl) / bl * 100 > BOX_MAX_RANGE_PCT:
        return 2, None
    if not (e20[i - 1] * 0.97 <= bl <= e20[i - 1] * 1.02):
        return 3, None
    prior = v[i - BOX_DAYS - 20:i - BOX_DAYS].mean()
    if prior <= 0 or v[bx].mean() >= VOL_COMPRESSION * prior:
        return 4, None
    ct = candle_type(a, i)
    if not (ct and c[i] > o[i] and c[i] > bh):
        return 5, None
    avg20 = v[i - 20:i].mean()
    if avg20 <= 0 or v[i] < VOL_EXPANSION * avg20:
        return 6, None
    return 7, {"candle": ct, "trigger": h[i], "sls": {"S2-candle": l[i], "S2-box": bl}}


def nearest_resistance(h, i, price):
    """Nearest resistance above `price`: 52-week high or a confirmed swing high
    (higher than the 5 sessions before and after it) within the last year."""
    cands = [h[max(0, i - 251):i + 1].max()]
    for p in range(max(5, i - 250), i - 5):
        if h[p] >= h[p - 5:p].max() and h[p] >= h[p + 1:p + 6].max():
            cands.append(h[p])
    above = [x for x in cands if x > price * 1.001]
    return min(above) if above else None


def resistance_ok(h, i, trigger, sl):
    lvl = nearest_resistance(h, i, trigger)
    return lvl is None or (lvl - trigger) >= RR * (trigger - sl)


def try_trade(name, a, df, sym, i, sig, sl_c, counts):
    """Plans and simulates one trade from a signal. Returns a trade row or None."""
    o, h, l, c = a["o"], a["h"], a["l"], a["c"]
    n = len(c)
    trig = sig["trigger"]
    sl0 = min(sl_c, trig * (1 - MIN_RISK_PCT / 100))
    if (trig - sl0) / trig * 100 > MAX_RISK_PCT:
        return None
    counts[name]["risk_ok"] += 1
    res_ok = resistance_ok(h, i, trig, sl0)
    counts[name]["res_ok"] += int(res_ok)
    for k in range(i + 1, min(i + WATCH_DAYS, n - 1) + 1):
        if k > i + 1 and (c[i + 1:k] < sl0).any():
            return None
        if h[k] >= trig:
            entry = max(o[k], trig)
            if entry > trig * (1 + MAX_EXTEND_PCT / 100):
                return None
            sl = min(sl_c, entry * (1 - MIN_RISK_PCT / 100))
            if (entry - sl) / entry * 100 > MAX_RISK_PCT:
                return None
            tgt = entry + RR * (entry - sl)
            res = B.exit_trade(o, h, l, c, k, k, sl, tgt, same_day=True)
            if res is None:
                return None
            row = B.trade_row(name, sym, df, i, k, entry, sl, tgt, res)
            row["res_ok"] = bool(res_ok)
            row["candle"] = sig["candle"]
            return row
    return None


def run_backtest(frames):
    funnel = {"S1": np.zeros(len(S1_STAGES) + 1, dtype=np.int64),
              "S2": np.zeros(len(S2_STAGES) + 1, dtype=np.int64)}
    counts = {k: {"risk_ok": 0, "res_ok": 0} for k in TITLES}
    signals = {"S1": 0, "S2": 0}
    trades = []
    for sym_t, df in frames.items():
        sym = sym_t.replace(".NS", "")
        a = arrays(df)
        last = {"S1": -10 ** 9, "S2": -10 ** 9}
        for i in range(EVAL_FROM, len(df)):
            for key, fn in (("S1", s1_stage), ("S2", s2_stage)):
                st, sig = fn(a, i)
                funnel[key][:st + 1] += 1
                if sig is None or i - last[key] < COOLDOWN:
                    continue
                last[key] = i
                signals[key] += 1
                for name, sl_c in sig["sls"].items():
                    t = try_trade(name, a, df, sym, i, sig, sl_c, counts)
                    if t:
                        trades.append(t)
    return funnel, signals, counts, pd.DataFrame(trades)


def line(label, t):
    s = B.summarize(t)
    if s is None:
        return f"   {label}: no trades"
    pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return (f"   {label}: {s['n']} trades | win {s['win_rate']:.0f}% | "
            f"avg {s['avg_ret']:+.2f}% | PF {pf}")


def build_report(frames, funnel, signals, counts, trades):
    first = min(df.index[EVAL_FROM].date() for df in frames.values())
    last = max(df.index[-1].date() for df in frames.values())
    years = max((pd.Timestamp(last) - pd.Timestamp(first)).days / 365.25, 0.1)
    L = [f"🧪 20 EMA STRATEGIES BACKTEST - Nifty 500 ({first} to {last}, {years:.1f} yrs)",
         f"Stocks tested: {len(frames)} | Entry = break of signal-candle high within 3 days | "
         f"SL / 1:1.5 target / 15-day exit",
         "Break-even win rate at 1:1.5 is about 40%. PF above 1.0 = profitable.", ""]
    for key, stages in (("S1", S1_STAGES), ("S2", S2_STAGES)):
        L.append(f"FUNNEL {key} (stock-days still passing)")
        for s, nm in enumerate(stages, start=1):
            prev, cur = funnel[key][s - 1], funnel[key][s]
            L.append(f"{s}. {nm}: {cur:,} ({(cur / prev * 100 if prev else 0):.1f}% of prev)")
        L.append(f"Signals (after cooldown): {signals[key]} = about {signals[key] / (years * 12):.1f} per month")
        L.append("")
    for name, title in TITLES.items():
        t = trades[trades.entry_type == name] if len(trades) else trades
        L.append(title)
        L.append(f"   Risk within 1-6%: {counts[name]['risk_ok']} signals | "
                 f"resistance space OK: {counts[name]['res_ok']}")
        for label, sub in (("WITH resistance filter", t[t.res_ok] if len(t) else t),
                           ("WITHOUT resistance filter", t)):
            L.append(line(label + " (all)", sub))
            if len(sub):
                d = pd.to_datetime(sub.entry_date)
                L.append(line("   2022-24", sub[d < SPLIT]))
                L.append(line("   2025-26", sub[d >= SPLIT]))
        L.append("")
    L += ["Caveats: today's Nifty 500 list, mostly rising market, no costs. Few trades = weak evidence.",
          "Trade list: download ema20_trades.csv from the run page (Artifacts)."]
    return "\n".join(L)


def main():
    uni = load_universe()
    frames = B.download_long(uni["ticker"].tolist())
    if not frames:
        safe_send("❌ 20 EMA backtest: could not download price data.")
        sys.exit(1)
    funnel, signals, counts, trades = run_backtest(frames)
    trades.to_csv("ema20_trades.csv", index=False)
    safe_send(build_report(frames, funnel, signals, counts, trades))


if __name__ == "__main__":
    main()
