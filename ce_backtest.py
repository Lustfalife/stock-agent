"""
ce_backtest.py - backtest of the "Compression -> Expansion near the 52-week high" strategy
(daily chart, Nifty 500, buy only), and how each fix changes the result.

BASE SIGNAL (the old evening scan): close above the 20 and 200 EMA, close within 5% of the
52-week high, volume compressed in the last 10 days (5-day average volume below 70% of the
20-day average at least once), and today's volume at least 1.5x the 20-day average.

FIXES TESTED, one after the other:
  1  20 EMA above the 200 EMA
  2  signal candle is green and closes in the top half of its range   (= NEW BASE)
  3  "move not already done": today's gain <= 3%
  4  close not more than 8% above the 20 EMA
  5  not up more than 10% in the last 5 days
Entry: break of the signal candle's high within 3 days (skip if it gaps more than 3% above).
SL: lowest low of the last 5 candles (at least 1% below entry, skip if more than 7.5%).
Target 1:1.5. Hold up to 30 days. Same-day stop and entry: the stop is counted.
"""
import sys

import numpy as np
import pandas as pd

import swing
import swing_backtest as B
import ema20_backtest as E
import core_backtest as C
from intraday import load_universe, safe_send

EVAL_FROM = 260
COOLDOWN = 5
NEAR_HIGH_PCT = 5
COMPRESS_RATIO = 0.7
COMPRESS_LOOKBACK = 10
VOL_EXPANSION = 1.5
MAX_DAY_GAIN = 3.0
MAX_ABOVE_EMA20 = 8.0
MAX_GAIN_5D = 10.0
SPLIT = C.SPLIT


def arrays_ce(df):
    a = E.arrays(df)
    v = pd.Series(a["v"])
    a["r"] = (v.rolling(5).mean() / v.rolling(20).mean()).values
    return a


def ce_signal(a, i):
    """The old scan's signal on session i, plus features for the fixes. None if it doesn't match."""
    o, h, l, c, v, e20, e200, r = (a[k] for k in ("o", "h", "l", "c", "v", "e20", "e200", "r"))
    if not (c[i] > e20[i] and c[i] > e200[i]):
        return None
    high52 = h[i - 251:i + 1].max()
    if (high52 - c[i]) / high52 * 100 > NEAR_HIGH_PCT:
        return None
    if not (np.nanmin(r[i - COMPRESS_LOOKBACK:i]) < COMPRESS_RATIO):
        return None
    avg20 = v[i - 20:i].mean()
    if avg20 <= 0 or v[i] < VOL_EXPANSION * avg20:
        return None
    rng = h[i] - l[i]
    return {"trigger": h[i], "sl_c": l[i - 4:i + 1].min(), "vol_x": v[i] / avg20,
            "ema_order": bool(e20[i] > e200[i]),
            "green_top": bool(c[i] > o[i] and rng > 0 and (c[i] - l[i]) / rng >= 0.5),
            "day_gain": (c[i] / c[i - 1] - 1) * 100, "above20": (c[i] / e20[i] - 1) * 100,
            "gain5": (c[i] / c[i - 5] - 1) * 100,
            "dist52": (high52 - c[i]) / high52 * 100}


def run(data):
    counts = {"CE": {"risk_ok": 0, "res_ok": 0}}
    rows, signals = [], 0
    with C.settings(7.5):
        for sym, df, a in data:
            last = -10 ** 9
            for i in range(EVAL_FROM, len(df)):
                sig = ce_signal(a, i)
                if sig is None or i - last < COOLDOWN:
                    continue
                last = i
                signals += 1
                t = E.try_trade("CE", a, df, sym, i, {"trigger": sig["trigger"], "candle": "CE"},
                                sig["sl_c"], counts)
                if t:
                    t.update({k: (bool(v) if isinstance(v, (bool, np.bool_)) else round(float(v), 2))
                              for k, v in sig.items() if k not in ("trigger", "sl_c")})
                    rows.append(t)
    return signals, pd.DataFrame(rows)


def variants(t):
    new = t[t.ema_order & t.green_top]
    return [
        ("0 Old scan as it was", t),
        ("1 + 20 EMA above 200 EMA", t[t.ema_order]),
        ("2 + green candle, close in top half  (NEW BASE)", new),
        ("3 NEW BASE + today's gain <= 3%", new[new.day_gain <= MAX_DAY_GAIN]),
        ("4 NEW BASE + close <= 8% above 20 EMA", new[new.above20 <= MAX_ABOVE_EMA20]),
        ("5 NEW BASE + up <= 10% in 5 days", new[new.gain5 <= MAX_GAIN_5D]),
        ("6 NEW BASE + all three 'not already moved' rules",
         new[(new.day_gain <= MAX_DAY_GAIN) & (new.above20 <= MAX_ABOVE_EMA20) & (new.gain5 <= MAX_GAIN_5D)]),
    ]


def main():
    uni = load_universe()
    frames = B.download_long(uni["ticker"].tolist())
    if not frames:
        safe_send("❌ CE backtest: could not download price data.")
        sys.exit(1)
    data = [(s.replace(".NS", ""), df, arrays_ce(df)) for s, df in frames.items()]
    signals, t = run(data)
    first = min(df.index[EVAL_FROM].date() for df in frames.values())
    last = max(df.index[-1].date() for df in frames.values())
    years = max((pd.Timestamp(last) - pd.Timestamp(first)).days / 365.25, 0.1)
    L = [f"🧪 COMPRESSION -> EXPANSION BACKTEST - Nifty 500 ({first} to {last})",
         "Entry = break of signal-candle high (3 days). SL = 5-candle swing low (max 7.5%). "
         "Target 1:1.5. Hold up to 30 days.",
         "Format: trades, win, avg (net of 0.3% cost) | 22-24: n / avg | 25-26: n / avg", "",
         f"Old-scan signals: {signals} (about {signals / (years * 12):.1f} per month); "
         f"trades with entry and stop inside the limits: {len(t)}"]
    if len(t):
        new = t[t.ema_order & t.green_top]
        L.append(f"Of the NEW BASE signals: {(new.day_gain > MAX_DAY_GAIN).mean() * 100:.0f}% had already gained "
                 f">3% that day, {(new.above20 > MAX_ABOVE_EMA20).mean() * 100:.0f}% closed >8% above the "
                 f"20 EMA, {(new.gain5 > MAX_GAIN_5D).mean() * 100:.0f}% were up >10% in 5 days.")
        L.append("")
        for name, g in variants(t):
            L.append(f"{name}: {C.fmt(g)}")
        t.to_csv("ce_trades.csv", index=False)
    L += ["", "Keep a rule only if it helps in BOTH periods and after costs. "
              "Caveats: today's Nifty 500 list, mostly rising market.",
          "The same-day volume entry cannot be backtested (no long 5-minute history). "
          "Trade list: ce_trades.csv (Artifacts)."]
    safe_send("\n".join(L))


if __name__ == "__main__":
    main()
