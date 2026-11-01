"""
ema20_variants.py - tests changes to the two 20-EMA strategies on ~4 years of data.

BOX strategy (SL = box low): hold 15 vs 30 days, then each loosening on its own
(box range 6%, box up to 4% above the 20 EMA, quiet volume 0.9, entry window 5 days),
all loosenings together, and a market filter (Nifty above its 200 EMA).

S1 pullback strategy (hold 30 days): each improvement on its own (engulfing/hammer
only, 5-15% below the 52-week high, steeper 20 EMA, market filter, max 3 signals a
day) and all of them combined.

A change is only worth keeping if it helps in BOTH 2022-24 and 2025-26.
"""
import sys
import contextlib

import pandas as pd
import yfinance as yf

import swing
import swing_backtest as B
import ema20_backtest as E
from intraday import load_universe, safe_send

SPLIT = E.SPLIT


@contextlib.contextmanager
def settings(hold=15, window=3):
    old = (swing.MAX_HOLD_DAYS, E.WATCH_DAYS)
    swing.MAX_HOLD_DAYS, E.WATCH_DAYS = hold, window
    try:
        yield
    finally:
        swing.MAX_HOLD_DAYS, E.WATCH_DAYS = old


def s2_signal(a, i, box_range, near_pct, vol_comp):
    """Box strategy signal with adjustable settings (SL = box low)."""
    if E.trend_stage(a, i) < 2:
        return None
    o, h, l, c, v, e20 = a["o"], a["h"], a["l"], a["c"], a["v"], a["e20"]
    bx = slice(i - E.BOX_DAYS, i)
    bh, bl = h[bx].max(), l[bx].min()
    if (bh - bl) / bl * 100 > box_range:
        return None
    if not (e20[i - 1] * 0.97 <= bl <= e20[i - 1] * (1 + near_pct / 100)):
        return None
    prior = v[i - E.BOX_DAYS - 20:i - E.BOX_DAYS].mean()
    if prior <= 0 or v[bx].mean() >= vol_comp * prior:
        return None
    ct = E.candle_type(a, i)
    if not (ct and c[i] > o[i] and c[i] > bh):
        return None
    avg20 = v[i - 20:i].mean()
    if avg20 <= 0 or v[i] < E.VOL_EXPANSION * avg20:
        return None
    return {"candle": ct, "trigger": h[i], "sls": {"S2-box": bl}}


def run_box(data, box_range=5, near_pct=2, vol_comp=0.8, hold=15, window=3):
    counts = {"S2-box": {"risk_ok": 0, "res_ok": 0}}
    rows = []
    with settings(hold, window):
        for sym, df, a in data:
            last = -10 ** 9
            for i in range(E.EVAL_FROM, len(df)):
                sig = s2_signal(a, i, box_range, near_pct, vol_comp)
                if sig is None or i - last < E.COOLDOWN:
                    continue
                last = i
                t = E.try_trade("S2-box", a, df, sym, i, sig, sig["sls"]["S2-box"], counts)
                if t:
                    rows.append(t)
    return pd.DataFrame(rows)


def run_s1(data, hold=30):
    counts = {"S1": {"risk_ok": 0, "res_ok": 0}}
    rows = []
    with settings(hold, 3):
        for sym, df, a in data:
            last = -10 ** 9
            for i in range(E.EVAL_FROM, len(df)):
                st, sig = E.s1_stage(a, i)
                if sig is None or i - last < E.COOLDOWN:
                    continue
                last = i
                t = E.try_trade("S1", a, df, sym, i, sig, sig["sls"]["S1"], counts)
                if t:
                    high52 = a["h"][max(0, i - 251):i + 1].max()
                    t["dist52"] = round((high52 - a["c"][i]) / high52 * 100, 2)
                    t["slope10"] = round((a["e20"][i] / a["e20"][i - 10] - 1) * 100, 2)
                    rows.append(t)
    return pd.DataFrame(rows)


def nifty_ok():
    """Returns a function date -> True if Nifty closed above its 200 EMA (or None)."""
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
    return (f"{len(t)} trades, win {(t.ret_pct > 0).mean() * 100:.0f}%, avg {t.ret_pct.mean():+.2f}%"
            f" | 22-24: {part(t[d < SPLIT])} | 25-26: {part(t[d >= SPLIT])}")


def top3(g):
    return g.sort_values("dist52").groupby("signal_date").head(3) if len(g) else g


def s1_lines(t, ok):
    if len(t) == 0:
        return ["No S1 trades."]
    L = [f"0 Base (every signal): {fmt(t)}"]
    eh = t[t.candle.isin(["ENGULFING", "HAMMER"])]
    L.append(f"1 Engulfing + hammer only: {fmt(eh)}")
    L.append(f"2 5-15% below 52w high: {fmt(t[(t.dist52 >= 5) & (t.dist52 <= 15)])}")
    L.append(f"3 8-15% below 52w high: {fmt(t[(t.dist52 >= 8) & (t.dist52 <= 15)])}")
    L.append(f"4 Steeper 20 EMA (2%+ in 10 days): {fmt(t[t.slope10 >= 2])}")
    combo = eh[(eh.dist52 >= 5) & (eh.dist52 <= 15) & (eh.slope10 >= 2)]
    if ok is not None:
        L.append(f"5 Market filter (Nifty > 200 EMA): {fmt(t[t.signal_date.map(ok)])}")
        combo = combo[combo.signal_date.map(ok)]
    else:
        L.append("5 Market filter: Nifty data unavailable")
    L.append(f"6 Max 3 per day (closest to 52w high): {fmt(top3(t))}")
    combo = top3(combo)
    L.append(f"7 COMBO (1+2+4+5+6): {fmt(combo)}")
    L.append(f"8 COMBO + resistance filter: {fmt(combo[combo.res_ok] if len(combo) else combo)}")
    return L


def main():
    uni = load_universe()
    frames = B.download_long(uni["ticker"].tolist())
    if not frames:
        safe_send("❌ Variants: could not download price data.")
        sys.exit(1)
    data = [(s.replace(".NS", ""), df, E.arrays(df)) for s, df in frames.items()]
    ok = nifty_ok()
    first = min(df.index[E.EVAL_FROM].date() for df in frames.values())
    last = max(df.index[-1].date() for df in frames.values())

    box_variants = [
        ("A Base, hold 15 (reference)", dict(hold=15)),
        ("B Hold 30", dict(hold=30)),
        ("C B + box range 6%", dict(hold=30, box_range=6)),
        ("D B + box up to 4% above 20 EMA", dict(hold=30, near_pct=4)),
        ("E B + quiet volume 0.9", dict(hold=30, vol_comp=0.9)),
        ("F B + entry window 5 days", dict(hold=30, window=5)),
        ("G B + all four loosenings", dict(hold=30, box_range=6, near_pct=4, vol_comp=0.9, window=5)),
    ]
    L = [f"🧪 20 EMA VARIANTS - Nifty 500 ({first} to {last})",
         "Entry = break of signal-candle high. Costs not included (take off about 0.3% per trade).",
         "Format: trades, win rate, average return | 22-24: trades / avg | 25-26: trades / avg",
         "Keep a change only if it beats the reference in BOTH periods.", "",
         "BOX STRATEGY (SL = box low)"]
    all_rows = []
    B_trades = None
    for name, kw in box_variants:
        t = run_box(data, **kw)
        if name.startswith("B "):
            B_trades = t
        L.append(name)
        L.append(f"   no resistance filter: {fmt(t)}")
        L.append(f"   with resistance filter: {fmt(t[t.res_ok] if len(t) else t)}")
        if len(t):
            all_rows.append(t.assign(variant=name[0]))
        print(name, "done")
    if ok is not None and B_trades is not None and len(B_trades):
        m = B_trades[B_trades.signal_date.map(ok)]
        L.append("H B + market filter (Nifty > 200 EMA)")
        L.append(f"   no resistance filter: {fmt(m)}")
        L.append(f"   with resistance filter: {fmt(m[m.res_ok] if len(m) else m)}")
    L += ["", "S1 PULLBACK (hold 30 days, no resistance filter unless stated)"]
    s1 = run_s1(data, hold=30)
    L += s1_lines(s1, ok)
    if len(s1):
        all_rows.append(s1.assign(variant="S1"))
    L += ["", "Caveats: today's Nifty 500 list, mostly rising market. Few trades = weak evidence.",
          "Trade list: download ema20_variants.csv from the run page (Artifacts)."]
    if all_rows:
        pd.concat(all_rows).to_csv("ema20_variants.csv", index=False)
    safe_send("\n".join(L))


if __name__ == "__main__":
    main()
