"""
core_variants.py - tries to improve the core 20-EMA strategy (max stop 7.5%, hold 30 days).

IDEA 2 - stronger stocks: beats Nifty over 3 / 6 months, steeper 20 EMA, near the 52-week high.
IDEA 3 - better candle: bigger candle (1.5x / 2x the last-5 average), close in top 80%.
IDEA 4 - market and sector health: share of Nifty 500 stocks above their own 20 EMA, and the
         share of the stock's own sector above its 20 EMA.
Volume, resistance and the 3-a-day cap are NOT used.

Each idea is tested alone and then combined. Keep a change only if it helps in BOTH
2022-24 and 2025-26 (after costs) - and look for patterns, not one lucky line.
"""
import sys

import numpy as np
import pandas as pd
import yfinance as yf

import swing
import swing_backtest as B
import ema20_backtest as E
import core_backtest as C
from intraday import load_universe, safe_send


def run_features(data):
    """The core strategy trades, with extra information saved for each trade."""
    counts = {"CORE": {"risk_ok": 0, "res_ok": 0}}
    rows = []
    with C.settings(7.5):
        for sym, df, a in data:
            last = -10 ** 9
            c, h, l, e20 = a["c"], a["h"], a["l"], a["e20"]
            for i in range(C.EVAL_FROM, len(df)):
                st, sig = C.core_stage(a, i)
                if sig is None or i - last < C.COOLDOWN:
                    continue
                last = i
                t = E.try_trade("CORE", a, df, sym, i,
                                {"trigger": sig["trigger"], "candle": sig["mode"]}, sig["sl_c"], counts)
                if not t:
                    continue
                high52 = h[max(0, i - 251):i + 1].max()
                rng = h[i] - l[i]
                t.update(mode=sig["mode"], range_x=round(sig["range_x"], 2),
                         close_pos=round((c[i] - l[i]) / rng, 2),
                         slope10=round((e20[i] / e20[i - 10] - 1) * 100, 2),
                         ret63=round((c[i] / c[i - 63] - 1) * 100, 2),
                         ret126=round((c[i] / c[i - 126] - 1) * 100, 2),
                         dist52=round((high52 - c[i]) / high52 * 100, 2))
                rows.append(t)
    return pd.DataFrame(rows)


def nifty_returns():
    """Nifty 3-month and 6-month returns (%) by date, or None."""
    try:
        d = yf.download("^NSEI", period="5y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = swing._fix_daily_index(d.dropna(subset=["Close"]).copy())
        return d["Close"].pct_change(63) * 100, d["Close"].pct_change(126) * 100
    except Exception as e:
        print("Nifty data unavailable:", e)
        return None, None


def breadth_tables(frames, sector_of):
    """% of stocks above their own 20 EMA, for the whole market and for each sector, by date."""
    cols = {}
    for t, df in frames.items():
        cols[t] = (df["Close"] > df["Close"].ewm(span=20, adjust=False).mean()).astype(float)
    above = pd.DataFrame(cols)
    market = above.mean(axis=1) * 100
    sec = pd.Series({t: sector_of.get(t, "?") for t in above.columns})
    sector = (above.T.groupby(sec).mean().T) * 100
    counts = sec.value_counts()
    for name in counts[counts < 5].index:            # too few stocks to judge a sector
        if name in sector.columns:
            sector[name] = np.nan
    return market, sector


def add_context(t, sector_of, market, sector, n63, n126):
    ts = pd.to_datetime(t["signal_date"])
    t = t.copy()
    t["mkt_breadth"] = [market.asof(d) for d in ts]
    secs = [sector_of.get(s + ".NS", "?") for s in t["symbol"]]
    t["sec_breadth"] = [sector[s].asof(d) if s in sector.columns else np.nan for s, d in zip(secs, ts)]
    t["nifty63"] = [n63.asof(d) for d in ts] if n63 is not None else np.nan
    t["nifty126"] = [n126.asof(d) for d in ts] if n126 is not None else np.nan
    return t


def variant_list(t):
    sec_ok = lambda x: t.sec_breadth.isna() | (t.sec_breadth >= x)       # unknown sector = not blocked
    leaders = (t.ret63 > t.nifty63) & (t.slope10 >= 2) & (t.dist52 <= 15)
    candle = (t.range_x >= 1.5) & (t.close_pos >= 0.8)
    market = (t.mkt_breadth >= 50) & sec_ok(50)
    return [
        ("Reference (core rules)", t),
        ("2a Beats Nifty over 3 months", t[t.ret63 > t.nifty63]),
        ("2b Beats Nifty over 6 months", t[t.ret126 > t.nifty126]),
        ("2c 20 EMA rising 2%+ in 10 days", t[t.slope10 >= 2]),
        ("2d 20 EMA rising 3%+ in 10 days", t[t.slope10 >= 3]),
        ("2e Within 15% of 52w high", t[t.dist52 <= 15]),
        ("2f Within 10% of 52w high", t[t.dist52 <= 10]),
        ("3a Candle 1.5x last-5 average", t[t.range_x >= 1.5]),
        ("3b Candle 2x last-5 average", t[t.range_x >= 2.0]),
        ("3c Close in top 80% of range", t[t.close_pos >= 0.8]),
        ("4a Market: 50%+ stocks above 20 EMA", t[t.mkt_breadth >= 50]),
        ("4b Market: 60%+ stocks above 20 EMA", t[t.mkt_breadth >= 60]),
        ("4c Sector: 50%+ above 20 EMA", t[sec_ok(50)]),
        ("4d Sector: 60%+ above 20 EMA", t[sec_ok(60)]),
        ("C1 Leaders (2a + 2c + 2e)", t[leaders]),
        ("C2 Candle (3a + 3c)", t[candle]),
        ("C3 Market + sector (4a + 4c)", t[market]),
        ("C4 All three groups", t[leaders & candle & market]),
    ]


def main():
    uni = load_universe()
    sector_of = dict(zip(uni["ticker"], uni["sector"]))
    frames = B.download_long(uni["ticker"].tolist())
    if not frames:
        safe_send("❌ Core variants: could not download price data.")
        sys.exit(1)
    data = [(s.replace(".NS", ""), df, E.arrays(df)) for s, df in frames.items()]
    trades = run_features(data)
    if trades.empty:
        safe_send("Core variants: no trades found.")
        return
    market, sector = breadth_tables(frames, sector_of)
    n63, n126 = nifty_returns()
    trades = add_context(trades, sector_of, market, sector, n63, n126)
    trades.to_csv("core_variants.csv", index=False)

    first = min(df.index[C.EVAL_FROM].date() for df in frames.values())
    last = max(df.index[-1].date() for df in frames.values())
    L = [f"🧪 CORE VARIANTS - Nifty 500 ({first} to {last})",
         "Max stop 7.5%, target 1:1.5, hold up to 30 days. No volume / resistance / cap rules.",
         "Format: trades, win, avg (net of 0.3% cost) | 22-24: n / avg | 25-26: n / avg",
         "Keep a change only if it helps in BOTH periods. Look for patterns, not one lucky line.", ""]
    for title, sub in (("PULLBACK ONLY", trades[trades["mode"] == "PB"]), ("ALL TRADES (pullback + box)", trades)):
        if title.startswith("ALL"):
            vlist = [v for v in variant_list(sub) if v[0].startswith(("Reference", "C"))]
        else:
            vlist = variant_list(sub)
        L.append(title)
        for name, g in vlist:
            L.append(f"{name}: {C.fmt(g)}")
        L.append("")
    L += ["Caveats: today's Nifty 500 list, mostly rising market. Many variants tested = some will look "
          "good by luck.", "Trade list with all the extra columns: core_variants.csv (Artifacts)."]
    safe_send("\n".join(L))


if __name__ == "__main__":
    main()
