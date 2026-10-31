"""
swing_backtest.py - tests the swing box-breakout rules on ~4 years of daily data.

ENTRY A: at the breakout-day close (exact on daily data).
ENTRY B: first touch of the breakout high within 3 sessions (no 30-minute filter,
         because 5-minute history is not available that far back).
SL / target / 15-day exit as in swing.py. SL and target in the same day = SL.
Gaps through a level are filled at the open.
Results go to Telegram and to backtest_trades.csv (download from the run page).
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

import swing
from intraday import load_universe, safe_send

COOLDOWN = swing.BOX_DAYS          # no new signal for a stock within 10 sessions
EVAL_FROM = 260                    # first session number that can be evaluated

STAGES = ["Above 20 & 200 EMA", "52w high made 10-40 days ago", "Pullback since high <= 7%",
          "Box range <= 6%", "Box top within 5% of high", "Volume compressed in box",
          "Breakout volume >= 1.5x", "Close above box, not >3% over", "Risk 1.5%-6% ok"]


def download_long(tickers, period="5y"):
    frames = {}
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i + 50]
        data = None
        for _ in range(2):
            try:
                data = yf.download(batch, period=period, interval="1d", group_by="ticker",
                                   auto_adjust=True, progress=False, threads=True)
                break
            except Exception as e:
                print("download error:", e)
                time.sleep(5)
        if data is None:
            continue
        if len(batch) == 1 and isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        for t in batch:
            try:
                df = data if len(batch) == 1 else data[t]
                df = swing._fix_daily_index(df.dropna(subset=["Close"]).copy())
                if len(df) >= 300:
                    frames[t] = df
            except Exception:
                continue
        print(f"downloaded {min(i + 50, len(tickers))}/{len(tickers)}")
        time.sleep(1)
    return frames


def analyze(a, i):
    """Number of rules (0-9) the session i passes, in the same order as swing.check_breakout."""
    c, h, l, v, e20, e200 = a
    if not (c[i] > e20[i] and c[i] > e200[i]):
        return 0
    seg = h[i - 252:i]
    pos = len(seg) - 1 - int(seg[::-1].argmax())
    high52 = seg[pos]
    ah = i - 252 + pos
    if not (swing.MIN_DAYS_SINCE_HIGH <= i - ah <= swing.MAX_DAYS_SINCE_HIGH):
        return 1
    if (high52 - l[ah:i].min()) / high52 * 100 > swing.MAX_PULLBACK_PCT:
        return 2
    bh, bl = h[i - swing.BOX_DAYS:i].max(), l[i - swing.BOX_DAYS:i].min()
    if (bh - bl) / bl * 100 > swing.MAX_BOX_RANGE_PCT:
        return 3
    if (high52 - bh) / high52 * 100 > swing.BOX_NEAR_HIGH_PCT:
        return 4
    prior = v[i - swing.BOX_DAYS - 50:i - swing.BOX_DAYS].mean()
    if prior <= 0 or v[i - swing.BOX_DAYS:i].mean() >= swing.VOL_COMPRESSION * prior:
        return 5
    avg20 = v[i - 20:i].mean()
    if avg20 <= 0 or v[i] / avg20 < swing.VOL_EXPANSION:
        return 6
    if c[i] <= bh or c[i] > bh * (1 + swing.MAX_EXTEND_PCT / 100):
        return 7
    if swing.plan_levels(h[i], l[i]) is None:
        return 8
    return 9


def exit_trade(o, h, l, c, first, entry_idx, sl, tgt, same_day):
    """Walks forward from `first`. Returns (reason, price, index) or None if still open."""
    n = len(c)
    last = entry_idx + swing.MAX_HOLD_DAYS
    for k in range(first, min(last, n - 1) + 1):
        hit_sl, hit_tg = l[k] <= sl, h[k] >= tgt
        if hit_sl:                                   # SL wins if both are hit
            return "SL", (sl if (same_day and k == first) else min(o[k], sl)), k
        if hit_tg:
            return "TARGET", (tgt if (same_day and k == first) else max(o[k], tgt)), k
    if last <= n - 1:
        return "TIME_EXIT", c[last], last
    return None


def trade_row(kind, sym, df, sig_i, entry_i, entry, sl, tgt, res):
    reason, price, k = res
    return {"entry_type": kind, "symbol": sym, "signal_date": df.index[sig_i].date(),
            "entry_date": df.index[entry_i].date(), "entry": round(entry, 2), "sl": round(sl, 2),
            "target": round(tgt, 2), "exit_date": df.index[k].date(), "exit": round(float(price), 2),
            "reason": reason, "ret_pct": round((price - entry) / entry * 100, 2),
            "R": round((price - entry) / (entry - sl), 2)}


def run_backtest(frames):
    stage_counts = np.zeros(10, dtype=np.int64)       # stock-days reaching >= s rules passed
    trades, signals, open_trades = [], [], 0
    for sym_t, df in frames.items():
        sym = sym_t.replace(".NS", "")
        df = swing.add_emas(df)
        a = (df["Close"].values, df["High"].values, df["Low"].values, df["Volume"].values,
             df["EMA20"].values, df["EMA200"].values)
        o = df["Open"].values
        c, h, l = a[0], a[1], a[2]
        n = len(df)
        last_sig = -10 ** 9
        for i in range(EVAL_FROM, n):
            st = analyze(a, i)
            stage_counts[:st + 1] += 1
            if st < 9 or i - last_sig < COOLDOWN:
                continue
            b = swing.check_breakout(df, i)
            if b is None:
                print("warning: funnel and check_breakout disagree", sym, df.index[i].date())
                continue
            last_sig = i
            signals.append((sym, df.index[i].date()))
            # ---- Entry A: breakout-day close
            plan = swing.plan_levels(c[i], l[i])
            if plan:
                sl, tgt, _ = plan
                res = exit_trade(o, h, l, c, i + 1, i, sl, tgt, same_day=False)
                if res:
                    trades.append(trade_row("A", sym, df, i, i, c[i], sl, tgt, res))
                else:
                    open_trades += 1
            # ---- Entry B: first touch of the breakout high within 3 sessions
            trig, clow = h[i], l[i]
            for k in range(i + 1, min(i + swing.WATCH_DAYS, n - 1) + 1):
                if k > i + 1 and (c[i + 1:k] < clow).any():
                    break
                if h[k] >= trig:
                    entry = max(o[k], trig)
                    if entry > trig * (1 + swing.MAX_EXTEND_PCT / 100):
                        break
                    p = swing.plan_levels(entry, clow)
                    if p:
                        sl, tgt, _ = p
                        res = exit_trade(o, h, l, c, k, k, sl, tgt, same_day=True)
                        if res:
                            trades.append(trade_row("B", sym, df, i, k, entry, sl, tgt, res))
                    break
    return stage_counts, signals, pd.DataFrame(trades), open_trades


def summarize(t):
    if t.empty:
        return None
    t = t.sort_values("entry_date")
    wins, losses = t[t.ret_pct > 0], t[t.ret_pct <= 0]
    streak = best = 0
    for r in t.ret_pct:
        streak = streak + 1 if r <= 0 else 0
        best = max(best, streak)
    gl = -losses.ret_pct.sum()
    return {"n": len(t), "target": int((t.reason == "TARGET").sum()), "sl": int((t.reason == "SL").sum()),
            "time": int((t.reason == "TIME_EXIT").sum()), "win_rate": len(wins) / len(t) * 100,
            "avg_ret": t.ret_pct.mean(), "avg_win": wins.ret_pct.mean() if len(wins) else 0.0,
            "avg_loss": losses.ret_pct.mean() if len(losses) else 0.0,
            "pf": (wins.ret_pct.sum() / gl) if gl > 0 else float("inf"),
            "avg_r": t.R.mean(), "streak": best}


def block(title, s):
    if s is None:
        return f"{title}\n   No closed trades."
    pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return (f"{title}\n"
            f"   Trades {s['n']} | Target {s['target']} | SL {s['sl']} | Time exit {s['time']}\n"
            f"   Win rate (profit) {s['win_rate']:.0f}% | Avg return/trade {s['avg_ret']:+.2f}%\n"
            f"   Avg win {s['avg_win']:+.1f}% | Avg loss {s['avg_loss']:+.1f}% | Profit factor {pf}\n"
            f"   Avg R {s['avg_r']:+.2f} | Longest losing streak {s['streak']}")


def build_report(frames, stage_counts, signals, trades, open_trades):
    first = min(df.index[EVAL_FROM].date() for df in frames.values())
    last = max(df.index[-1].date() for df in frames.values())
    years = max((pd.Timestamp(last) - pd.Timestamp(first)).days / 365.25, 0.1)
    lines = [f"🧪 SWING BACKTEST - Nifty 500 ({first} to {last}, {years:.1f} years)",
             f"Stocks tested: {len(frames)} | Stock-days checked: {stage_counts[0]:,}", "",
             "RULE FUNNEL (stock-days still passing)"]
    for s, name in enumerate(STAGES, start=1):
        prev, cur = stage_counts[s - 1], stage_counts[s]
        lines.append(f"{s}. {name}: {cur:,} ({(cur / prev * 100 if prev else 0):.1f}% of previous)")
    lines += ["", f"Signals (breakout days): {len(signals)} = about {len(signals) / (years * 12):.1f} per month", ""]
    for kind, title in (("A", "ENTRY A - at breakout-day close"),
                        ("B", "ENTRY B - first touch of breakout high (no 30-min filter)")):
        lines.append(block(title, summarize(trades[trades.entry_type == kind]) if len(trades) else None))
        lines.append("")
    if len(trades):
        ta = trades[trades.entry_type == "A"].copy()
        if len(ta):
            ta["year"] = pd.to_datetime(ta.entry_date).dt.year
            lines.append("ENTRY A by year (trades, win rate, avg return):")
            for y, g in ta.groupby("year"):
                lines.append(f"   {y}: {len(g)} trades, {(g.ret_pct > 0).mean() * 100:.0f}% win, {g.ret_pct.mean():+.2f}%")
            lines.append("")
    lines += [f"Trades still open (excluded): {open_trades}",
              "Notes: needs about 40% wins to break even at 1:1.5. Uses today's Nifty 500 list (survivorship bias) "
              "and a mostly rising market, so real results can be worse. No costs or slippage included.",
              "Full trade list: download backtest_trades.csv from the run page (Artifacts)."]
    return "\n".join(lines)


def main():
    uni = load_universe()
    frames = download_long(uni["ticker"].tolist())
    if not frames:
        safe_send("❌ Backtest: could not download price data.")
        sys.exit(1)
    stage_counts, signals, trades, open_trades = run_backtest(frames)
    trades.to_csv("backtest_trades.csv", index=False)
    safe_send(build_report(frames, stage_counts, signals, trades, open_trades))


if __name__ == "__main__":
    main()
