"""
swing_variants.py - compares different rule settings on the same 4-year data.
V0 = current rules, V1 = looser entry rules, V2 = V1 + 20-day hold.
Results: Telegram + backtest_variants.csv (download from the run page).
A setting only counts as better if it helps in BOTH 2022-24 and 2025-26.
"""
import sys

import pandas as pd

import swing
import swing_backtest as B
from intraday import load_universe, safe_send

LOOSER = {"MAX_PULLBACK_PCT": 9, "VOL_EXPANSION": 1.2, "MAX_EXTEND_PCT": 4,
          "VOL_COMPRESSION": 0.9, "MAX_DAYS_SINCE_HIGH": 120}

VARIANTS = [
    ("V0 Current rules (7%, 1.5x, 3%, 0.8, 40 days, hold 15)", {}),
    ("V1 Looser (9%, 1.2x, 4%, 0.9, 120 days, hold 15)", LOOSER),
    ("V2 = V1 + hold 20 days", dict(LOOSER, MAX_HOLD_DAYS=20)),
]
SPLIT = pd.Timestamp("2025-01-01")


def run_variant(frames, settings):
    old = {k: getattr(swing, k) for k in settings}
    try:
        for k, v in settings.items():
            setattr(swing, k, v)
        return B.run_backtest(frames)
    finally:
        for k, v in old.items():
            setattr(swing, k, v)


def line(label, t):
    s = B.summarize(t)
    if s is None:
        return f"   {label}: no trades"
    pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return (f"   {label}: {s['n']} trades | win {s['win_rate']:.0f}% | "
            f"avg {s['avg_ret']:+.2f}% | PF {pf}")


def report_variant(name, frames, result, years):
    stage_counts, signals, trades, open_trades = result
    out = [name, f"   Signals {len(signals)} (about {len(signals) / (years * 12):.1f} per month)"]
    if len(trades):
        trades = trades.copy()
        trades["entry_date"] = pd.to_datetime(trades["entry_date"])
        for kind in ("A", "B"):
            t = trades[trades.entry_type == kind]
            out.append(line(f"Entry {kind} all", t))
            out.append(line(f"Entry {kind} 2022-24", t[t.entry_date < SPLIT]))
            out.append(line(f"Entry {kind} 2025-26", t[t.entry_date >= SPLIT]))
    return out


def main():
    uni = load_universe()
    frames = B.download_long(uni["ticker"].tolist())
    if not frames:
        safe_send("❌ Variants: could not download price data.")
        sys.exit(1)
    first = min(df.index[B.EVAL_FROM].date() for df in frames.values())
    last = max(df.index[-1].date() for df in frames.values())
    years = max((pd.Timestamp(last) - pd.Timestamp(first)).days / 365.25, 0.1)
    lines = [f"🧪 SWING VARIANTS - Nifty 500 ({first} to {last})",
             "Entry A = at breakout-day close | Entry B = first touch of breakout high",
             "PF = profit factor (above 1.0 = profitable). Break-even win rate at 1:1.5 is about 40%.", ""]
    all_trades = []
    for name, settings in VARIANTS:
        result = run_variant(frames, settings)
        lines += report_variant(name, frames, result, years) + [""]
        if len(result[2]):
            t = result[2].copy()
            t.insert(0, "variant", name.split(" ")[0])
            all_trades.append(t)
    lines += ["A change counts only if it helps in BOTH periods. Few trades = weak evidence.",
              "Same caveats as before: today's Nifty 500 list, mostly rising market, no costs.",
              "Trade list: download backtest_variants.csv from the run page (Artifacts)."]
    if all_trades:
        pd.concat(all_trades).to_csv("backtest_variants.csv", index=False)
    safe_send("\n".join(lines))


if __name__ == "__main__":
    main()
