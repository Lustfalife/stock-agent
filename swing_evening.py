"""
swing_evening.py - runs after market close (about 4:30 PM IST).
Finds stocks that broke out of their consolidation box TODAY, sends them as a
watchlist for tomorrow, and logs a paper "DAY_CLOSE" trade for comparison.
Set TEST_ONCE=1 to run on the latest data any time (nothing is logged).
"""
import sys

import intraday
import swing
import tradelog
from intraday import now_ist, safe_send, load_universe

TEST = intraday.TEST_ONCE


def main():
    uni = load_universe()
    sector_of = dict(zip(uni["ticker"], uni["sector"]))
    daily = swing.download_daily(uni["ticker"].tolist())
    if not daily:
        safe_send("❌ Swing scan: could not download daily prices.")
        sys.exit(1)
    last = max(df.index[-1].date() for df in daily.values())
    if not TEST and last != now_ist().date():
        print("No new trading session today (holiday?). Exiting.")
        return

    found = []
    for t, df in daily.items():
        df = swing.add_emas(df[df.index.date <= last])
        b = swing.check_breakout(df, len(df) - 1)
        if b:
            found.append((t, b))
    found.sort(key=lambda x: -x[1]["vol_x"])

    day = last.strftime("%d %b %Y")
    head = "🧪 TEST - " if TEST else ""
    if not found:
        safe_send(f"{head}📋 Swing scan {day}: no breakouts from a consolidation box today.")
        return
    lines = [f"{head}📋 Swing watchlist after {day} ({len(found)} breakout"
             f"{'s' if len(found) != 1 else ''})", ""]
    paper = []
    for t, b in found:
        sym = t.replace(".NS", "")
        lines.append(f"{sym} ({sector_of[t]})")
        lines.append(f"   Close {b['close']:.2f} | Volume {b['vol_x']:.1f}x avg | "
                     f"52w high {b['high52']:.2f} ({b['days_since_high']} days ago)")
        lines.append(f"   Entry alert only if price stays above {b['trigger']:.2f} for 30 min "
                     f"(next {swing.WATCH_DAYS} sessions)")
        plan = swing.plan_levels(b["close"], b["candle_low"])
        if plan:
            sl, tgt, _ = plan
            paper.append({
                "ID": f"{b['date']}|{sym}|SWING|CLOSE", "Date": b["date"], "Time": "15:30",
                "Strategy": "Swing 52w-high box breakout", "Stock": sym, "Sector": sector_of[t],
                "Side": "BUY", "Entry Type": "DAY_CLOSE", "Entry": round(b["close"], 2),
                "SL": round(sl, 2), "Target": round(tgt, 2), "Status": "OPEN",
                "Exit Price": "", "Exit Time": "", "R": "",
                "Notes": "Paper trade at breakout-day close (no alert), for comparison"})
        lines.append("")
    safe_send("\n".join(lines))
    if paper and not TEST:
        tradelog.add_rows("Swing", paper)


if __name__ == "__main__":
    main()
