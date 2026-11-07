"""
live_evening.py - evening signal list for BOTH live swing strategies (Telegram).
Runs after the close. Entry alerts come from live_monitor.py the next morning.
Set TEST_ONCE=1 to run any time (it only sends the list; nothing is logged here).
"""
import datetime as dt
import sys

import swing
import live_core as L
from intraday import now_ist, safe_send, load_universe


def main():
    uni = load_universe()
    sector_of = dict(zip(uni["ticker"], uni["sector"]))
    frames = swing.download_daily(uni["ticker"].tolist())
    if not frames:
        safe_send("❌ Live swing scan: could not download price data.")
        sys.exit(1)
    note = ""
    now = now_ist()
    if now.time() < dt.time(16, 0):                 # ignore today's unfinished candle
        if any(df.index[-1].date() == now.date() for df in frames.values()):
            frames = {t: (df.iloc[:-1] if df.index[-1].date() == now.date() else df)
                      for t, df in frames.items()}
            note = " (market still open - last completed day used)"
    items = L.prepare_items(frames)
    lines, last_date = [], None
    for key in ("PB", "CE"):
        new, waiting, entered, skipped, last_date = L.scan(items, sector_of, key)
        lines += L.section_lines(key, new, waiting, entered, skipped) + [""]
    head = [f"📈 LIVE SWING SIGNALS - after close {last_date.strftime('%d %b %Y')}{note}",
            "Buy only if price breaks above the signal high within the next 3 trading days. "
            "Entry alerts come tomorrow from 9:20 AM.", ""]
    foot = ["Exit at stop or target; otherwise at the close of day 30. Check the chart before trading."]
    safe_send("\n".join(head + lines + foot))


if __name__ == "__main__":
    main()
