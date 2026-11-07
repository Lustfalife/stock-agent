"""
live_monitor.py - runs 9:20 AM to 3:00 PM IST. Watches the signals from the last 3 sessions of BOTH
live swing strategies and sends the ENTRY alert when price breaks above the signal candle's high.
Every entry is written to the Google Sheet (tab "Pullback" or "CE").
Set TEST_ONCE=1 for one check on the latest data any time (nothing is logged).
"""
import datetime as dt
import time

import intraday
import swing
import tradelog
import ce_backtest as X
import live_core as L
from intraday import now_ist, safe_send, load_universe, download, news_today
from scanner import get_earnings_date

TEST = intraday.TEST_ONCE
START_TIME = dt.time(9, 20)
END_TIME = dt.time(15, 0)
NO_ENTRY_AFTER = dt.time(14, 45)


def main():
    if not TEST:
        if now_ist().time() >= END_TIME:
            print("Past end time, exiting.")
            return
        while now_ist().time() < START_TIME:
            time.sleep(20)
    uni = load_universe()
    sector_of = dict(zip(uni["ticker"], uni["sector"]))
    daily = swing.download_daily(uni["ticker"].tolist())
    if not daily:
        safe_send("⚠️ Live monitor: could not download daily prices.")
        return
    today = max(df.index[-1].date() for df in daily.values()) if TEST else now_ist().date()
    items = []
    for t, df in daily.items():
        d = df[df.index.date < today]
        if len(d) >= 270:
            items.append((t, d, X.arrays_ce(d)))
    wl = L.build_watchlist(items, sector_of)
    if not wl:
        print("Nothing to watch today.")
        if TEST:
            safe_send("🧪 TEST - Live monitor: nothing to watch on the latest data.")
        return

    done, announced = set(), False
    while True:
        now = now_ist()
        if not TEST and now.time() >= END_TIME:
            break
        waiting = [w for w in wl if (w["key"], w["t"], w["date"]) not in done]
        if not waiting:
            break
        try:
            frames = download(sorted({w["t"] for w in waiting}))
            frames = {t: intraday.completed_only(d) for t, d in frames.items()}
            data_date = max((d.index[-1].date() for d in frames.values() if len(d)), default=None)
            if data_date != today and not TEST:
                if now.time() > dt.time(9, 50):
                    print("No data for today - market closed. Exiting.")
                    return
            elif not announced:
                head = "🧪 TEST - " if TEST else ""
                lines = [f"{head}👀 Live swing watch today ({len(wl)} signals)", ""]
                for w in wl[:L.MAX_LISTED]:
                    lines.append(f"{w['sym']} ({L.STRATS[w['key']]['name']}) - buy above "
                                 f"{w['sig']['trigger']:.2f} (signal {w['date'].strftime('%d %b')})")
                safe_send("\n".join(lines))
                announced = True

            msgs = []
            for w in waiting:
                d5 = frames.get(w["t"])
                if d5 is None:
                    continue
                hit = L.find_break(d5, today, w["sig"]["trigger"])
                if hit is None:
                    continue
                done.add((w["key"], w["t"], w["date"]))
                if not TEST and now.time() >= NO_ENTRY_AFTER:
                    continue
                res = L.handle_break(w, d5, today, *hit)
                if res is None:
                    print(w["sym"], w["key"], "skipped (gap, stop too wide, or stop already hit)")
                    continue
                text, row = res
                if not TEST:
                    res_log = tradelog.add_rows(L.STRATS[w["key"]]["tab"], [row])
                    if res_log is not None and res_log.get("added") == 0:
                        continue                               # already alerted earlier
                ed = get_earnings_date(w["t"])
                if ed:
                    text += f"\n   📅 Earnings date near: {ed}"
                for title in news_today(w["t"]):
                    text += f"\n   ⚠️ News today: {title}"
                msgs.append(text)
            if msgs:
                safe_send(f"⏱ {now.strftime('%H:%M')} IST\n\n" + "\n\n".join(msgs))
        except Exception as e:
            print("cycle error:", type(e).__name__, e)

        if TEST:
            safe_send("✅ Live monitor test finished.")
            return
        n = now_ist()
        mins = 5 - (n.minute % 5)
        nxt = (n + dt.timedelta(minutes=mins)).replace(second=30, microsecond=0)
        time.sleep(max(5, (nxt - now_ist()).total_seconds()))


if __name__ == "__main__":
    main()
