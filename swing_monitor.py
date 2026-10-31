"""
swing_monitor.py - runs during market hours (9:20 AM to 3:00 PM IST).
Watches the stocks that broke out of their box in the last 3 sessions.
Sends the ENTRY alert when 6 five-minute candles in a row (30 min) close above
the breakout day's high. Every entry is logged in the "Swing" tab.
Set TEST_ONCE=1 to run one check on the latest data any time (nothing is logged).
"""
import time
import datetime as dt

import intraday
import swing
import tradelog
from intraday import (now_ist, safe_send, load_universe, download, news_today)
from scanner import get_earnings_date

TEST = intraday.TEST_ONCE
START_TIME = dt.time(9, 20)
END_TIME = dt.time(15, 0)
NO_ENTRY_AFTER = dt.time(14, 45)


def find_candidates(daily, today):
    """Stocks with a valid breakout in the last WATCH_DAYS completed sessions."""
    out = {}
    for t, df in daily.items():
        df = df[df.index.date < today]
        if len(df) < 270:
            continue
        df = swing.add_emas(df)
        n = len(df)
        for back in range(1, swing.WATCH_DAYS + 1):
            j = n - back
            b = swing.check_breakout(df, j)
            if not b:
                continue
            if (df["Close"].iloc[j + 1:] < b["candle_low"]).any():
                continue                      # breakout already failed
            out[t] = b
            break
    return out


def check_confirmation(d, b, today):
    """d = completed 5-min candles. Returns ('CONFIRM', time, close),
    ('CANCEL', reason) or None."""
    t5 = d[d.index.date == today]
    if (t5["Close"] < b["candle_low"]).any():
        return ("CANCEL", "fell below the breakout candle low")
    if len(t5) < swing.CONFIRM_CANDLES:
        return None
    last = t5.iloc[-swing.CONFIRM_CANDLES:]
    if (last["Close"] > b["trigger"]).all():
        return ("CONFIRM", last.index[-1], float(last["Close"].iloc[-1]))
    return None


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
        safe_send("⚠️ Swing monitor: could not download daily prices.")
        return
    today = (max(df.index[-1].date() for df in daily.values()) if TEST else now_ist().date())
    cands = find_candidates(daily, today)
    if not cands:
        print("No swing candidates to watch today.")
        if TEST:
            safe_send("🧪 TEST - Swing monitor: no candidates on the latest data.")
        return

    done = set()
    announced = False
    while True:
        now = now_ist()
        if not TEST and now.time() >= END_TIME:
            break
        waiting = [t for t in cands if t not in done]
        if not waiting:
            break
        try:
            frames = download(waiting)
            frames = {t: intraday.completed_only(d) for t, d in frames.items()}
            data_date = max((d.index[-1].date() for d in frames.values() if len(d)), default=None)
            if data_date != today and not TEST:
                if now.time() > dt.time(9, 50):
                    print("No data for today - market closed. Exiting.")
                    return
            elif not announced:
                head = "🧪 TEST - " if TEST else ""
                lines = [f"{head}👀 Swing watch today ({len(cands)} stock{'s' if len(cands) != 1 else ''})", ""]
                for t, b in cands.items():
                    lines.append(f"{t.replace('.NS', '')} ({sector_of[t]}) - needs 30 min above "
                                 f"{b['trigger']:.2f} (breakout {b['date']})")
                safe_send("\n".join(lines))
                announced = True

            msgs = []
            for t in waiting:
                if t not in frames:
                    continue
                b = cands[t]
                res = check_confirmation(frames[t], b, today)
                if res is None:
                    continue
                done.add(t)
                if res[0] == "CANCEL":
                    print(t, "cancelled:", res[1])
                    continue
                _, ts, entry = res
                if not TEST and now.time() >= NO_ENTRY_AFTER:
                    continue
                if entry > b["trigger"] * (1 + swing.MAX_EXTEND_PCT / 100):
                    print(t, "skipped: already extended above the breakout high")
                    continue
                plan = swing.plan_levels(entry, b["candle_low"])
                if plan is None:
                    print(t, "skipped: risk too large")
                    continue
                sl, tgt, risk_pct = plan
                sym = t.replace(".NS", "")
                if not TEST:
                    res_log = tradelog.add_rows("Swing", [{
                        "ID": f"{b['date']}|{sym}|SWING|CONFIRM", "Date": today.strftime("%Y-%m-%d"),
                        "Time": ts.strftime("%H:%M"), "Strategy": "Swing 52w-high box breakout",
                        "Stock": sym, "Sector": sector_of[t], "Side": "BUY",
                        "Entry Type": "CONFIRMED", "Entry": round(entry, 2), "SL": round(sl, 2),
                        "Target": round(tgt, 2), "Status": "OPEN", "Exit Price": "",
                        "Exit Time": "", "R": "", "Notes": f"Breakout day {b['date']}"}])
                    if res_log is not None and res_log.get("added") == 0:
                        continue                       # already alerted earlier
                text = (f"🟢 SWING ENTRY - {sym} ({sector_of[t]})\n"
                        f"   Held above breakout high {b['trigger']:.2f} for 30 min "
                        f"(breakout day {b['date']}, {b['vol_x']:.1f}x volume)\n"
                        f"   Entry {entry:.2f} | SL {sl:.2f} | Target {tgt:.2f}\n"
                        f"   Risk {risk_pct:.1f}% | Reward:Risk 1:{swing.RR} | "
                        f"Time exit after {swing.MAX_HOLD_DAYS} trading days")
                ed = get_earnings_date(t)
                if ed:
                    text += f"\n   📅 Earnings date near: {ed}"
                for title in news_today(t):
                    text += f"\n   ⚠️ News today: {title}"
                msgs.append(text)
            if msgs:
                safe_send(f"⏱ {now.strftime('%H:%M')} IST\n\n" + "\n\n".join(msgs))
        except Exception as e:
            print("cycle error:", type(e).__name__, e)

        if TEST:
            safe_send("✅ Swing monitor test finished.")
            return
        n = now_ist()
        mins = 5 - (n.minute % 5)
        nxt = (n + dt.timedelta(minutes=mins)).replace(second=30, microsecond=0)
        time.sleep(max(5, (nxt - now_ist()).total_seconds()))


if __name__ == "__main__":
    main()
