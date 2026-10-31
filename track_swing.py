"""
track_swing.py - runs after market close.
Marks every OPEN swing trade in the "Swing" tab:
  TARGET / SL        whichever was hit first on the 5-minute candles
                     (a gap through the level is filled at the open)
  TIME_EXIT          neither hit within 15 trading days (closed at that day's close)
SL and target in the same candle counts as SL. R = result in units of risk.
"""
import sys
import datetime as dt

import pandas as pd

import swing
import tradelog
from intraday import IST
from scanner import send_telegram


def _result(row, status, price, ts, risk, entry, note=""):
    r = (float(price) - entry) / risk
    return {"ID": row["ID"], "Status": status, "Exit Price": round(float(price), 2),
            "Exit Time": ts.strftime("%d %b %H:%M"), "R": round(r, 2), "Notes": note}


def evaluate(row, df, today):
    """Returns an update dict, or None if the trade is still open."""
    entry, sl, tgt = float(row["Entry"]), float(row["SL"]), float(row["Target"])
    risk = entry - sl
    if risk <= 0:
        return None
    start = pd.Timestamp(f"{row['Date']} {row['Time']}").tz_localize(IST)
    d = df[df.index > start]
    if d.empty:
        return None
    post_days = sorted({x for x in d.index.date if x > start.date()})
    last_day = post_days[swing.MAX_HOLD_DAYS - 1] if len(post_days) >= swing.MAX_HOLD_DAYS else None
    if last_day is not None:
        d = d[d.index.date <= last_day]
    for ts, c in d.iterrows():
        o, hi, lo = float(c["Open"]), float(c["High"]), float(c["Low"])
        hit_sl, hit_tg = lo <= sl, hi >= tgt
        if hit_sl and hit_tg:
            return _result(row, "SL", min(o, sl), ts, risk, entry,
                           "SL and target in the same candle - counted as SL")
        if hit_sl:
            return _result(row, "SL", min(o, sl), ts, risk, entry)
        if hit_tg:
            return _result(row, "TARGET", max(o, tgt), ts, risk, entry)
    if last_day is None:
        return None
    last_ts = d.index[-1]
    if not (last_day < today or last_ts.strftime("%H:%M") >= "15:20"):
        return None
    return _result(row, "TIME_EXIT", float(d["Close"].iloc[-1]), last_ts, risk, entry,
                   f"Closed after {swing.MAX_HOLD_DAYS} trading days")


def main():
    rows = tradelog.open_rows("Swing")
    if rows is None:
        print("Could not read the trade log.")
        sys.exit(1)
    if not rows:
        print("No open swing trades.")
        return
    frames = swing.download_5m(sorted({r["Stock"] + ".NS" for r in rows}), "1mo")
    today = dt.datetime.now(IST).date()
    updates = []
    for r in rows:
        df = frames.get(r["Stock"] + ".NS")
        u = evaluate(r, df, today) if df is not None else None
        if u is None and (today - pd.Timestamp(str(r["Date"])).date()).days >= 30:
            u = {"ID": r["ID"], "Status": "NO_DATA", "Exit Price": "", "Exit Time": "",
                 "R": "", "Notes": "Price data no longer available"}
        if u:
            updates.append(u)
    if updates:
        tradelog.update_rows("Swing", updates)
        t = sum(u["Status"] == "TARGET" for u in updates)
        s = sum(u["Status"] == "SL" for u in updates)
        x = sum(u["Status"] == "TIME_EXIT" for u in updates)
        net = sum(u["R"] for u in updates if u["R"] != "")
        send_telegram(f"📒 Swing log updated: {len(updates)} closed "
                      f"({t} target, {s} SL, {x} time exit). Net {net:+.2f}R. "
                      f"Still open: {len(rows) - len(updates)}")
    else:
        print("Nothing to update.")


if __name__ == "__main__":
    main()
