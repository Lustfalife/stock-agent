"""
track_intraday.py - runs after market close.
Looks at every OPEN intraday trade in the sheet and marks it:
  TARGET     target was hit first
  SL         stop loss was hit first (also used if both fall in one candle)
  TIME_EXIT  neither was hit by the end of the day (closed at the last price)
R = profit or loss in units of risk (SL hit = -1, target hit = +1.5).
"""
import sys
import datetime as dt

import pandas as pd

import tradelog
from intraday import download, IST
from scanner import send_telegram


def _result(row, status, price, ts, r, note=""):
    return {"ID": row["ID"], "Status": status, "Exit Price": round(float(price), 2),
            "Exit Time": ts.strftime("%H:%M"), "R": round(float(r), 2), "Notes": note}


def evaluate(row, df, today):
    """Returns an update dict for the sheet, or None if the trade is still open."""
    side = row["Side"]
    entry, sl, tgt = float(row["Entry"]), float(row["SL"]), float(row["Target"])
    risk = abs(entry - sl)
    if risk <= 0:
        return None
    sign = 1 if side == "BUY" else -1
    day = pd.Timestamp(str(row["Date"])).date()
    d = df[df.index.date == day]
    d = d[d.index.strftime("%H:%M") > str(row["Time"])]      # candles after the alert
    if d.empty:
        return None
    for ts, c in d.iterrows():
        hi, lo = float(c["High"]), float(c["Low"])
        hit_sl = lo <= sl if side == "BUY" else hi >= sl
        hit_tg = hi >= tgt if side == "BUY" else lo <= tgt
        if hit_sl and hit_tg:
            return _result(row, "SL", sl, ts, -1.0, "SL and target in the same candle - counted as SL")
        if hit_sl:
            return _result(row, "SL", sl, ts, -1.0)
        if hit_tg:
            return _result(row, "TARGET", tgt, ts, sign * (tgt - entry) / risk)
    last_ts = d.index[-1]
    day_over = day < today or last_ts.strftime("%H:%M") >= "15:20"
    if not day_over:
        return None
    close = float(d["Close"].iloc[-1])
    return _result(row, "TIME_EXIT", close, last_ts, sign * (close - entry) / risk,
                   "Closed at end of day")


def main():
    rows = tradelog.open_rows("Intraday")
    if rows is None:
        print("Could not read the trade log.")
        sys.exit(1)
    if not rows:
        print("No open intraday trades.")
        return
    tickers = sorted({r["Stock"] + ".NS" for r in rows})
    frames = download(tickers)
    today = dt.datetime.now(IST).date()
    updates = []
    for r in rows:
        df = frames.get(r["Stock"] + ".NS")
        u = evaluate(r, df, today) if df is not None else None
        if u is None and (today - pd.Timestamp(str(r["Date"])).date()).days >= 4:
            u = {"ID": r["ID"], "Status": "NO_DATA", "Exit Price": "", "Exit Time": "",
                 "R": "", "Notes": "Price data no longer available"}
        if u:
            updates.append(u)
    if updates:
        tradelog.update_rows("Intraday", updates)
        t = sum(u["Status"] == "TARGET" for u in updates)
        s = sum(u["Status"] == "SL" for u in updates)
        x = sum(u["Status"] == "TIME_EXIT" for u in updates)
        net = sum(u["R"] for u in updates if u["R"] != "")
        send_telegram(f"📒 Intraday log updated: {len(updates)} closed "
                      f"({t} target, {s} SL, {x} time exit). Net {net:+.2f}R. "
                      f"Still open: {len(rows) - len(updates)}")
    else:
        print("Nothing to update.")


if __name__ == "__main__":
    main()
