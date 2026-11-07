"""
track_live.py - runs after market close. Marks every OPEN trade in the "Pullback" and "CE" tabs:
TARGET / SL (first hit on the 5-minute candles; a gap is filled at the open) or TIME_EXIT after
30 trading days. SL and target in the same candle counts as SL. R = result in units of risk.
"""
import datetime as dt
import sys

import pandas as pd

import swing
import tradelog
import track_swing as T
from intraday import IST
from scanner import send_telegram

TABS = [("Pullback", "Pullback"), ("CE", "Compression-Expansion")]
HOLD_DAYS = 30


def main():
    swing.MAX_HOLD_DAYS = HOLD_DAYS
    open_rows = {}
    for tab, _ in TABS:
        rows = tradelog.open_rows(tab)
        if rows is None:
            print(f"Could not read the {tab} tab.")
            sys.exit(1)
        open_rows[tab] = rows
    tickers = sorted({r["Stock"] + ".NS" for rows in open_rows.values() for r in rows})
    if not tickers:
        print("No open trades.")
        return
    frames = swing.download_5m(tickers, "60d")
    today = dt.datetime.now(IST).date()
    parts = []
    for tab, label in TABS:
        rows = open_rows[tab]
        updates = []
        for r in rows:
            df = frames.get(r["Stock"] + ".NS")
            u = T.evaluate(r, df, today) if df is not None else None
            if u is None and (today - pd.Timestamp(str(r["Date"])).date()).days >= 50:
                u = {"ID": r["ID"], "Status": "NO_DATA", "Exit Price": "", "Exit Time": "",
                     "R": "", "Notes": "Price data no longer available"}
            if u:
                updates.append(u)
        if updates:
            tradelog.update_rows(tab, updates)
            t = sum(u["Status"] == "TARGET" for u in updates)
            s = sum(u["Status"] == "SL" for u in updates)
            x = sum(u["Status"] == "TIME_EXIT" for u in updates)
            net = sum(u["R"] for u in updates if u["R"] != "")
            parts.append(f"{label}: {len(updates)} closed ({t} target, {s} SL, {x} time exit), "
                         f"net {net:+.2f}R, still open {len(rows) - len(updates)}")
        elif rows:
            parts.append(f"{label}: nothing closed, {len(rows)} still open")
    if parts:
        send_telegram("📒 Live swing log updated\n" + "\n".join(parts))
    else:
        print("Nothing to update.")


if __name__ == "__main__":
    main()
