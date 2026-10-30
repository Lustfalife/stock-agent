"""
intraday.py - 5-minute intraday scanner (9:45 AM to 2:30 PM IST).

1. Works out sector bias (most sectors green -> BUY side, red -> SELL side).
   Provisional before 10:00, locked at the first check after 10:00.
2. Every 5 minutes, looks for the 100 EMA pullback candle:
   BUY : close > 200 EMA and > 100 EMA, candle low < 100 EMA
   SELL: close < 200 EMA and < 100 EMA, candle high > 100 EMA
3. Only stocks from sectors matching the bias are alerted. Once per stock per day.

Set TEST_ONCE=1 to run a single check on the latest available data (any time).
"""
import io
import os
import sys
import time
import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

from scanner import send_telegram, HEADERS, NIFTY500_URL

IST = ZoneInfo("Asia/Kolkata")
START_TIME = dt.time(9, 45)
BIAS_TIME = dt.time(10, 0)
END_TIME = dt.time(14, 30)
TEST_ONCE = os.getenv("TEST_ONCE") == "1"
MIN_STOCKS_PER_SECTOR = 3


def safe_send(text):
    try:
        send_telegram(text)
    except SystemExit:
        print("Telegram send failed (continuing).")


def now_ist():
    return dt.datetime.now(IST)


def load_universe():
    if os.path.exists("nifty500.csv"):
        df = pd.read_csv("nifty500.csv")
    else:
        r = requests.get(NIFTY500_URL, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    df["ticker"] = df["Symbol"].astype(str).str.strip() + ".NS"
    df["sector"] = df["Industry"].astype(str).str.strip()
    return df[["ticker", "sector"]]


def download(tickers):
    frames = {}
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i + 50]
        data = None
        for _ in range(2):
            try:
                data = yf.download(batch, period="5d", interval="5m", group_by="ticker",
                                   auto_adjust=True, progress=False, threads=True)
                break
            except Exception as e:
                print("download error:", e)
                time.sleep(5)
        if data is None:
            continue
        for t in batch:
            try:
                df = data if len(batch) == 1 else data[t]
                df = df.dropna(subset=["Close"]).copy()
                if len(df) < 210:
                    continue
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                df.index = df.index.tz_convert(IST)
                frames[t] = df
            except Exception:
                continue
        time.sleep(1)
    return frames


def completed_only(df):
    """Drop the candle still in progress."""
    if TEST_ONCE:
        return df
    return df[df.index + pd.Timedelta(minutes=5) <= now_ist()]


def sector_changes(frames, sector_of, data_date):
    rows = []
    for t, df in frames.items():
        prev = df[df.index.date < data_date]
        cur = df[df.index.date == data_date]
        if prev.empty or cur.empty:
            continue
        pct = (float(cur["Close"].iloc[-1]) / float(prev["Close"].iloc[-1]) - 1) * 100
        rows.append((sector_of[t], pct))
    if not rows:
        return None
    s = pd.DataFrame(rows, columns=["sector", "pct"])
    g = s.groupby("sector")["pct"].agg(["mean", "count"])
    return g[g["count"] >= MIN_STOCKS_PER_SECTOR]["mean"].sort_values(ascending=False)


def check_signal(df):
    """Returns 'BUY', 'SELL' or None for the latest completed candle."""
    close = df["Close"]
    ema100 = close.ewm(span=100, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    c, lo, hi = float(close.iloc[-1]), float(df["Low"].iloc[-1]), float(df["High"].iloc[-1])
    e100, e200 = float(ema100.iloc[-1]), float(ema200.iloc[-1])
    if c > e200 and c > e100 and lo < e100:
        return "BUY", c, e100, e200
    if c < e200 and c < e100 and hi > e100:
        return "SELL", c, e100, e200
    return None


def bias_message(changes, bias, locked):
    green = int((changes > 0).sum())
    red = int((changes < 0).sum())
    top = ", ".join(f"{k} {v:+.1f}%" for k, v in changes.head(3).items())
    bot = ", ".join(f"{k} {v:+.1f}%" for k, v in changes.tail(3).items())
    tag = "" if locked else " (provisional, final at 10:00)"
    return (f"🧭 Market bias: {bias} side{tag}\n"
            f"Sectors: {green} green / {red} red\n"
            f"Strongest: {top}\nWeakest: {bot}")


def run():
    now = now_ist()
    if not TEST_ONCE:
        if now.time() >= END_TIME:
            print("Past end time, exiting.")
            return
        while now_ist().time() < START_TIME:
            time.sleep(20)

    universe = load_universe()
    sector_of = dict(zip(universe["ticker"], universe["sector"]))
    all_tickers = universe["ticker"].tolist()

    bias = None
    locked = False
    watch = all_tickers
    alerted = set()
    fails = 0
    announced = False

    while True:
        now = now_ist()
        if not TEST_ONCE and now.time() >= END_TIME:
            break
        try:
            frames = download(watch)
            frames = {t: completed_only(d) for t, d in frames.items()}
            frames = {t: d for t, d in frames.items() if len(d) >= 210}
            if not frames:
                raise RuntimeError("no data downloaded")
            data_date = max(d.index[-1].date() for d in frames.values())
            if not TEST_ONCE and data_date != now.date():
                safe_send("📴 Intraday scanner: no data for today, market looks closed. Stopping.")
                return
            fails = 0

            # ----- sector bias (use the whole universe until locked) -----
            if not locked:
                changes = sector_changes(frames, sector_of, data_date)
                if changes is not None and len(changes):
                    green, red = (changes > 0).sum(), (changes < 0).sum()
                    new_bias = "BUY" if green >= red else "SELL"
                    ready = TEST_ONCE or now.time() >= BIAS_TIME
                    if ready:
                        locked = True
                    if ready or not announced or new_bias != bias:
                        safe_send(bias_message(changes, new_bias, ready))
                        announced = True
                    bias = new_bias
                    good = set(changes[changes > 0].index) if bias == "BUY" else set(changes[changes < 0].index)
                    if locked:
                        watch = [t for t in all_tickers if sector_of[t] in good]
                        frames = {t: d for t, d in frames.items() if t in watch}
                    else:
                        frames = {t: d for t, d in frames.items() if sector_of[t] in good}

            # ----- candle signals -----
            if bias:
                hits = []
                for t, df in frames.items():
                    if t in alerted:
                        continue
                    res = check_signal(df)
                    if res and res[0] == bias:
                        side, c, e100, e200 = res
                        alerted.add(t)
                        hits.append(f"{'🟢' if side == 'BUY' else '🔴'} {t.replace('.NS', '')} "
                                    f"({sector_of[t]})\n   {side} | Close {c:.2f} | 100 EMA {e100:.2f} | 200 EMA {e200:.2f}"
                                    f"\n   Candle: {df.index[-1].strftime('%H:%M')}")
                if hits:
                    safe_send(f"⏱ {now.strftime('%H:%M')} IST - {bias} setups\n\n" + "\n\n".join(hits))
                else:
                    print(f"{now.strftime('%H:%M')} no new setups")
        except Exception as e:
            fails += 1
            print("cycle error:", type(e).__name__, e)
            if fails == 3:
                safe_send(f"⚠️ Intraday scanner: data problems for 3 checks in a row ({e}). Still trying.")

        if TEST_ONCE:
            safe_send("✅ Test finished (TEST_ONCE mode).")
            return
        # wait for the next 5-minute boundary + 30 seconds
        n = now_ist()
        mins = 5 - (n.minute % 5)
        nxt = (n + dt.timedelta(minutes=mins)).replace(second=30, microsecond=0)
        time.sleep(max(5, (nxt - now_ist()).total_seconds()))

    safe_send("🏁 Intraday scanner finished for today (2:30 PM).")


if __name__ == "__main__":
    run()
