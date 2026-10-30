"""
scanner.py - the engine. You normally never need to edit this file.
Downloads Nifty 500 prices, runs every strategy in strategies.py,
checks news for matches, and sends the result to Telegram.
"""
import io
import os
import sys
import time
import datetime as dt
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import pandas as pd
import requests
import yfinance as yf

from strategies import STRATEGIES

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

NEWS_DAYS = 4
EARNINGS_WINDOW_DAYS = 7
BATCH_SIZE = 50
HEADERS = {"User-Agent": "Mozilla/5.0"}
NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

CATALYST_WORDS = [
    "earnings", "results", "profit", "quarter", "q1", "q2", "q3", "q4",
    "acquisition", "acquire", "merger", "block deal", "bulk deal", "order win",
    "bags order", "wins order", "contract", "sebi", "penalty", "fraud", "probe",
    "rating", "upgrade", "downgrade", "dividend", "buyback", "stake", "fundraise",
    "qip", "ipo", "approval", "usfda", "guidance", "promoter",
]


# ---------------- Telegram ----------------
def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets missing. Message was:\n", text)
        sys.exit(1)
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    # Telegram limit is 4096 characters, so split into chunks
    chunks, current = [], ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > 3800:
            chunks.append(current)
            current = ""
        current += line + "\n"
    if current.strip():
        chunks.append(current)
    for chunk in chunks:
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk,
                                     "disable_web_page_preview": True}, timeout=30)
        if not r.ok:
            print("Telegram error:", r.status_code, r.text)
            sys.exit(1)
        time.sleep(1)
    print("Telegram message sent.")


# ---------------- Stock list ----------------
def load_symbols():
    # Use your own file if present (handy if NSE blocks the download)
    if os.path.exists("nifty500.csv"):
        df = pd.read_csv("nifty500.csv")
        print("Loaded nifty500.csv from repo.")
    else:
        r = requests.get(NIFTY500_URL, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        print("Downloaded Nifty 500 list from NSE.")
    symbols = df["Symbol"].astype(str).str.strip().tolist()
    return [s + ".NS" for s in symbols]


# ---------------- Price data ----------------
def download_batch(tickers):
    data = yf.download(tickers, period="2y", interval="1d", group_by="ticker",
                       auto_adjust=True, progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            df = data[t] if len(tickers) > 1 else data
            df = df.dropna(subset=["Close"])
            if len(df) > 0:
                out[t] = df
        except Exception:
            continue
    return out


# ---------------- News & events ----------------
def get_news(symbol):
    """Headlines from the last few days via Google News RSS."""
    q = requests.utils.quote(f"{symbol} stock India when:{NEWS_DAYS}d")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    items = []
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        root = ET.fromstring(r.content)
        for it in root.iter("item"):
            title = (it.findtext("title") or "").strip()
            pub = it.findtext("pubDate")
            when = parsedate_to_datetime(pub) if pub else None
            items.append((title, when))
    except Exception as e:
        print(f"  news fetch failed for {symbol}: {e}")
    return items[:6]


def get_earnings_date(ticker):
    """Upcoming earnings date if within the next few days."""
    try:
        cal = yf.Ticker(ticker).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if not dates:
            return None
        today = dt.date.today()
        for d in dates:
            d = d.date() if hasattr(d, "date") else d
            if today - dt.timedelta(days=NEWS_DAYS) <= d <= today + dt.timedelta(days=EARNINGS_WINDOW_DAYS):
                return d
    except Exception:
        pass
    return None


def news_block(ticker):
    symbol = ticker.replace(".NS", "")
    lines = []
    earn = get_earnings_date(ticker)
    if earn:
        lines.append(f"   📅 EARNINGS DATE: {earn}")
    news = get_news(symbol)
    flagged = False
    for title, _ in news[:4]:
        hit = any(w in title.lower() for w in CATALYST_WORDS)
        flagged = flagged or hit
        lines.append(f"   {'🔥' if hit else '•'} {title}")
    if not news:
        lines.append("   • No recent headlines found")
    if flagged or earn:
        lines.insert(0, "   ⚠️ Possible catalyst - check before trading")
    return "\n".join(lines)


# ---------------- Main ----------------
def main():
    if os.getenv("TEST_MESSAGE") == "1":
        send_telegram("✅ Test message: your scanner is connected to Telegram.")
        return

    symbols = load_symbols()
    print(f"Scanning {len(symbols)} stocks...")

    matches = {name: [] for name in STRATEGIES}
    scanned = 0

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        try:
            frames = download_batch(batch)
        except Exception as e:
            print(f"Batch {i} failed: {e}")
            continue
        for ticker, df in frames.items():
            scanned += 1
            for name, fn in STRATEGIES.items():
                try:
                    result = fn(df)
                except Exception as e:
                    print(f"{ticker} / {name} error: {e}")
                    continue
                if result:
                    matches[name].append((ticker, result))
        time.sleep(1)

    today = dt.date.today().strftime("%d %b %Y")
    if scanned == 0:
        send_telegram(f"❌ Scanner ran on {today} but could not download any price data.")
        sys.exit(1)

    msg = [f"📊 Stock Scan - {today}", f"Scanned {scanned} stocks", ""]
    total = 0
    for name, found in matches.items():
        msg.append(f"▶ {name}: {len(found)} match(es)")
        for ticker, detail in found:
            total += 1
            msg.append(f"\n{ticker.replace('.NS', '')}\n   {detail}")
            msg.append(news_block(ticker))
        msg.append("")
    if total == 0:
        msg.append("No stocks matched today.")
    msg.append("⚠️ Screening tool only. Check the chart before trading.")
    send_telegram("\n".join(msg))


if __name__ == "__main__":
    main()
