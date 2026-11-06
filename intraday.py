"""
intraday.py - 5-minute intraday scanner, 9:45 AM to 12:00 PM IST.

RULES
 BUY : close > 100 EMA > 200 EMA, candle low < 100 EMA, candle closes above the 100 EMA.
 SELL: close < 100 EMA < 200 EMA, candle high > 100 EMA, candle closes below the 100 EMA.
 The signal candle must be at least 1.3x the average range of the previous 3 candles.
 SECTOR: each stock follows its OWN sector, checked live. Sector up more than 0.1% = BUYS
 only, down more than 0.1% = SELLS only, flat = no trade. The sector index (Nifty IT, Auto,
 Pharma, FMCG, Metal, Energy, Realty, Media) is used where Yahoo has it; otherwise the
 average move of the sector's Nifty 500 stocks.
 ENTRY: if there is room (1.5R) before the day's / previous day's high (buy) or low (sell),
 alert at the signal candle's close (PULLBACK); otherwise wait for a candle to CLOSE beyond
 that level (BREAKOUT). SL from the last 4 candles, target 1:1.5.
 LIMITS: at most 5 alerts a day and 2 per sector, strongest candle first.

Set TEST_ONCE=1 to run a single check on the latest data any time (nothing is logged).
"""
import io
import os
import sys
import time
import datetime as dt
from collections import defaultdict
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

from scanner import send_telegram, HEADERS, NIFTY500_URL, get_news, CATALYST_WORDS

IST = ZoneInfo("Asia/Kolkata")
START_TIME = dt.time(9, 45)
END_TIME = dt.time(12, 1)            # last check at 12:00 (it looks at the 11:55 candle)
TEST_ONCE = os.getenv("TEST_ONCE") == "1"

# ===================== SETTINGS YOU CAN CHANGE =====================
RR = 1.5                    # reward : risk  (1 : 1.5)
SIZE_MULT = 1.3             # signal candle >= 1.3 x average range of the previous 3 candles
SIZE_CANDLES = 3
SECTOR_THRESHOLD_PCT = 0.1  # sector must be beyond +/-0.1% to allow a direction
MAX_ALERTS_PER_DAY = 5
MAX_PER_SECTOR = 2
SL_CANDLES = 4              # SL = lowest low / highest high of last 4 candles
DAY_NEAR_PCT = 0.15         # use day high/low for SL if within this % of it
EXPIRY_CANDLES = 12         # a waiting breakout setup expires after 12 candles
MIN_STOCKS_PER_SECTOR = 3

ENABLE_PULLBACK_ENTRY = True
ENABLE_BREAKOUT_ENTRY = True

F_EMA_GAP = True            # skip tangled / flat 100 & 200 EMAs
EMA_GAP_MIN_PCT = 0.15      # min gap between the EMAs, % of price
SLOPE_CANDLES = 6           # both EMAs must slope with the trade over 6 candles

F_ROOM = True               # room to run decides pullback vs breakout entry

F_VOLUME_BREAK = True       # breakout candle needs above-average volume
VOL_MULT = 1.2              # x the average volume of the last 20 candles

F_ATR_SL = True             # skip tiny or very wide stop losses
ATR_MIN_MULT = 0.8          # SL distance must be at least 0.8 x ATR
ATR_MAX_MULT = 3.0          # ... and at most 3 x ATR

F_EARNINGS_SKIP = True      # skip stocks with earnings today or yesterday
F_NEWS_FLAG = True          # add a news warning line to alerts
# ====================================================================

# sector name (any part of it) -> Yahoo sector index. Sectors not listed use the stock average.
SECTOR_INDEX_KEYWORDS = [
    ("information technology", "^CNXIT"), ("auto", "^CNXAUTO"), ("health", "^CNXPHARMA"),
    ("fast moving", "^CNXFMCG"), ("metal", "^CNXMETAL"), ("oil", "^CNXENERGY"),
    ("power", "^CNXENERGY"), ("realty", "^CNXREALTY"), ("media", "^CNXMEDIA"),
]


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


def prepare(df):
    """Clean one stock's table and put the index in IST."""
    df = df.dropna(subset=["Close"]).copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(IST)
    return df


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
        if len(batch) == 1 and isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        for t in batch:
            try:
                df = data if len(batch) == 1 else data[t]
                df = prepare(df)
                if len(df) >= 210:
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
    """Average % move today of the stocks in each sector (needs 3+ stocks)."""
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


# ---------------- Sector direction ----------------
def index_for(sector):
    s = sector.lower()
    for kw, tk in SECTOR_INDEX_KEYWORDS:
        if kw in s:
            return tk
    return None


def index_changes(tickers, data_date):
    """% change today for each Yahoo sector index (None if its data is missing or stale)."""
    out = {}
    for tk in sorted(set(tickers)):
        try:
            d = yf.download(tk, period="5d", interval="5m", auto_adjust=True, progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d = completed_only(prepare(d))
            if d.empty or d.index[-1].date() != data_date:
                out[tk] = None
                continue
            prev = d[d.index.date < data_date]
            cur = d[d.index.date == data_date]
            if prev.empty or cur.empty:
                out[tk] = None
                continue
            out[tk] = (float(cur["Close"].iloc[-1]) / float(prev["Close"].iloc[-1]) - 1) * 100
        except Exception as e:
            print(f"index {tk} unavailable:", e)
            out[tk] = None
    return out


def sector_view(sectors, sector_avg, idx_pct):
    """sector -> (% move, source, allowed direction 'BUY' / 'SELL' / None)."""
    view = {}
    for sec in sectors:
        tk = index_for(sec)
        if tk and idx_pct.get(tk) is not None:
            pct, how = float(idx_pct[tk]), f"index {tk.lstrip('^')}"
        elif sec in sector_avg:
            pct, how = float(sector_avg[sec]), "stock average"
        else:
            continue
        direction = "BUY" if pct > SECTOR_THRESHOLD_PCT else "SELL" if pct < -SECTOR_THRESHOLD_PCT else None
        view[sec] = (pct, how, direction)
    return view


def snapshot_message(view, last_candle, now, sent_total):
    lag = (now - (last_candle + pd.Timedelta(minutes=5))).total_seconds() / 60
    L = [f"🧭 Sector direction - {now.strftime('%H:%M')} IST",
         f"Data: latest 5-min candle {last_candle.strftime('%H:%M')} "
         f"({max(lag, 0):.0f} min behind the clock) | alerts so far: {sent_total}/{MAX_ALERTS_PER_DAY}"]
    for sec, (pct, how, d) in sorted(view.items(), key=lambda x: -x[1][0]):
        side = {"BUY": "BUYS only", "SELL": "SELLS only", None: "no trade"}[d]
        L.append(f"{sec}: {pct:+.2f}% ({how}) -> {side}")
    return "\n".join(L)


# ---------------- Strategy pieces ----------------
def recent_level(df, kind, n):
    """Lowest low ('low') or highest high ('high') of the last n candles.
    If the day's low/high is within DAY_NEAR_PCT of it, the day's level is used."""
    day = df[df.index.date == df.index[-1].date()]
    if kind == "low":
        swing = float(df["Low"].iloc[-n:].min())
        ext = float(day["Low"].min())
        return ext if swing - ext <= swing * DAY_NEAR_PCT / 100 else swing
    swing = float(df["High"].iloc[-n:].max())
    ext = float(day["High"].max())
    return ext if ext - swing <= swing * DAY_NEAR_PCT / 100 else swing


def day_levels(df):
    """(day high, day low, previous day high, previous day low)."""
    last_date = df.index[-1].date()
    today = df[df.index.date == last_date]
    before = df[df.index.date < last_date]
    dh, dl = float(today["High"].max()), float(today["Low"].min())
    if before.empty:
        return dh, dl, None, None
    prev_day = before[before.index.date == before.index[-1].date()]
    return dh, dl, float(prev_day["High"].max()), float(prev_day["Low"].min())


def atr(df, n=20):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return float(tr.iloc[-n:].mean())


def raw_setup(df):
    """The 100 EMA candle. Returns a dict or None."""
    close = df["Close"]
    ema100 = close.ewm(span=100, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    c, lo, hi = float(close.iloc[-1]), float(df["Low"].iloc[-1]), float(df["High"].iloc[-1])
    e100, e200 = float(ema100.iloc[-1]), float(ema200.iloc[-1])
    if e100 > e200 and c > e100 and lo < e100:
        side = "BUY"
    elif e100 < e200 and c < e100 and hi > e100:
        side = "SELL"
    else:
        return None
    rngs = (df["High"] - df["Low"]).iloc[-(SIZE_CANDLES + 1):]
    avg_prev = float(rngs.iloc[:-1].mean())
    rng = float(rngs.iloc[-1])
    if avg_prev <= 0 or rng < SIZE_MULT * avg_prev:
        return None
    return {"side": side, "c": c, "e100": e100, "e200": e200, "ema100": ema100,
            "ema200": ema200, "range_x": rng / avg_prev}


def trend_ok(s):
    """EMA gap and slope filter."""
    gap = abs(s["e100"] - s["e200"]) / s["c"] * 100
    if gap < EMA_GAP_MIN_PCT:
        return False
    k = SLOPE_CANDLES
    p100, p200 = float(s["ema100"].iloc[-1 - k]), float(s["ema200"].iloc[-1 - k])
    if s["side"] == "BUY":
        return s["e100"] > p100 and s["e200"] > p200
    return s["e100"] < p100 and s["e200"] < p200


def plan_trade(df, s):
    """Decides PULLBACK or BREAKOUT entry and works out entry and SL."""
    side, c = s["side"], s["c"]
    sl = recent_level(df, "low" if side == "BUY" else "high", SL_CANDLES)
    risk = (c - sl) if side == "BUY" else (sl - c)
    if risk <= 0:
        return None
    dh, dl, ph, pl = day_levels(df)
    if side == "BUY":
        levels = sorted((v, n) for v, n in ((dh, "day high"), (ph, "prev day high"))
                        if v is not None and v > c)               # nearest first
        zone = c + RR * risk
        blocking = [x for x in levels if x[0] < zone]
    else:
        levels = sorted(((v, n) for v, n in ((dl, "day low"), (pl, "prev day low"))
                         if v is not None and v < c), reverse=True)  # nearest first
        zone = c - RR * risk
        blocking = [x for x in levels if x[0] > zone]
    nxt = levels[0] if levels else None
    if F_ROOM and blocking:
        level, label = blocking[-1]        # farthest level inside the 1.5R zone
        return {"mode": "BREAKOUT", "entry": level, "sl": sl, "label": label, "next": nxt}
    return {"mode": "PULLBACK", "entry": c, "sl": sl, "label": None, "next": nxt}


def atr_ok(df, plan):
    a = atr(df)
    risk = abs(plan["entry"] - plan["sl"])
    return a > 0 and ATR_MIN_MULT * a <= risk <= ATR_MAX_MULT * a


def sector_line(sector, view):
    v = view.get(sector)
    if not v:
        return ""
    how = "index" if v[1].startswith("index") else "stock avg"
    return f"\n   Sector today: {v[0]:+.2f}% ({how})"


def check_pending(df, p):
    """Looks at candles after the setup candle (breakout entries only).
    Returns ('TRIGGER', time, close, volume_multiple), ('CANCEL', ...) or None."""
    vol_avg = df["Volume"].rolling(20).mean().shift(1)
    newer = df[df.index > p["ts"]]
    for n, (ts, row) in enumerate(newer.iterrows(), start=1):
        va = float(vol_avg.loc[ts]) if pd.notna(vol_avg.loc[ts]) else 0.0
        volx = float(row["Volume"]) / va if va > 0 else 0.0
        vol_ok = (not F_VOLUME_BREAK) or volx >= VOL_MULT
        if p["side"] == "BUY":
            if float(row["Close"]) > p["entry"] and vol_ok:
                return "TRIGGER", ts, float(row["Close"]), volx
            if float(row["Low"]) <= p["sl"]:
                return "CANCEL", ts, None, volx
        else:
            if float(row["Close"]) < p["entry"] and vol_ok:
                return "TRIGGER", ts, float(row["Close"]), volx
            if float(row["High"]) >= p["sl"]:
                return "CANCEL", ts, None, volx
        if n >= EXPIRY_CANDLES:
            return "CANCEL", ts, None, volx
    return None


# ---------------- News / earnings ----------------
def earnings_near(ticker):
    """True if earnings are today or yesterday."""
    try:
        cal = yf.Ticker(ticker).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if not dates:
            return False
        today = now_ist().date()
        for d in dates:
            d = d.date() if hasattr(d, "date") else d
            if d in (today, today - dt.timedelta(days=1)):
                return True
    except Exception:
        pass
    return False


def news_today(ticker):
    out = []
    try:
        for title, when in get_news(ticker.replace(".NS", "")):
            if when is None or when.astimezone(IST).date() != now_ist().date():
                continue
            if any(w in title.lower() for w in CATALYST_WORDS):
                out.append(title)
    except Exception:
        pass
    return out[:2]


# ---------------- Alert text ----------------
def build_alert(ticker, name, side, mode, entry, sl, label, ts, extra="", sec_line=""):
    """Returns the alert text, or None if the stock is skipped (earnings)."""
    if F_EARNINGS_SKIP and earnings_near(ticker):
        print(f"{ticker} skipped: earnings today/yesterday")
        return None
    risk = abs(entry - sl)
    tgt = entry + RR * risk if side == "BUY" else entry - RR * risk
    icon = "🟢" if side == "BUY" else "🔴"
    head = f"{side} - {mode} ENTRY"
    if mode == "BREAKOUT":
        head += f" (closed {'above' if side == 'BUY' else 'below'} {label})"
    text = (f"{icon} {name}\n   {head}\n"
            f"   Entry {entry:.2f} | SL {sl:.2f} | Target {tgt:.2f}\n"
            f"   Risk:Reward 1:{RR} | Candle {ts.strftime('%H:%M')}{extra}{sec_line}")
    if F_NEWS_FLAG:
        for title in news_today(ticker):
            text += f"\n   ⚠️ News today: {title}"
    return text


# ---------------- Main loop ----------------
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
    sectors = sorted(set(sector_of.values()))
    index_tickers = [index_for(s) for s in sectors if index_for(s)]

    alerted, pending = set(), {}
    sent_total, sent_sector = 0, defaultdict(int)
    fails = 0
    first_snapshot = mid_snapshot = False

    while True:
        now = now_ist()
        if not TEST_ONCE and now.time() >= END_TIME:
            break
        try:
            frames = download(all_tickers)
            frames = {t: completed_only(d) for t, d in frames.items()}
            frames = {t: d for t, d in frames.items() if len(d) >= 210}
            if not frames:
                raise RuntimeError("no data downloaded")
            data_date = max(d.index[-1].date() for d in frames.values())
            if not TEST_ONCE and data_date != now.date():
                safe_send("📴 Intraday scanner: no data for today, market looks closed. Stopping.")
                return
            fails = 0

            # ----- live sector direction -----
            avg = sector_changes(frames, sector_of, data_date)
            avg = avg.to_dict() if avg is not None else {}
            view = sector_view(sectors, avg, index_changes(index_tickers, data_date))
            last_candle = max(d.index[-1] for d in frames.values())
            if not first_snapshot or (not mid_snapshot and now.time() >= dt.time(11, 0)):
                safe_send(snapshot_message(view, last_candle, now, sent_total))
                if first_snapshot:
                    mid_snapshot = True
                first_snapshot = True

            # ----- setups, breakouts -----
            cands = []
            for t, df in frames.items():
                if t in alerted:
                    continue
                sec = sector_of[t]
                direction = view.get(sec, (0, "", None))[2]
                name = f"{t.replace('.NS', '')} ({sec})"

                if t in pending:                           # waiting breakout setup
                    p = pending[t]
                    res = check_pending(df, p)
                    if res is None:
                        continue
                    kind, ts, cl, volx = res
                    del pending[t]
                    if kind == "TRIGGER" and direction == p["side"]:
                        cands.append({"t": t, "name": name, "sector": sec, "side": p["side"],
                                      "mode": "BREAKOUT", "entry": p["entry"], "sl": p["sl"],
                                      "label": p["label"], "ts": ts, "score": p["score"],
                                      "extra": f" | Last close {cl:.2f} | Volume {volx:.1f}x avg"})
                    continue

                if direction is None:
                    continue
                s = raw_setup(df)
                if not s or s["side"] != direction:
                    continue
                if F_EMA_GAP and not trend_ok(s):
                    continue
                plan = plan_trade(df, s)
                if not plan:
                    continue
                if plan["mode"] == "PULLBACK" and not ENABLE_PULLBACK_ENTRY:
                    continue
                if plan["mode"] == "BREAKOUT" and not ENABLE_BREAKOUT_ENTRY:
                    continue
                if F_ATR_SL and not atr_ok(df, plan):
                    continue
                if plan["mode"] == "PULLBACK":
                    extra = (f" | Next level {plan['next'][0]:.2f} ({plan['next'][1]})"
                             if plan["next"] else "")
                    extra += f" | Candle {s['range_x']:.1f}x prev 3"
                    cands.append({"t": t, "name": name, "sector": sec, "side": s["side"],
                                  "mode": "PULLBACK", "entry": plan["entry"], "sl": plan["sl"],
                                  "label": None, "ts": df.index[-1], "score": s["range_x"],
                                  "extra": extra})
                else:
                    pending[t] = {"side": s["side"], "entry": plan["entry"], "sl": plan["sl"],
                                  "label": plan["label"], "ts": df.index[-1], "score": s["range_x"]}

            # ----- limits: strongest candle first, 5 a day, 2 per sector -----
            msgs = []
            for c in sorted(cands, key=lambda x: -x["score"]):
                if sent_total >= MAX_ALERTS_PER_DAY:
                    break
                if sent_sector[c["sector"]] >= MAX_PER_SECTOR:
                    continue
                alerted.add(c["t"])
                txt = build_alert(c["t"], c["name"], c["side"], c["mode"], c["entry"], c["sl"],
                                  c["label"], c["ts"], c["extra"], sector_line(c["sector"], view))
                if txt:
                    msgs.append(txt)
                    sent_total += 1
                    sent_sector[c["sector"]] += 1
            if msgs:
                safe_send(f"⏱ {now.strftime('%H:%M')} IST - ENTRY ALERTS "
                          f"({sent_total} of {MAX_ALERTS_PER_DAY} today)\n\n" + "\n\n".join(msgs))
            else:
                print(f"{now.strftime('%H:%M')} no alerts; {len(pending)} breakout setups waiting")
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

    safe_send(f"🏁 Intraday scanner finished (12:00 PM). Alerts today: {sent_total}/{MAX_ALERTS_PER_DAY}")


if __name__ == "__main__":
    run()
