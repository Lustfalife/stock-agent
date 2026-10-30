"""
intraday.py - 5-minute intraday scanner (9:45 AM to 2:30 PM IST).

FLOW
1. Sector bias: most sectors green -> BUY side, red -> SELL side.
   Provisional before 10:00, locked at the first check after 10:00.
2. Live sector check: a BUY is blocked if its own sector is clearly red right
   now (down 0.2% or more); a SELL is blocked if its sector is clearly green.
   (Optional Nifty 50 gate is available but switched off.)
3. SETUP: 100 EMA pullback candle
   BUY : 100 EMA > 200 EMA, close > 100 EMA, candle low < 100 EMA
   SELL: 100 EMA < 200 EMA, close < 100 EMA, candle high > 100 EMA
4. ROOM CHECK: is there at least 1.5R of space before the day's / previous
   day's high (buy) or low (sell)?
   - Yes  -> PULLBACK ENTRY: alert at the close of the setup candle.
   - No   -> BREAKOUT ENTRY: wait for a 5-min candle to CLOSE beyond that level.
5. Entry alert has Entry, SL and Target (1 : 1.5).

Every filter below has an on/off switch (True / False).
Set TEST_ONCE=1 to run a single check on the latest available data.
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

from scanner import send_telegram, HEADERS, NIFTY500_URL, get_news, CATALYST_WORDS

IST = ZoneInfo("Asia/Kolkata")
START_TIME = dt.time(9, 45)
BIAS_TIME = dt.time(10, 0)
END_TIME = dt.time(14, 30)
TEST_ONCE = os.getenv("TEST_ONCE") == "1"

# ===================== SETTINGS YOU CAN CHANGE =====================
RR = 1.5                    # reward : risk  (1 : 1.5)
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

F_TIME = True               # no new entries late in the day
NO_ENTRY_AFTER = dt.time(13, 30)

F_SECTOR_LIVE = True        # block trades whose own sector is clearly against them
SECTOR_BUFFER_PCT = 0.2     # "clearly against" = sector average move of 0.2% or more
F_NIFTY_GATE = False        # Nifty 50 must agree with the bias (off by default)
F_REL_STRENGTH = False      # sector must beat Nifty (buy) / lag Nifty (sell)
F_EARNINGS_SKIP = True      # skip stocks with earnings today or yesterday
F_NEWS_FLAG = True          # add a news warning line to alerts
# ====================================================================


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


# ---------------- Nifty gate ----------------
def nifty_state():
    """Nifty 50 change today and whether it is above its 20 EMA. None if unavailable."""
    try:
        d = yf.download("^NSEI", period="5d", interval="5m", auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = completed_only(prepare(d))
        day = d.index[-1].date()
        prev = d[d.index.date < day]
        cur = d[d.index.date == day]
        pct = (float(cur["Close"].iloc[-1]) / float(prev["Close"].iloc[-1]) - 1) * 100
        ema20 = float(d["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
        return {"pct": pct, "above": float(d["Close"].iloc[-1]) > ema20}
    except Exception as e:
        print("Nifty data unavailable:", e)
        return None


def gate_open(bias, ns):
    if not F_NIFTY_GATE or ns is None:
        return True
    if bias == "BUY":
        return ns["pct"] > 0 and ns["above"]
    return ns["pct"] < 0 and not ns["above"]


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
    """The 100 EMA pullback candle. Returns a dict or None."""
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
    return {"side": side, "c": c, "e100": e100, "e200": e200, "ema100": ema100, "ema200": ema200}


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


def rel_ok(side, sector, sector_pct, ns):
    if not F_REL_STRENGTH or ns is None or sector not in sector_pct:
        return True
    return sector_pct[sector] > ns["pct"] if side == "BUY" else sector_pct[sector] < ns["pct"]


def sector_ok(side, sector, sector_pct):
    """Soft check: only blocks when the stock's sector is clearly against the trade."""
    v = sector_pct.get(sector)
    if not F_SECTOR_LIVE or v is None:
        return True
    return v > -SECTOR_BUFFER_PCT if side == "BUY" else v < SECTOR_BUFFER_PCT


def sector_line(sector, sector_pct):
    v = sector_pct.get(sector)
    return f"\n   Sector today: {v:+.2f}%" if v is not None else ""


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


def bias_message(changes, bias, locked):
    green = int((changes > 0).sum())
    red = int((changes < 0).sum())
    top = ", ".join(f"{k} {v:+.1f}%" for k, v in changes.head(3).items())
    bot = ", ".join(f"{k} {v:+.1f}%" for k, v in changes.tail(3).items())
    tag = "" if locked else " (provisional, final at 10:00)"
    return (f"🧭 Market bias: {bias} side{tag}\n"
            f"Sectors: {green} green / {red} red\n"
            f"Strongest: {top}\nWeakest: {bot}")


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

    bias = None
    locked = False
    watch = all_tickers
    alerted = set()
    pending = {}
    fails = 0
    announced = False
    gate_prev = None

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

            # ----- sector bias (whole universe until locked) -----
            changes = sector_changes(frames, sector_of, data_date)
            sector_pct = changes.to_dict() if changes is not None else {}
            if not locked and changes is not None and len(changes):
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

            if bias:
                # ----- Nifty gate -----
                ns = nifty_state() if (F_NIFTY_GATE or F_REL_STRENGTH) else None
                open_now = gate_open(bias, ns)
                if F_NIFTY_GATE and ns is not None and open_now != gate_prev:
                    if not (gate_prev is None and open_now):
                        state = "OPEN" if open_now else "CLOSED"
                        safe_send(f"🚦 Nifty gate {state} for {bias} trades "
                                  f"(Nifty {ns['pct']:+.2f}% today, "
                                  f"{'above' if ns['above'] else 'below'} its 20 EMA)")
                    gate_prev = open_now
                late = F_TIME and not TEST_ONCE and now.time() >= NO_ENTRY_AFTER

                # ----- setups, breakouts, alerts -----
                msgs = []
                for t, df in frames.items():
                    if t in alerted:
                        continue
                    name = f"{t.replace('.NS', '')} ({sector_of[t]})"

                    if t in pending:                       # waiting breakout setup
                        p = pending[t]
                        res = check_pending(df, p)
                        if res is None:
                            continue
                        kind, ts, cl, volx = res
                        del pending[t]
                        if (kind == "TRIGGER" and open_now and not late
                                and sector_ok(p["side"], sector_of[t], sector_pct)):
                            alerted.add(t)
                            txt = build_alert(t, name, p["side"], "BREAKOUT", p["entry"], p["sl"],
                                              p["label"], ts,
                                              f" | Last close {cl:.2f} | Volume {volx:.1f}x avg",
                                              sector_line(sector_of[t], sector_pct))
                            if txt:
                                msgs.append(txt)
                        continue

                    if not open_now or late:               # new setups need an open gate
                        continue
                    s = raw_setup(df)
                    if not s or s["side"] != bias:
                        continue
                    if F_EMA_GAP and not trend_ok(s):
                        continue
                    if not sector_ok(s["side"], sector_of[t], sector_pct):
                        continue
                    if not rel_ok(s["side"], sector_of[t], sector_pct, ns):
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
                        alerted.add(t)
                        extra = f" | Next level {plan['next'][0]:.2f} ({plan['next'][1]})" if plan["next"] else ""
                        txt = build_alert(t, name, s["side"], "PULLBACK", plan["entry"], plan["sl"],
                                          None, df.index[-1], extra,
                                          sector_line(sector_of[t], sector_pct))
                        if txt:
                            msgs.append(txt)
                    else:
                        pending[t] = {"side": s["side"], "entry": plan["entry"], "sl": plan["sl"],
                                      "label": plan["label"], "ts": df.index[-1]}

                if msgs:
                    safe_send(f"⏱ {now.strftime('%H:%M')} IST - {bias} ENTRY ALERTS\n\n" + "\n\n".join(msgs))
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

    safe_send("🏁 Intraday scanner finished for today (2:30 PM).")


if __name__ == "__main__":
    run()
