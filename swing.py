"""
swing.py - the swing strategy rules (daily chart) and shared helpers.

STRATEGY: breakout from a consolidation box near the 52-week high.
 1. Price above the 20 and 200 EMA.
 2. The 52-week high was made 10 to 40 trading days ago.
 3. Since that high the stock never fell more than 7%.
 4. The last 10 sessions form a tight box (range up to 6%), with the box top
    within 5% of the 52-week high.
 5. Volume was compressed in the box (below 80% of the 50 sessions before it).
 6. Breakout day: close above the box top, volume above 1.5x the 20-day
    average, and not more than 3% above the box top (no chasing).
 7. Entry only next day(s): after 6 five-minute candles (30 min) in a row
    close above the breakout day's HIGH.
 SL = breakout candle low (at least 1.5% below entry; skip if risk > 6%).
 Target = 1 : 1.5. Time exit after 15 trading days.
"""
import time

import pandas as pd
import yfinance as yf

import intraday

IST = intraday.IST

# ===================== SETTINGS YOU CAN CHANGE =====================
RR = 1.5
BOX_DAYS = 10
MIN_DAYS_SINCE_HIGH = 10
MAX_DAYS_SINCE_HIGH = 40
MAX_PULLBACK_PCT = 7
MAX_BOX_RANGE_PCT = 6
BOX_NEAR_HIGH_PCT = 5
VOL_COMPRESSION = 0.8
VOL_EXPANSION = 1.5
MAX_EXTEND_PCT = 3          # breakout close / entry must be within 3% of the level
MIN_RISK_PCT = 1.5
MAX_RISK_PCT = 6
CONFIRM_CANDLES = 6         # 6 x 5 min = 30 minutes above the breakout high
WATCH_DAYS = 3              # a breakout stays on the watchlist for 3 sessions
MAX_HOLD_DAYS = 15
# ====================================================================


def _fix_daily_index(df):
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(IST).tz_localize(None)
    df.index = idx.normalize()
    return df


def download_daily(tickers):
    frames = {}
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i + 50]
        data = None
        for _ in range(2):
            try:
                data = yf.download(batch, period="2y", interval="1d", group_by="ticker",
                                   auto_adjust=True, progress=False, threads=True)
                break
            except Exception as e:
                print("daily download error:", e)
                time.sleep(5)
        if data is None:
            continue
        if len(batch) == 1 and isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        for t in batch:
            try:
                df = data if len(batch) == 1 else data[t]
                df = _fix_daily_index(df.dropna(subset=["Close"]).copy())
                if len(df) >= 270:
                    frames[t] = df
            except Exception:
                continue
        time.sleep(1)
    return frames


def download_5m(tickers, period="1mo"):
    frames = {}
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i + 50]
        data = None
        for _ in range(2):
            try:
                data = yf.download(batch, period=period, interval="5m", group_by="ticker",
                                   auto_adjust=True, progress=False, threads=True)
                break
            except Exception as e:
                print("5m download error:", e)
                time.sleep(5)
        if data is None:
            continue
        if len(batch) == 1 and isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        for t in batch:
            try:
                df = data if len(batch) == 1 else data[t]
                df = intraday.prepare(df)
                if len(df):
                    frames[t] = df
            except Exception:
                continue
        time.sleep(1)
    return frames


def add_emas(df):
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    return df


def plan_levels(entry, candle_low):
    """Returns (sl, target, risk_pct) or None if the risk is too large."""
    sl = min(candle_low, entry * (1 - MIN_RISK_PCT / 100))
    risk = entry - sl
    risk_pct = risk / entry * 100
    if risk <= 0 or risk_pct > MAX_RISK_PCT:
        return None
    return sl, entry + RR * risk, risk_pct


def check_breakout(df, i):
    """Is session number i (position in df, with EMA20/EMA200 columns) a valid
    breakout day? Returns a dict or None."""
    if i < 260 or i >= len(df):
        return None
    row = df.iloc[i]
    close, hi, lo, vol = (float(row["Close"]), float(row["High"]),
                          float(row["Low"]), float(row["Volume"]))
    if not (close > float(row["EMA20"]) and close > float(row["EMA200"])):
        return None
    # the 52-week high (before the breakout day) and how long ago it was made
    highs = df["High"].iloc[i - 252:i].values
    pos = len(highs) - 1 - int(highs[::-1].argmax())
    high52 = float(highs[pos])
    abs_high = i - 252 + pos
    since = i - abs_high
    if not (MIN_DAYS_SINCE_HIGH <= since <= MAX_DAYS_SINCE_HIGH):
        return None
    # the stock must not have fallen much since that high
    pb_low = float(df["Low"].iloc[abs_high:i].min())
    if (high52 - pb_low) / high52 * 100 > MAX_PULLBACK_PCT:
        return None
    # the consolidation box
    box = df.iloc[i - BOX_DAYS:i]
    box_high, box_low = float(box["High"].max()), float(box["Low"].min())
    if (box_high - box_low) / box_low * 100 > MAX_BOX_RANGE_PCT:
        return None
    if (high52 - box_high) / high52 * 100 > BOX_NEAR_HIGH_PCT:
        return None
    # volume compressed in the box
    prior = float(df["Volume"].iloc[i - BOX_DAYS - 50:i - BOX_DAYS].mean())
    if prior <= 0 or float(box["Volume"].mean()) >= VOL_COMPRESSION * prior:
        return None
    # breakout day itself
    avg20 = float(df["Volume"].iloc[i - 20:i].mean())
    if avg20 <= 0:
        return None
    volx = vol / avg20
    if volx < VOL_EXPANSION:
        return None
    if close <= box_high or close > box_high * (1 + MAX_EXTEND_PCT / 100):
        return None
    if plan_levels(hi, lo) is None:
        return None
    return {"date": df.index[i].strftime("%Y-%m-%d"), "trigger": hi, "candle_low": lo,
            "close": close, "box_high": box_high, "box_low": box_low,
            "high52": high52, "days_since_high": since, "vol_x": volx}
