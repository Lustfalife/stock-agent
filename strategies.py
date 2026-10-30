"""
strategies.py  -  YOUR TRADING RULES LIVE HERE.
Each strategy is a function that receives daily prices for ONE stock.
It returns None if the stock does NOT match, or a short text if it does.
To add a new strategy: write a new function, then add it to STRATEGIES at the bottom.
"""

# ---------- settings you can change ----------
NEAR_HIGH_PCT = 5        # within 5% of 52-week high
COMPRESSION_RATIO = 0.7  # 5-day avg volume < 70% of 20-day avg volume
EXPANSION_MULT = 1.5     # today's volume > 1.5 x 20-day average volume
LOOKBACK_DAYS = 10       # how far back to look for the compression
# --------------------------------------------


def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def compressed_then_expanding(df):
    """Above 20 & 200 EMA, near 52-week high, volume was compressed,
    and today's volume expands above 1.5x the 20-day average."""
    if len(df) < 230:
        return None

    close, high, vol = df["Close"], df["High"], df["Volume"]
    price = float(close.iloc[-1])

    # 1) Trend: above both EMAs
    if not (price > ema(close, 20).iloc[-1] and price > ema(close, 200).iloc[-1]):
        return None

    # 2) Near 52-week high
    high52 = float(high.iloc[-252:].max())
    dist = (high52 - price) / high52 * 100
    if dist > NEAR_HIGH_PCT:
        return None

    # 3) Volume expansion today vs the previous 20 days
    avg20_prev = float(vol.iloc[-21:-1].mean())
    if avg20_prev <= 0:
        return None
    ratio = float(vol.iloc[-1]) / avg20_prev
    if ratio < EXPANSION_MULT:
        return None

    # 4) Volume was compressed at some point in the last few days
    r = vol.rolling(5).mean() / vol.rolling(20).mean()
    recent_min = float(r.iloc[-(LOOKBACK_DAYS + 1):-1].min())
    if not recent_min < COMPRESSION_RATIO:
        return None

    return (f"Price {price:.2f} | {dist:.1f}% below 52w high ({high52:.2f}) | "
            f"Volume {ratio:.1f}x avg | prior compression {recent_min:.2f}")


# Add new strategy functions above, then list them here:
STRATEGIES = {
    "Compression → Volume Expansion": compressed_then_expanding,
}
