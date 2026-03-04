"""Extended indicators for mega backtest.

Adds: Supertrend, Donchian Channel, Parabolic SAR, Hull MA, ROC, Z-Score,
OBV Slope, and 4H resampled indicators.

All functions operate on pandas Series/DataFrames and return Series.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Supertrend (ATR-based trend follower)
# ---------------------------------------------------------------------------

def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 10, multiplier: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """Supertrend indicator.

    Returns:
        (supertrend_line, direction) where direction is +1 (bullish) or -1 (bearish).
    """
    hl2 = (high + low) / 2

    # ATR via Wilder's EWM
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()

    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    n = len(close)
    st = np.zeros(n)
    direction = np.zeros(n)
    upper_band = upper_basic.values.copy()
    lower_band = lower_basic.values.copy()
    close_vals = close.values

    # Initialize
    st[0] = upper_band[0]
    direction[0] = 1

    for i in range(1, n):
        # Adjust bands
        if lower_band[i] > lower_band[i - 1] or close_vals[i - 1] < lower_band[i - 1]:
            pass  # keep lower_band[i]
        else:
            lower_band[i] = lower_band[i - 1]

        if upper_band[i] < upper_band[i - 1] or close_vals[i - 1] > upper_band[i - 1]:
            pass  # keep upper_band[i]
        else:
            upper_band[i] = upper_band[i - 1]

        # Direction
        if st[i - 1] == upper_band[i - 1]:
            if close_vals[i] > upper_band[i]:
                st[i] = lower_band[i]
                direction[i] = 1
            else:
                st[i] = upper_band[i]
                direction[i] = -1
        else:
            if close_vals[i] < lower_band[i]:
                st[i] = upper_band[i]
                direction[i] = -1
            else:
                st[i] = lower_band[i]
                direction[i] = 1

    return (
        pd.Series(st, index=close.index),
        pd.Series(direction, index=close.index),
    )


# ---------------------------------------------------------------------------
# Donchian Channel
# ---------------------------------------------------------------------------

def donchian_channel(high: pd.Series, low: pd.Series,
                     period: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Donchian Channel.

    Returns: (upper, middle, lower)
    """
    upper = high.rolling(period).max()
    lower = low.rolling(period).min()
    middle = (upper + lower) / 2
    return upper, middle, lower


# ---------------------------------------------------------------------------
# Parabolic SAR
# ---------------------------------------------------------------------------

def parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series,
                  af_start: float = 0.02, af_step: float = 0.02,
                  af_max: float = 0.20) -> tuple[pd.Series, pd.Series]:
    """Parabolic SAR.

    Returns: (sar_values, direction) where direction +1 = bullish, -1 = bearish.
    """
    n = len(close)
    sar = np.zeros(n)
    direction = np.zeros(n)

    h = high.values
    l = low.values

    # Initialize assuming uptrend
    direction[0] = 1
    sar[0] = l[0]
    ep = h[0]  # extreme point
    af = af_start

    for i in range(1, n):
        prev_sar = sar[i - 1]
        prev_dir = direction[i - 1]

        if prev_dir == 1:  # was bullish
            sar[i] = prev_sar + af * (ep - prev_sar)
            # SAR cannot be above prior two lows
            sar[i] = min(sar[i], l[i - 1])
            if i >= 2:
                sar[i] = min(sar[i], l[i - 2])

            if l[i] < sar[i]:  # reversal to bearish
                direction[i] = -1
                sar[i] = ep
                ep = l[i]
                af = af_start
            else:
                direction[i] = 1
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + af_step, af_max)
        else:  # was bearish
            sar[i] = prev_sar + af * (ep - prev_sar)
            # SAR cannot be below prior two highs
            sar[i] = max(sar[i], h[i - 1])
            if i >= 2:
                sar[i] = max(sar[i], h[i - 2])

            if h[i] > sar[i]:  # reversal to bullish
                direction[i] = 1
                sar[i] = ep
                ep = h[i]
                af = af_start
            else:
                direction[i] = -1
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + af_step, af_max)

    return (
        pd.Series(sar, index=close.index),
        pd.Series(direction, index=close.index),
    )


# ---------------------------------------------------------------------------
# Hull Moving Average
# ---------------------------------------------------------------------------

def hull_ma(series: pd.Series, period: int = 9) -> pd.Series:
    """Hull Moving Average — faster response than EMA.

    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    half = max(int(period / 2), 1)
    sqrt_p = max(int(np.sqrt(period)), 1)

    wma_half = series.rolling(half).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    )
    wma_full = series.rolling(period).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    )
    diff = 2 * wma_half - wma_full
    hma = diff.rolling(sqrt_p).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    )
    return hma


# ---------------------------------------------------------------------------
# Rate of Change (ROC)
# ---------------------------------------------------------------------------

def roc(series: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change — percent change over N periods."""
    shifted = series.shift(period)
    return ((series - shifted) / shifted.replace(0, np.nan)) * 100


# ---------------------------------------------------------------------------
# Z-Score
# ---------------------------------------------------------------------------

def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Z-Score: (close - SMA) / StdDev."""
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - sma) / std.replace(0, np.nan)


# ---------------------------------------------------------------------------
# OBV Slope
# ---------------------------------------------------------------------------

def obv_slope(close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """OBV slope over rolling window — trend of volume flow."""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    obv = (volume * direction).cumsum()
    # Simple slope approximation: (OBV_now - OBV_n_ago) / n
    return (obv - obv.shift(period)) / period


# ---------------------------------------------------------------------------
# VWAP (rolling, not cumulative — for intraday context)
# ---------------------------------------------------------------------------

def rolling_vwap(high: pd.Series, low: pd.Series, close: pd.Series,
                 volume: pd.Series, period: int = 20) -> pd.Series:
    """Rolling VWAP over N periods (not cumulative)."""
    typical = (high + low + close) / 3
    tp_vol = (typical * volume).rolling(period).sum()
    vol_sum = volume.rolling(period).sum()
    return tp_vol / vol_sum.replace(0, np.nan)


# ---------------------------------------------------------------------------
# 4H Resampling from 1H data
# ---------------------------------------------------------------------------

def resample_1h_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1H OHLCV DataFrame to 4H.

    Expects columns: timestamp, open, high, low, close, volume
    Returns 4H DataFrame with same column structure.
    """
    ts = df["timestamp"].copy()
    if not pd.api.types.is_datetime64_any_dtype(ts):
        ts = pd.to_datetime(ts)

    df_indexed = df.set_index(ts)
    df_4h = df_indexed.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])

    df_4h = df_4h.reset_index()
    df_4h.rename(columns={"index": "timestamp"}, inplace=True)
    return df_4h


# ---------------------------------------------------------------------------
# Master function: compute all extended indicators on a DataFrame
# ---------------------------------------------------------------------------

def compute_extended_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all extended indicator columns to the DataFrame.

    Expects the DataFrame already has base indicators from compute_all_indicators().
    Adds: supertrend, donchian, psar, hull_ma, roc, zscore, obv_slope, rolling_vwap.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Supertrend (10, 3.0)
    st_line, st_dir = supertrend(high, low, close, 10, 3.0)
    df["supertrend"] = st_line
    df["supertrend_dir"] = st_dir

    # Also compute with alt params for grid
    st_line2, st_dir2 = supertrend(high, low, close, 7, 2.0)
    df["supertrend_fast"] = st_line2
    df["supertrend_fast_dir"] = st_dir2

    # Donchian (20)
    dc_upper, dc_mid, dc_lower = donchian_channel(high, low, 20)
    df["donchian_upper"] = dc_upper
    df["donchian_mid"] = dc_mid
    df["donchian_lower"] = dc_lower

    # Donchian (55) for longer-term breakouts
    dc_upper_55, dc_mid_55, dc_lower_55 = donchian_channel(high, low, 55)
    df["donchian_upper_55"] = dc_upper_55
    df["donchian_lower_55"] = dc_lower_55

    # Parabolic SAR
    psar_val, psar_dir = parabolic_sar(high, low, close)
    df["psar"] = psar_val
    df["psar_dir"] = psar_dir

    # Hull MA (9 and 21)
    df["hull_9"] = hull_ma(close, 9)
    df["hull_21"] = hull_ma(close, 21)

    # ROC (12 and 24)
    df["roc_12"] = roc(close, 12)
    df["roc_24"] = roc(close, 24)

    # Z-Score (20)
    df["zscore_20"] = zscore(close, 20)

    # OBV Slope (14)
    df["obv_slope"] = obv_slope(close, volume, 14)

    # Rolling VWAP (20)
    df["rolling_vwap"] = rolling_vwap(high, low, close, volume, 20)

    # Prev-bar columns for crossover detection
    df["prev_supertrend_dir"] = df["supertrend_dir"].shift(1)
    df["prev_psar_dir"] = df["psar_dir"].shift(1)
    df["prev_hull_9"] = df["hull_9"].shift(1)
    df["prev_hull_21"] = df["hull_21"].shift(1)
    df["prev_close"] = close.shift(1)
    df["prev_high"] = high.shift(1)
    df["prev_low"] = low.shift(1)

    return df
