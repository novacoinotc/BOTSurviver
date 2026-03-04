"""Compute ALL technical indicators from OHLCV data for backtesting.

Self-contained module — no imports from bot modules. Pure numpy/pandas.
Uses the exact same mathematical formulas as the live bot (bot/strategy/indicators.py).

Indicators computed (added as columns to the DataFrame):
 1. RSI(7), RSI(14)
 2. EMA(9), EMA(21), EMA(50), EMA(100), EMA(200)
 3. SMA(20), SMA(50), SMA(200)
 4. Bollinger Bands(20,2) — upper, lower, bb_pct, bb_width, bb_squeeze
 5. MACD(12,26,9) — macd_line, macd_signal_line, macd_hist, macd_signal
 6. ATR(14) — atr_14, atr_pct
 7. ADX(14), +DI, -DI
 8. Stochastic RSI(14,14,3,3) — K and D
 9. MFI(14)
10. Volume ratio — current volume / SMA(20) of volume
11. EMA alignment — score -1 to +1
12. MACD signal — categorical column
13. RSI divergence — bullish_div, bearish_div, none
14. Consecutive direction — count of same-direction candles
15. VWAP
16. OBV — On Balance Volume
17. Williams %R(14)
18. CCI(20)
19. Keltner Channels(20,1.5) — for squeeze detection
20. Ichimoku — tenkan(9), kijun(26), cloud
21. Price position in range — 0 to 1

Also computes 5m indicators by resampling 1m data.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core indicator functions (matching bot/strategy/indicators.py formulas)
# ---------------------------------------------------------------------------

def _rsi(series: pd.Series, length: int) -> pd.Series:
    """Relative Strength Index using Wilder's EWM smoothing."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=length, adjust=False).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(length).mean()


def _bbands(series: pd.Series, length: int = 20, std: float = 2.0):
    """Bollinger Bands -> (upper, mid, lower)."""
    mid = series.rolling(length).mean()
    std_dev = series.rolling(length).std()
    upper = mid + std * std_dev
    lower = mid - std * std_dev
    return upper, mid, lower


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD -> (macd_line, signal_line, histogram)."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range using Wilder's EWM smoothing."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14):
    """ADX + DI+ / DI- -> (ADX, +DI, -DI). Wilder's method."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    plus_dm = (high - prev_high).where(
        (high - prev_high) > (prev_low - low), 0.0
    ).clip(lower=0)
    minus_dm = (prev_low - low).where(
        (prev_low - low) > (high - prev_high), 0.0
    ).clip(lower=0)
    atr = _atr(high, low, close, length)
    plus_di = 100 * (
        plus_dm.ewm(alpha=1 / length, min_periods=length).mean()
        / atr.replace(0, np.nan)
    )
    minus_di = 100 * (
        minus_dm.ewm(alpha=1 / length, min_periods=length).mean()
        / atr.replace(0, np.nan)
    )
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.ewm(alpha=1 / length, min_periods=length).mean()
    return adx, plus_di, minus_di


def _stoch_rsi(
    close: pd.Series,
    rsi_length: int = 14,
    stoch_length: int = 14,
    k: int = 3,
    d: int = 3,
):
    """Stochastic RSI -> (K, D) scaled 0-100."""
    rsi = _rsi(close, rsi_length)
    rsi_min = rsi.rolling(stoch_length).min()
    rsi_max = rsi.rolling(stoch_length).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    k_line = stoch_rsi.rolling(k).mean() * 100
    d_line = k_line.rolling(d).mean()
    return k_line, d_line


def _mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    length: int = 14,
) -> pd.Series:
    """Money Flow Index — volume-weighted RSI."""
    typical = (high + low + close) / 3
    raw_money_flow = typical * volume
    delta = typical.diff()
    pos_flow = raw_money_flow.where(delta > 0, 0.0).rolling(length).sum()
    neg_flow = raw_money_flow.where(delta <= 0, 0.0).rolling(length).sum()
    return 100 - (100 / (1 + pos_flow / neg_flow.replace(0, np.nan)))


def _vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume Weighted Average Price (cumulative from start of data)."""
    typical = (high + low + close) / 3
    cum_vol = volume.cumsum()
    cum_tp_vol = (typical * volume).cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume."""
    direction = np.sign(close.diff())
    # First value has no diff, set direction to 0
    direction.iloc[0] = 0
    return (volume * direction).cumsum()


def _williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.Series:
    """Williams %R oscillator (-100 to 0)."""
    highest = high.rolling(length).max()
    lowest = low.rolling(length).min()
    wr = -100 * (highest - close) / (highest - lowest).replace(0, np.nan)
    return wr


def _cci(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20
) -> pd.Series:
    """Commodity Channel Index."""
    typical = (high + low + close) / 3
    sma_tp = typical.rolling(length).mean()
    mean_dev = typical.rolling(length).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return (typical - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))


def _keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
    mult: float = 1.5,
):
    """Keltner Channels -> (upper, mid, lower)."""
    mid = _ema(close, length)
    atr = _atr(high, low, close, length)
    upper = mid + mult * atr
    lower = mid - mult * atr
    return upper, mid, lower


def _ichimoku(high: pd.Series, low: pd.Series, tenkan_len: int = 9, kijun_len: int = 26):
    """Ichimoku Cloud -> (tenkan, kijun, senkou_a, senkou_b).

    senkou_a and senkou_b are shifted forward by kijun_len periods
    in standard Ichimoku, but for backtesting we align them to
    the current bar (no forward shift) so they can be used as
    real-time signals without look-ahead bias.
    """
    tenkan = (high.rolling(tenkan_len).max() + low.rolling(tenkan_len).min()) / 2
    kijun = (high.rolling(kijun_len).max() + low.rolling(kijun_len).min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (high.rolling(52).max() + low.rolling(52).min()) / 2
    return tenkan, kijun, senkou_a, senkou_b


# ---------------------------------------------------------------------------
# Vectorized helper functions
# ---------------------------------------------------------------------------

def _ema_alignment_series(ema_9: pd.Series, ema_21: pd.Series, ema_50: pd.Series) -> pd.Series:
    """EMA alignment score: -1 to +1 (vectorized).
    +1 = perfect bullish stack (9>21>50), -1 = perfect bearish stack.
    """
    score = pd.Series(0.0, index=ema_9.index)
    score = score + np.where(ema_9 > ema_21, 0.5, -0.5)
    score = score + np.where(ema_21 > ema_50, 0.5, -0.5)
    return score


def _macd_signal_series(histogram: pd.Series) -> pd.Series:
    """Classify MACD signal per bar (vectorized).

    Returns categorical Series: bullish, bearish, bullish_cross, bearish_cross.
    """
    prev_hist = histogram.shift(1)
    result = pd.Series("none", index=histogram.index, dtype="object")
    # Crosses
    result = result.where(
        ~((prev_hist < 0) & (histogram > 0)), "bullish_cross"
    )
    result = result.where(
        ~((prev_hist > 0) & (histogram < 0)), "bearish_cross"
    )
    # Sustained direction (only where no cross was detected)
    mask_no_cross = ~result.isin(["bullish_cross", "bearish_cross"])
    result = result.where(
        ~(mask_no_cross & (histogram > 0)), "bullish"
    )
    result = result.where(
        ~(mask_no_cross & (histogram <= 0)), "bearish"
    )
    return result


def _rsi_divergence_series(
    close: pd.Series, rsi: pd.Series, lookback: int = 14
) -> pd.Series:
    """Detect RSI/price divergence (vectorized over rolling windows).

    Returns Series with values: bullish_div, bearish_div, none.
    """
    result = pd.Series("none", index=close.index, dtype="object")

    min_len = lookback * 2
    if len(close) < min_len:
        return result

    # Rolling min/max for recent and older windows
    recent_c_min = close.rolling(lookback).min()
    recent_c_max = close.rolling(lookback).max()
    recent_r_min = rsi.rolling(lookback).min()
    recent_r_max = rsi.rolling(lookback).max()

    # Older window: shift by lookback
    older_c_min = close.shift(lookback).rolling(lookback).min()
    older_c_max = close.shift(lookback).rolling(lookback).max()
    older_r_min = rsi.shift(lookback).rolling(lookback).min()
    older_r_max = rsi.shift(lookback).rolling(lookback).max()

    # Bullish divergence: price lower low, RSI higher low
    bull = (recent_c_min < older_c_min) & (recent_r_min > older_r_min)
    result = result.where(~bull, "bullish_div")

    # Bearish divergence: price higher high, RSI lower high
    bear = (recent_c_max > older_c_max) & (recent_r_max < older_r_max)
    # Only set bearish where not already bullish
    result = result.where(~(bear & (result != "bullish_div")), "bearish_div")

    return result


def _consecutive_direction_series(close: pd.Series) -> pd.Series:
    """Count consecutive same-direction candles (vectorized).

    Positive = green streak, negative = red streak.
    """
    diff = close.diff()
    direction = np.sign(diff).fillna(0)
    # Group by runs of same direction
    group = (direction != direction.shift(1)).cumsum()
    streak = direction.groupby(group).cumsum().fillna(0).astype(int)
    return streak


def _price_position_in_range_series(close: pd.Series, lookback: int = 20) -> pd.Series:
    """Position of price within recent high/low range (0=low, 1=high)."""
    roll_high = close.rolling(lookback).max()
    roll_low = close.rolling(lookback).min()
    rng = (roll_high - roll_low).replace(0, np.nan)
    return (close - roll_low) / rng


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_all_indicators(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Takes raw 1m OHLCV DataFrame, returns same DataFrame with all indicator columns added.

    Expected input columns: timestamp, open, high, low, close, volume
    Optional columns: quote_volume, taker_buy_quote_volume, num_trades

    Also computes 5m indicators by resampling.
    """
    df = df_1m.copy()

    if df.empty or len(df) < 30:
        return df

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    has_quote_vol = "quote_volume" in df.columns
    has_taker_buy = "taker_buy_quote_volume" in df.columns

    # -----------------------------------------------------------------------
    # 1. RSI
    # -----------------------------------------------------------------------
    rsi_14 = _rsi(close, 14)
    df["rsi_7"] = _rsi(close, 7)
    df["rsi_14"] = rsi_14

    # -----------------------------------------------------------------------
    # 2. EMA
    # -----------------------------------------------------------------------
    ema_9 = _ema(close, 9)
    ema_21 = _ema(close, 21)
    ema_50 = _ema(close, 50)
    df["ema_9"] = ema_9
    df["ema_21"] = ema_21
    df["ema_50"] = ema_50
    df["ema_100"] = _ema(close, 100)
    df["ema_200"] = _ema(close, 200)

    # -----------------------------------------------------------------------
    # 3. SMA
    # -----------------------------------------------------------------------
    df["sma_20"] = _sma(close, 20)
    df["sma_50"] = _sma(close, 50)
    df["sma_200"] = _sma(close, 200)

    # -----------------------------------------------------------------------
    # 4. Bollinger Bands (20, 2)
    # -----------------------------------------------------------------------
    bb_upper, bb_mid, bb_lower = _bbands(close, 20, 2.0)
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_pct"] = (close - bb_lower) / bb_range
    df["bb_width"] = (bb_upper - bb_lower) / close.replace(0, np.nan)
    df["bb_squeeze"] = df["bb_width"] < 0.02

    # -----------------------------------------------------------------------
    # 5. MACD (12, 26, 9)
    # -----------------------------------------------------------------------
    macd_line, signal_line, histogram = _macd(close, 12, 26, 9)
    df["macd_line"] = macd_line
    df["macd_signal_line"] = signal_line
    df["macd_hist"] = histogram
    df["macd_signal"] = _macd_signal_series(histogram)

    # -----------------------------------------------------------------------
    # 6. ATR (14)
    # -----------------------------------------------------------------------
    atr_14 = _atr(high, low, close, 14)
    df["atr_14"] = atr_14
    df["atr_pct"] = atr_14 / close.replace(0, np.nan) * 100

    # -----------------------------------------------------------------------
    # 7. ADX (14), +DI, -DI
    # -----------------------------------------------------------------------
    adx, plus_di, minus_di = _adx(high, low, close, 14)
    df["adx"] = adx
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # -----------------------------------------------------------------------
    # 8. Stochastic RSI (14, 14, 3, 3)
    # -----------------------------------------------------------------------
    stoch_k, stoch_d = _stoch_rsi(close, 14, 14, 3, 3)
    df["stoch_rsi_k"] = stoch_k
    df["stoch_rsi_d"] = stoch_d

    # -----------------------------------------------------------------------
    # 9. MFI (14)
    # -----------------------------------------------------------------------
    df["mfi"] = _mfi(high, low, close, volume, 14)

    # -----------------------------------------------------------------------
    # 10. Volume ratio — current volume / SMA(20) of volume
    # -----------------------------------------------------------------------
    vol_sma_20 = _sma(volume, 20)
    df["volume_ratio"] = volume / vol_sma_20.replace(0, np.nan)

    # -----------------------------------------------------------------------
    # 11. EMA alignment score (-1 to +1)
    # -----------------------------------------------------------------------
    df["ema_alignment"] = _ema_alignment_series(ema_9, ema_21, ema_50)

    # -----------------------------------------------------------------------
    # 12. MACD signal — already computed above as df["macd_signal"]
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # 13. RSI divergence
    # -----------------------------------------------------------------------
    df["rsi_divergence"] = _rsi_divergence_series(close, rsi_14, 14)

    # -----------------------------------------------------------------------
    # 14. Consecutive direction
    # -----------------------------------------------------------------------
    df["consecutive_direction"] = _consecutive_direction_series(close)

    # -----------------------------------------------------------------------
    # 15. VWAP
    # -----------------------------------------------------------------------
    df["vwap"] = _vwap(high, low, close, volume)

    # -----------------------------------------------------------------------
    # 16. OBV — On Balance Volume
    # -----------------------------------------------------------------------
    df["obv"] = _obv(close, volume)

    # -----------------------------------------------------------------------
    # 17. Williams %R (14)
    # -----------------------------------------------------------------------
    df["williams_r"] = _williams_r(high, low, close, 14)

    # -----------------------------------------------------------------------
    # 18. CCI (20)
    # -----------------------------------------------------------------------
    df["cci"] = _cci(high, low, close, 20)

    # -----------------------------------------------------------------------
    # 19. Keltner Channels (20, 1.5)
    # -----------------------------------------------------------------------
    kc_upper, kc_mid, kc_lower = _keltner_channels(high, low, close, 20, 1.5)
    df["kc_upper"] = kc_upper
    df["kc_lower"] = kc_lower
    # Squeeze detection: BB inside Keltner
    df["kc_squeeze"] = (bb_lower > kc_lower) & (bb_upper < kc_upper)

    # -----------------------------------------------------------------------
    # 20. Ichimoku — tenkan(9), kijun(26), cloud
    # -----------------------------------------------------------------------
    tenkan, kijun, senkou_a, senkou_b = _ichimoku(high, low, 9, 26)
    df["ichimoku_tenkan"] = tenkan
    df["ichimoku_kijun"] = kijun
    df["ichimoku_senkou_a"] = senkou_a
    df["ichimoku_senkou_b"] = senkou_b
    # Cloud direction: bullish when senkou_a > senkou_b
    df["ichimoku_cloud_bullish"] = senkou_a > senkou_b

    # -----------------------------------------------------------------------
    # 21. Price position in range (0 to 1)
    # -----------------------------------------------------------------------
    df["price_position_range"] = _price_position_in_range_series(close, 20)

    # -----------------------------------------------------------------------
    # 5m indicators by resampling
    # -----------------------------------------------------------------------
    df = _compute_5m_indicators(df)

    return df


def _compute_5m_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1m data to 5m, compute key indicators, and forward-fill back.

    Adds columns: ema_9_5m, ema_21_5m, ema_trend_5m
    """
    if len(df) < 10:
        df["ema_9_5m"] = np.nan
        df["ema_21_5m"] = np.nan
        df["ema_trend_5m"] = "neutral"
        return df

    # Ensure timestamp is datetime and set as index for resampling
    ts_col = df["timestamp"].copy()
    if not pd.api.types.is_datetime64_any_dtype(ts_col):
        ts_col = pd.to_datetime(ts_col)

    df_indexed = df.set_index(ts_col)

    # Resample to 5m OHLCV
    df_5m = df_indexed.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])

    if len(df_5m) < 5:
        df["ema_9_5m"] = np.nan
        df["ema_21_5m"] = np.nan
        df["ema_trend_5m"] = "neutral"
        return df

    # Compute EMAs on 5m close
    close_5m = df_5m["close"]
    ema_9_5m = _ema(close_5m, 9)
    ema_21_5m = _ema(close_5m, 21)

    # Determine trend per 5m bar
    trend_5m = pd.Series("neutral", index=df_5m.index, dtype="object")
    price = close_5m

    strong_bull = (price > ema_9_5m) & (ema_9_5m > ema_21_5m)
    bull = (~strong_bull) & ((price > ema_9_5m) | (price > ema_21_5m))
    strong_bear = (price < ema_9_5m) & (ema_9_5m < ema_21_5m)
    bear = (~strong_bear) & ((price < ema_9_5m) | (price < ema_21_5m))

    trend_5m = trend_5m.where(~strong_bull, "strong_bullish")
    trend_5m = trend_5m.where(~bull, "bullish")
    trend_5m = trend_5m.where(~strong_bear, "strong_bearish")
    trend_5m = trend_5m.where(~bear, "bearish")

    # Build a 5m result DataFrame and forward-fill to 1m
    result_5m = pd.DataFrame({
        "ema_9_5m": ema_9_5m,
        "ema_21_5m": ema_21_5m,
        "ema_trend_5m": trend_5m,
    }, index=df_5m.index)

    # Reindex to 1m timestamps and forward-fill
    result_1m = result_5m.reindex(df_indexed.index, method="ffill")

    # Assign back to original df (using positional index alignment)
    df["ema_9_5m"] = result_1m["ema_9_5m"].values
    df["ema_21_5m"] = result_1m["ema_21_5m"].values
    df["ema_trend_5m"] = result_1m["ema_trend_5m"].values

    # Fill any remaining NaN in trend column
    df["ema_trend_5m"] = df["ema_trend_5m"].fillna("neutral")

    return df
