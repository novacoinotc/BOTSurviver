"""Technical indicators implemented with pure numpy/pandas (no pandas_ta dependency)."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _rsi(series: pd.Series, length: int) -> pd.Series:
    """Relative Strength Index."""
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


def _bbands(series: pd.Series, length: int = 20, std: float = 2.0):
    """Bollinger Bands → (upper, mid, lower)."""
    mid = series.rolling(length).mean()
    std_dev = series.rolling(length).std()
    upper = mid + std * std_dev
    lower = mid - std * std_dev
    return upper, mid, lower


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD → (macd_line, signal_line, histogram)."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length).mean()


def _stoch_rsi(close: pd.Series, rsi_length: int = 14, stoch_length: int = 14, k: int = 3, d: int = 3):
    """Stochastic RSI → (K line, D line). Key for scalping overbought/oversold."""
    rsi = _rsi(close, rsi_length)
    rsi_min = rsi.rolling(stoch_length).min()
    rsi_max = rsi.rolling(stoch_length).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    k_line = stoch_rsi.rolling(k).mean() * 100
    d_line = k_line.rolling(d).mean()
    return k_line, d_line


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14):
    """ADX + DI+ / DI- → (ADX, +DI, -DI). Trend strength indicator."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    plus_dm = (high - prev_high).where((high - prev_high) > (prev_low - low), 0.0).clip(lower=0)
    minus_dm = (prev_low - low).where((prev_low - low) > (high - prev_high), 0.0).clip(lower=0)
    atr = _atr(high, low, close, length)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / length, min_periods=length).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / length, min_periods=length).mean() / atr.replace(0, np.nan))
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.ewm(alpha=1 / length, min_periods=length).mean()
    return adx, plus_di, minus_di


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 14) -> pd.Series:
    """Money Flow Index - volume-weighted RSI. Detects divergences."""
    typical = (high + low + close) / 3
    raw_money_flow = typical * volume
    delta = typical.diff()
    pos_flow = raw_money_flow.where(delta > 0, 0.0).rolling(length).sum()
    neg_flow = raw_money_flow.where(delta <= 0, 0.0).rolling(length).sum()
    mfi = 100 - (100 / (1 + pos_flow / neg_flow.replace(0, np.nan)))
    return mfi


def calculate_all(df: pd.DataFrame) -> dict:
    """Calculate all indicators from OHLCV DataFrame. Returns a flat dict."""
    if df.empty or len(df) < 21:
        return {}

    result = {}

    try:
        close = df["close"]

        # RSI
        rsi_7 = _rsi(close, 7)
        rsi_14 = _rsi(close, 14)
        result["rsi_7"] = _last(rsi_7)
        result["rsi_14"] = _last(rsi_14)

        # EMA
        ema_9 = _ema(close, 9)
        ema_21 = _ema(close, 21)
        ema_50 = _ema(close, 50)
        result["ema_9"] = _last(ema_9)
        result["ema_21"] = _last(ema_21)
        result["ema_50"] = _last(ema_50)

        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = _bbands(close, 20, 2.0)
        result["bb_upper"] = _last(bb_upper)
        result["bb_lower"] = _last(bb_lower)
        price = close.iloc[-1]
        upper = result.get("bb_upper")
        lower = result.get("bb_lower")
        if upper and lower and (upper - lower) > 0:
            result["bb_pct"] = round((price - lower) / (upper - lower), 4)

        # MACD
        macd_line, signal_line, histogram = _macd(close, 12, 26, 9)
        hist_val = _last(histogram)
        result["macd_hist"] = hist_val

        hist_clean = histogram.dropna()
        if len(hist_clean) >= 2:
            prev = hist_clean.iloc[-2]
            curr = hist_clean.iloc[-1]
            if prev < 0 and curr > 0:
                result["macd_signal"] = "bullish_cross"
            elif prev > 0 and curr < 0:
                result["macd_signal"] = "bearish_cross"
            elif curr > 0:
                result["macd_signal"] = "bullish"
            else:
                result["macd_signal"] = "bearish"

        # ATR
        atr = _atr(df["high"], df["low"], close, 14)
        result["atr_14"] = _last(atr)

        # VWAP
        if "volume" in df.columns:
            typical = (df["high"] + df["low"] + close) / 3
            cum_vol = df["volume"].cumsum()
            cum_tp_vol = (typical * df["volume"]).cumsum()
            vwap_series = cum_tp_vol / cum_vol
            result["vwap"] = _last(vwap_series)
            vwap_val = result.get("vwap")
            if vwap_val:
                result["price_vs_vwap"] = "above" if price > vwap_val else "below"

        # Volume Delta
        if "taker_buy_quote_volume" in df.columns and "quote_volume" in df.columns:
            last_5 = df.tail(5)
            buy_vol = last_5["taker_buy_quote_volume"].sum()
            total_vol = last_5["quote_volume"].sum()
            sell_vol = total_vol - buy_vol
            result["volume_delta_5m"] = round(buy_vol - sell_vol, 2)

        # Stochastic RSI (scalping overbought/oversold)
        stoch_k, stoch_d = _stoch_rsi(close, 14, 14, 3, 3)
        result["stoch_rsi_k"] = _last(stoch_k)
        result["stoch_rsi_d"] = _last(stoch_d)

        # ADX + DI (trend strength)
        adx, plus_di, minus_di = _adx(df["high"], df["low"], close, 14)
        result["adx"] = _last(adx)
        result["plus_di"] = _last(plus_di)
        result["minus_di"] = _last(minus_di)

        # MFI (volume-weighted momentum)
        if "volume" in df.columns:
            mfi = _mfi(df["high"], df["low"], close, df["volume"], 14)
            result["mfi"] = _last(mfi)

        # BB Squeeze (volatility compression = breakout imminent)
        atr_val = result.get("atr_14")
        bb_upper_val = result.get("bb_upper")
        bb_lower_val = result.get("bb_lower")
        if atr_val and bb_upper_val and bb_lower_val and price > 0:
            bb_width = (bb_upper_val - bb_lower_val) / price
            result["bb_width"] = round(bb_width, 6)
            result["bb_squeeze"] = bb_width < 0.02  # tight bands = squeeze

        # Normalized ATR (% of price, for cross-pair comparison)
        if atr_val and price > 0:
            result["atr_pct"] = round(atr_val / price * 100, 4)

    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")

    return result


def _last(series) -> Optional[float]:
    """Get last non-NaN value from a series."""
    if series is None or series.empty:
        return None
    val = series.dropna()
    if val.empty:
        return None
    v = float(val.iloc[-1])
    return round(v, 6) if abs(v) < 1e10 else round(v, 2)
