"""Phase 3 strategy signal detectors: 6 new strategies.

1A. BTC Correlation Lag — alts lag BTC moves
1B. Fibonacci Retracement — pullback to fib levels
1C. Heikin-Ashi Trend — smooth trend detection
1D. Range Compression Breakout — volatility squeeze breakout
1E. Momentum Exhaustion — extreme momentum reversal
1F. Multi-EMA Ribbon Expansion — ribbon squeeze -> expansion

Each returns list of {"bar", "dir", "strat", "score"} dicts.
"""

import numpy as np
import pandas as pd


def _safe(val):
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


# ===========================================================================
# 1A. BTC Correlation Lag
# ===========================================================================

def precompute_btc_beta(df_btc: pd.DataFrame, df_alt: pd.DataFrame,
                        window: int = 168) -> pd.DataFrame:
    """Precompute rolling beta and correlation between BTC and an alt.

    Aligns on timestamp. Returns DataFrame with columns:
        btc_return_1h, alt_return_1h, beta, correlation, lag_gap
    indexed to match df_alt's index.
    """
    # Get close prices aligned by timestamp
    btc_ts = df_btc[["timestamp", "close"]].copy()
    alt_ts = df_alt[["timestamp", "close"]].copy()
    btc_ts.columns = ["timestamp", "btc_close"]
    alt_ts.columns = ["timestamp", "alt_close"]

    merged = alt_ts.merge(btc_ts, on="timestamp", how="left")
    merged["btc_return"] = merged["btc_close"].pct_change()
    merged["alt_return"] = merged["alt_close"].pct_change()

    # Rolling correlation and beta
    rolling_corr = merged["btc_return"].rolling(window).corr(merged["alt_return"])
    btc_var = merged["btc_return"].rolling(window).var()
    cov = merged["btc_return"].rolling(window).cov(merged["alt_return"])
    beta = cov / btc_var.replace(0, np.nan)

    merged["correlation"] = rolling_corr
    merged["beta"] = beta
    merged["expected_alt_move"] = merged["btc_return"] * beta
    merged["lag_gap"] = merged["expected_alt_move"] - merged["alt_return"]

    return merged


def detect_btc_correlation_lag(df_alt: pd.DataFrame,
                               btc_data: pd.DataFrame) -> list[dict]:
    """#1A BTC Correlation Lag: alts that lag BTC moves.

    Args:
        df_alt: alt pair DataFrame with indicators
        btc_data: precomputed beta DataFrame from precompute_btc_beta()
    """
    signals = []
    if btc_data is None or len(btc_data) < 170:
        return signals

    for i in range(168, len(btc_data)):
        btc_ret = _safe(btc_data.iloc[i].get("btc_return"))
        lag_gap = _safe(btc_data.iloc[i].get("lag_gap"))
        corr = _safe(btc_data.iloc[i].get("correlation"))

        if btc_ret is None or lag_gap is None or corr is None:
            continue
        if corr < 0.7:
            continue

        score = 5

        # LONG: BTC pumped, alt hasn't caught up
        if btc_ret > 0.015 and lag_gap > 0.01:
            if lag_gap > 0.015:
                score += 1
            if corr > 0.85:
                score += 1
            signals.append({"bar": i, "dir": "LONG",
                          "strat": "btc_correlation_lag", "score": score})

        # SHORT: BTC dumped, alt hasn't caught up
        elif btc_ret < -0.015 and lag_gap < -0.01:
            if lag_gap < -0.015:
                score += 1
            if corr > 0.85:
                score += 1
            signals.append({"bar": i, "dir": "SHORT",
                          "strat": "btc_correlation_lag", "score": score})

    return signals


# ===========================================================================
# 1B. Fibonacci Retracement
# ===========================================================================

def _find_zigzag_swings(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                        atr: np.ndarray, atr_mult: float = 2.5) -> list[dict]:
    """Detect zigzag swings where movement > ATR * multiplier.

    Returns list of {"bar": int, "price": float, "type": "high"|"low"}.
    """
    swings = []
    n = len(close)
    if n < 20:
        return swings

    # Start with first bar
    last_swing_type = None
    last_swing_bar = 0
    last_swing_price = close[0]

    for i in range(1, n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        threshold = atr[i] * atr_mult

        # Check for new high swing
        if high[i] - last_swing_price > threshold and last_swing_type != "high":
            swings.append({"bar": i, "price": high[i], "type": "high"})
            last_swing_type = "high"
            last_swing_bar = i
            last_swing_price = high[i]
        # Check for new low swing
        elif last_swing_price - low[i] > threshold and last_swing_type != "low":
            swings.append({"bar": i, "price": low[i], "type": "low"})
            last_swing_type = "low"
            last_swing_bar = i
            last_swing_price = low[i]
        # Update last swing if price extends
        elif last_swing_type == "high" and high[i] > last_swing_price:
            last_swing_price = high[i]
            swings[-1] = {"bar": i, "price": high[i], "type": "high"}
        elif last_swing_type == "low" and low[i] < last_swing_price:
            last_swing_price = low[i]
            swings[-1] = {"bar": i, "price": low[i], "type": "low"}

    return swings


def detect_fibonacci_retracement(df: pd.DataFrame) -> list[dict]:
    """#1B Fibonacci Retracement: price touches fib 0.618 of last swing."""
    signals = []
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    atr_col = df.get("atr_14")
    if atr_col is None:
        return signals
    atr = atr_col.values

    rsi_col = df.get("rsi_14")
    vol_ratio_col = df.get("volume_ratio")

    swings = _find_zigzag_swings(high, low, close, atr, 2.5)
    if len(swings) < 2:
        return signals

    for i in range(len(df)):
        # Find last completed swing pair before bar i
        relevant = [s for s in swings if s["bar"] < i]
        if len(relevant) < 2:
            continue

        swing_a = relevant[-2]
        swing_b = relevant[-1]

        # Need a swing from high to low (bullish retracement = down pullback)
        # or low to high (bearish retracement = up pullback)
        if swing_a["type"] == "low" and swing_b["type"] == "high":
            # Previous swing was bullish (low -> high), now pulling back down
            swing_range = swing_b["price"] - swing_a["price"]
            if swing_range <= 0:
                continue
            fib_382 = swing_b["price"] - 0.382 * swing_range
            fib_500 = swing_b["price"] - 0.500 * swing_range
            fib_618 = swing_b["price"] - 0.618 * swing_range

            price = close[i]
            tolerance = price * 0.005  # 0.5%

            if abs(price - fib_618) < tolerance:
                rsi = _safe(rsi_col.iloc[i]) if rsi_col is not None else None
                vol_r = _safe(vol_ratio_col.iloc[i]) if vol_ratio_col is not None else None

                # LONG: pullback to fib 618, RSI in oversold zone
                if rsi is not None and 35 <= rsi <= 45:
                    score = 5
                    if abs(price - fib_500) < tolerance:
                        score += 1
                    if vol_r is not None and vol_r < 0.8:
                        score += 1
                    signals.append({"bar": i, "dir": "LONG",
                                  "strat": "fibonacci_retracement", "score": score})

        elif swing_a["type"] == "high" and swing_b["type"] == "low":
            # Previous swing was bearish (high -> low), now pulling back up
            swing_range = swing_a["price"] - swing_b["price"]
            if swing_range <= 0:
                continue
            fib_382 = swing_b["price"] + 0.382 * swing_range
            fib_500 = swing_b["price"] + 0.500 * swing_range
            fib_618 = swing_b["price"] + 0.618 * swing_range

            price = close[i]
            tolerance = price * 0.005

            if abs(price - fib_618) < tolerance:
                rsi = _safe(rsi_col.iloc[i]) if rsi_col is not None else None
                vol_r = _safe(vol_ratio_col.iloc[i]) if vol_ratio_col is not None else None

                # SHORT: pullback to fib 618, RSI in overbought zone
                if rsi is not None and 55 <= rsi <= 65:
                    score = 5
                    if abs(price - fib_500) < tolerance:
                        score += 1
                    if vol_r is not None and vol_r < 0.8:
                        score += 1
                    signals.append({"bar": i, "dir": "SHORT",
                                  "strat": "fibonacci_retracement", "score": score})

    return signals


# ===========================================================================
# 1C. Heikin-Ashi Trend
# ===========================================================================

def _compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Heikin-Ashi candles from OHLC data.

    Returns DataFrame with ha_open, ha_high, ha_low, ha_close, ha_green, ha_no_lower_wick, ha_no_upper_wick.
    """
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)

    ha_close = (o + h + l + c) / 4
    ha_open = np.zeros(n)
    ha_open[0] = (o[0] + c[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2

    ha_high = np.maximum(h, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(l, np.minimum(ha_open, ha_close))

    ha_green = ha_close > ha_open
    # Strong trend: no lower wick on green (open == low), no upper wick on red (open == high)
    ha_no_lower_wick = np.abs(ha_open - ha_low) < (ha_high - ha_low) * 0.01 + 1e-10
    ha_no_upper_wick = np.abs(ha_open - ha_high) < (ha_high - ha_low) * 0.01 + 1e-10

    return pd.DataFrame({
        "ha_open": ha_open,
        "ha_high": ha_high,
        "ha_low": ha_low,
        "ha_close": ha_close,
        "ha_green": ha_green,
        "ha_no_lower_wick": ha_no_lower_wick,
        "ha_no_upper_wick": ha_no_upper_wick,
    }, index=df.index)


def detect_heikin_ashi_trend(df: pd.DataFrame) -> list[dict]:
    """#1C Heikin-Ashi Trend: 3+ consecutive HA candles with no wick + ADX > 25."""
    signals = []
    ha = _compute_heikin_ashi(df)

    adx_col = df.get("adx")
    vol_ratio_col = df.get("volume_ratio")

    n = len(df)
    for i in range(3, n):
        adx = _safe(adx_col.iloc[i]) if adx_col is not None else None
        if adx is None or adx < 25:
            continue

        # Count consecutive green HA candles with no lower wick
        green_streak = 0
        for j in range(i, max(i - 8, -1), -1):
            if ha.iloc[j]["ha_green"] and ha.iloc[j]["ha_no_lower_wick"]:
                green_streak += 1
            else:
                break

        # Count consecutive red HA candles with no upper wick
        red_streak = 0
        for j in range(i, max(i - 8, -1), -1):
            if not ha.iloc[j]["ha_green"] and ha.iloc[j]["ha_no_upper_wick"]:
                red_streak += 1
            else:
                break

        vol_r = _safe(vol_ratio_col.iloc[i]) if vol_ratio_col is not None else None

        if green_streak >= 3:
            score = 5 + (green_streak - 3)  # +1 per extra candle
            if vol_r is not None and vol_r > 1.5:
                score += 1
            signals.append({"bar": i, "dir": "LONG",
                          "strat": "heikin_ashi_trend", "score": min(score, 7)})

        elif red_streak >= 3:
            score = 5 + (red_streak - 3)
            if vol_r is not None and vol_r > 1.5:
                score += 1
            signals.append({"bar": i, "dir": "SHORT",
                          "strat": "heikin_ashi_trend", "score": min(score, 7)})

    return signals


# ===========================================================================
# 1D. Range Compression Breakout
# ===========================================================================

def detect_range_compression_breakout(df: pd.DataFrame,
                                      lookback: int = 20) -> list[dict]:
    """#1D Range Compression Breakout: extreme compression then breakout."""
    signals = []
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    atr_col = df.get("atr_14")
    vol_ratio_col = df.get("volume_ratio")
    if atr_col is None:
        return signals
    atr = atr_col.values

    n = len(df)
    for i in range(lookback + 1, n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        # Compute range of last lookback bars
        window_high = np.max(high[i - lookback:i])
        window_low = np.min(low[i - lookback:i])
        window_range = window_high - window_low
        avg_atr = np.nanmean(atr[i - lookback:i])

        if avg_atr <= 0:
            continue

        compression_ratio = window_range / avg_atr

        # Compression: range < 60% of ATR average
        if compression_ratio >= 0.6:
            continue

        # Current bar breaks out of the range
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        candle_range = curr_high - curr_low

        vol_r = _safe(vol_ratio_col.iloc[i]) if vol_ratio_col is not None else None
        if vol_r is None or vol_r < 2.0:
            continue

        score = 5
        if compression_ratio < 0.4:
            score += 1
        if vol_r > 3.0:
            score += 1

        # Breakout UP: close above range, close in upper third of candle
        if curr_close > window_high and candle_range > 0:
            if (curr_close - curr_low) / candle_range > 0.66:
                signals.append({"bar": i, "dir": "LONG",
                              "strat": "range_compression_breakout", "score": score})

        # Breakout DOWN: close below range, close in lower third
        elif curr_close < window_low and candle_range > 0:
            if (curr_high - curr_close) / candle_range > 0.66:
                signals.append({"bar": i, "dir": "SHORT",
                              "strat": "range_compression_breakout", "score": score})

    return signals


# ===========================================================================
# 1E. Momentum Exhaustion
# ===========================================================================

def detect_momentum_exhaustion(df: pd.DataFrame) -> list[dict]:
    """#1E Momentum Exhaustion: extreme RSI + BB + stoch + consecutive candles."""
    signals = []

    rsi_col = df.get("rsi_14")
    bb_pct_col = df.get("bb_pct")
    stoch_k_col = df.get("stoch_rsi_k")
    consec_col = df.get("consecutive_direction")

    if any(c is None for c in [rsi_col, bb_pct_col, stoch_k_col, consec_col]):
        return signals

    for i in range(1, len(df)):
        rsi = _safe(rsi_col.iloc[i])
        bb_pct = _safe(bb_pct_col.iloc[i])
        stoch_k = _safe(stoch_k_col.iloc[i])
        consec = _safe(consec_col.iloc[i])

        if any(v is None for v in [rsi, bb_pct, stoch_k, consec]):
            continue

        # Bearish exhaustion (reversal LONG)
        if rsi < 20 and bb_pct < 0.05 and stoch_k < 10 and consec <= -3:
            score = 5
            if rsi < 15:
                score += 1
            if abs(consec) >= 5:
                score += 1
            signals.append({"bar": i, "dir": "LONG",
                          "strat": "momentum_exhaustion", "score": score})

        # Bullish exhaustion (reversal SHORT)
        elif rsi > 80 and bb_pct > 0.95 and stoch_k > 90 and consec >= 3:
            score = 5
            if rsi > 85:
                score += 1
            if abs(consec) >= 5:
                score += 1
            signals.append({"bar": i, "dir": "SHORT",
                          "strat": "momentum_exhaustion", "score": score})

    return signals


# ===========================================================================
# 1F. Multi-EMA Ribbon Expansion
# ===========================================================================

def detect_ema_ribbon_expansion(df: pd.DataFrame) -> list[dict]:
    """#1F Multi-EMA Ribbon Expansion: compressed ribbon expands with trend."""
    signals = []

    ema_9_col = df.get("ema_9")
    ema_21_col = df.get("ema_21")
    ema_50_col = df.get("ema_50")
    adx_col = df.get("adx")
    vol_ratio_col = df.get("volume_ratio")
    close_col = df["close"]

    if any(c is None for c in [ema_9_col, ema_21_col, ema_50_col]):
        return signals

    for i in range(2, len(df)):
        ema_9 = _safe(ema_9_col.iloc[i])
        ema_21 = _safe(ema_21_col.iloc[i])
        ema_50 = _safe(ema_50_col.iloc[i])
        close = _safe(close_col.iloc[i])

        if any(v is None for v in [ema_9, ema_21, ema_50, close]) or close == 0:
            continue

        # Previous bar ribbon width
        prev_ema_9 = _safe(ema_9_col.iloc[i - 1])
        prev_ema_21 = _safe(ema_21_col.iloc[i - 1])
        prev_ema_50 = _safe(ema_50_col.iloc[i - 1])
        prev_close = _safe(close_col.iloc[i - 1])

        if any(v is None for v in [prev_ema_9, prev_ema_21, prev_ema_50, prev_close]):
            continue
        if prev_close == 0:
            continue

        prev_ribbon = abs(prev_ema_9 - prev_ema_50) / prev_close
        curr_ribbon = abs(ema_9 - ema_50) / close

        # Ribbon was compressed (< 0.5%) and now expanding
        if prev_ribbon >= 0.005 or curr_ribbon <= prev_ribbon:
            continue

        adx = _safe(adx_col.iloc[i]) if adx_col is not None else None
        prev_adx = _safe(adx_col.iloc[i - 1]) if adx_col is not None else None
        vol_r = _safe(vol_ratio_col.iloc[i]) if vol_ratio_col is not None else None

        score = 5
        if adx is not None and adx > 25:
            score += 1
        if vol_r is not None and vol_r > 1.5:
            score += 1

        # LONG: bullish ribbon (9 > 21 > 50) expanding
        if ema_9 > ema_21 > ema_50:
            # ADX crossing 20 upward (optional boost)
            if adx is not None and prev_adx is not None and prev_adx < 20 and adx >= 20:
                score += 1
            signals.append({"bar": i, "dir": "LONG",
                          "strat": "ema_ribbon_expansion", "score": score})

        # SHORT: bearish ribbon (9 < 21 < 50) expanding
        elif ema_9 < ema_21 < ema_50:
            if adx is not None and prev_adx is not None and prev_adx < 20 and adx >= 20:
                score += 1
            signals.append({"bar": i, "dir": "SHORT",
                          "strat": "ema_ribbon_expansion", "score": score})

    return signals


# ===========================================================================
# Registry
# ===========================================================================

PHASE3_STRATEGIES = {
    "btc_correlation_lag": detect_btc_correlation_lag,  # special: needs btc_data
    "fibonacci_retracement": detect_fibonacci_retracement,
    "heikin_ashi_trend": detect_heikin_ashi_trend,
    "range_compression_breakout": detect_range_compression_breakout,
    "momentum_exhaustion": detect_momentum_exhaustion,
    "ema_ribbon_expansion": detect_ema_ribbon_expansion,
}

# btc_correlation_lag has special signature, handle separately
BTC_STRATEGIES = {"btc_correlation_lag"}

# Standard strategies (no special args)
STANDARD_PHASE3 = {k: v for k, v in PHASE3_STRATEGIES.items()
                   if k not in BTC_STRATEGIES}
