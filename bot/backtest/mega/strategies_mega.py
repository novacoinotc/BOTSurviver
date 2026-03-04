"""27 strategy signal detectors for mega backtest.

Each function takes a DataFrame (with indicators pre-computed) and returns a
list of signal dicts:
    {"bar": int, "dir": "LONG"|"SHORT", "strat": str, "score": float}

All strategies use the same signal format for compatibility with the simulator.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

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
# EXISTING STRATEGIES (12) — converted to signal-list format
# ===========================================================================

def detect_macd_cross_1h(df: pd.DataFrame) -> list[dict]:
    """#1 MACD cross + EMA alignment + volume (ACTUAL winner)."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        macd_sig = row.get("macd_signal")
        ema_align = _safe(row.get("ema_alignment"))
        vol_ratio = _safe(row.get("volume_ratio")) or 0

        if macd_sig is None or ema_align is None:
            continue

        if macd_sig == "bullish_cross" and ema_align >= 0 and vol_ratio > 0.5:
            signals.append({"bar": i, "dir": "LONG", "strat": "macd_cross_1h", "score": 5})
        elif macd_sig == "bearish_cross" and ema_align <= 0 and vol_ratio > 0.5:
            signals.append({"bar": i, "dir": "SHORT", "strat": "macd_cross_1h", "score": 5})
    return signals


def detect_trend_follow_1h(df: pd.DataFrame) -> list[dict]:
    """#2 ADX + EMA aligned + MACD + RSI pullback."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        adx = _safe(row.get("adx"))
        ema_align = _safe(row.get("ema_alignment"))
        plus_di = _safe(row.get("plus_di"))
        minus_di = _safe(row.get("minus_di"))
        rsi_14 = _safe(row.get("rsi_14"))
        macd_sig = row.get("macd_signal")
        vol_ratio = _safe(row.get("volume_ratio")) or 0
        stoch_k = _safe(row.get("stoch_rsi_k"))

        if not all(v is not None for v in [adx, ema_align, plus_di, minus_di, rsi_14, macd_sig]):
            continue

        # LONG
        if (adx > 25 and ema_align > 0.3 and plus_di > minus_di
                and 30 <= rsi_14 <= 55 and macd_sig in ("bullish", "bullish_cross")):
            score = 4
            if adx > 35: score += 1
            if ema_align > 0.5: score += 1
            if macd_sig == "bullish_cross": score += 2
            if vol_ratio > 1.2: score += 1
            if stoch_k and stoch_k < 40: score += 1
            signals.append({"bar": i, "dir": "LONG", "strat": "trend_follow_1h", "score": score})
        # SHORT
        elif (adx > 25 and ema_align < -0.3 and minus_di > plus_di
                and 45 <= rsi_14 <= 70 and macd_sig in ("bearish", "bearish_cross")):
            score = 4
            if adx > 35: score += 1
            if ema_align < -0.5: score += 1
            if macd_sig == "bearish_cross": score += 2
            if vol_ratio > 1.2: score += 1
            if stoch_k and stoch_k > 60: score += 1
            signals.append({"bar": i, "dir": "SHORT", "strat": "trend_follow_1h", "score": score})
    return signals


def detect_multi_confirm(df: pd.DataFrame) -> list[dict]:
    """#3 Multi-indicator confluence (same as run_low_freq)."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        price = row["close"]
        rsi_14 = _safe(row.get("rsi_14"))
        adx = _safe(row.get("adx"))
        ema_align = _safe(row.get("ema_alignment"))
        plus_di = _safe(row.get("plus_di"))
        minus_di = _safe(row.get("minus_di"))
        macd_sig = row.get("macd_signal")
        macd_hist = _safe(row.get("macd_hist"))
        bb_pct = _safe(row.get("bb_pct"))
        stoch_k = _safe(row.get("stoch_rsi_k"))
        vol_ratio = _safe(row.get("volume_ratio")) or 0
        mfi = _safe(row.get("mfi"))
        williams_r = _safe(row.get("williams_r"))
        cci = _safe(row.get("cci"))
        tenkan = _safe(row.get("ichimoku_tenkan"))
        kijun = _safe(row.get("ichimoku_kijun"))
        senkou_a = _safe(row.get("ichimoku_senkou_a"))
        senkou_b = _safe(row.get("ichimoku_senkou_b"))
        rsi_div = row.get("rsi_divergence")

        bull_score = bear_score = 0

        # MACD (0-3)
        if macd_sig == "bullish_cross": bull_score += 3
        elif macd_sig == "bullish": bull_score += 1
        if macd_sig == "bearish_cross": bear_score += 3
        elif macd_sig == "bearish": bear_score += 1

        # EMA alignment (0-2)
        if ema_align is not None:
            if ema_align > 0.5: bull_score += 2
            elif ema_align > 0: bull_score += 1
            if ema_align < -0.5: bear_score += 2
            elif ema_align < 0: bear_score += 1

        # RSI (0-2)
        if rsi_14 is not None:
            if rsi_14 < 45: bull_score += 1
            if rsi_14 < 35: bull_score += 1
            if rsi_14 > 55: bear_score += 1
            if rsi_14 > 65: bear_score += 1

        # ADX (0-2)
        if adx is not None:
            if adx > 25: bull_score += 1; bear_score += 1
            if adx > 35: bull_score += 1; bear_score += 1

        # DI (0-1)
        if plus_di is not None and minus_di is not None:
            if plus_di > minus_di: bull_score += 1
            if minus_di > plus_di: bear_score += 1

        # Volume (0-2)
        if vol_ratio > 1.5: bull_score += 2; bear_score += 2
        elif vol_ratio > 1.0: bull_score += 1; bear_score += 1

        # StochRSI (0-2)
        if stoch_k is not None:
            if stoch_k < 25: bull_score += 2
            elif stoch_k < 40: bull_score += 1
            if stoch_k > 75: bear_score += 2
            elif stoch_k > 60: bear_score += 1

        # BB (0-1)
        if bb_pct is not None:
            if bb_pct < 0.25: bull_score += 1
            if bb_pct > 0.75: bear_score += 1

        # Ichimoku (0-2)
        if all(v is not None for v in [tenkan, kijun, senkou_a, senkou_b]):
            if price > max(senkou_a, senkou_b) and tenkan > kijun: bull_score += 2
            elif price > min(senkou_a, senkou_b): bull_score += 1
            if price < min(senkou_a, senkou_b) and tenkan < kijun: bear_score += 2
            elif price < max(senkou_a, senkou_b): bear_score += 1

        # Williams + CCI (0-2)
        if williams_r is not None:
            if williams_r < -75: bull_score += 1
            if williams_r > -25: bear_score += 1
        if cci is not None:
            if cci < -100: bull_score += 1
            if cci > 100: bear_score += 1

        # MFI (0-1)
        if mfi is not None:
            if mfi < 30: bull_score += 1
            if mfi > 70: bear_score += 1

        # RSI divergence (0-2)
        if rsi_div == "bullish_div": bull_score += 2
        elif rsi_div == "bearish_div": bear_score += 2

        if bull_score >= 6 and bull_score > bear_score + 2:
            signals.append({"bar": i, "dir": "LONG", "strat": "multi_confirm", "score": bull_score})
        elif bear_score >= 6 and bear_score > bull_score + 2:
            signals.append({"bar": i, "dir": "SHORT", "strat": "multi_confirm", "score": bear_score})
    return signals


def detect_breakout(df: pd.DataFrame) -> list[dict]:
    """#4 BB squeeze release + volume."""
    signals = []
    for i in range(2, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        prev_squeeze = prev.get("bb_squeeze") if not pd.isna(prev.get("bb_squeeze", np.nan)) else None
        curr_squeeze = row.get("bb_squeeze") if not pd.isna(row.get("bb_squeeze", np.nan)) else None
        vol_ratio = _safe(row.get("volume_ratio")) or 0
        macd_sig = row.get("macd_signal")
        ema_align = _safe(row.get("ema_alignment"))

        if prev_squeeze is None or curr_squeeze is None:
            continue
        if not (prev_squeeze is True and curr_squeeze is False):
            continue
        if vol_ratio < 1.5 or macd_sig is None or ema_align is None:
            continue

        score = 5
        if vol_ratio > 2.0: score += 1
        if vol_ratio > 3.0: score += 1

        if macd_sig in ("bullish", "bullish_cross") and ema_align > 0:
            signals.append({"bar": i, "dir": "LONG", "strat": "breakout", "score": score})
        elif macd_sig in ("bearish", "bearish_cross") and ema_align < 0:
            signals.append({"bar": i, "dir": "SHORT", "strat": "breakout", "score": score})
    return signals


def detect_mean_reversion(df: pd.DataFrame) -> list[dict]:
    """#5 RSI extreme + BB extreme + ADX low."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        rsi = _safe(row.get("rsi_14"))
        bb_pct = _safe(row.get("bb_pct"))
        stoch_k = _safe(row.get("stoch_rsi_k"))
        adx = _safe(row.get("adx"))

        if any(v is None for v in [rsi, bb_pct, stoch_k, adx]):
            continue

        if rsi < 22 and bb_pct < 0.05 and stoch_k < 10 and adx < 25:
            score = 5
            if rsi < 15: score += 1
            if bb_pct < 0.02: score += 1
            signals.append({"bar": i, "dir": "LONG", "strat": "mean_reversion", "score": score})
        elif rsi > 78 and bb_pct > 0.95 and stoch_k > 90 and adx < 25:
            score = 5
            if rsi > 85: score += 1
            if bb_pct > 0.98: score += 1
            signals.append({"bar": i, "dir": "SHORT", "strat": "mean_reversion", "score": score})
    return signals


def detect_momentum(df: pd.DataFrame) -> list[dict]:
    """#6 RSI + MACD histogram + ADX."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        rsi = _safe(row.get("rsi_14"))
        prev_rsi = _safe(prev.get("rsi_14"))
        adx = _safe(row.get("adx"))
        macd_sig = row.get("macd_signal")
        vol_ratio = _safe(row.get("volume_ratio")) or 0

        if any(v is None for v in [rsi, prev_rsi, adx]):
            continue
        if macd_sig is None:
            continue

        # RSI crosses above 50
        if (prev_rsi < 50 and rsi > 50 and adx > 25
                and macd_sig in ("bullish", "bullish_cross") and vol_ratio > 1.0):
            score = 5
            if adx > 35: score += 1
            if macd_sig == "bullish_cross": score += 1
            signals.append({"bar": i, "dir": "LONG", "strat": "momentum", "score": score})
        elif (prev_rsi > 50 and rsi < 50 and adx > 25
                and macd_sig in ("bearish", "bearish_cross") and vol_ratio > 1.0):
            score = 5
            if adx > 35: score += 1
            if macd_sig == "bearish_cross": score += 1
            signals.append({"bar": i, "dir": "SHORT", "strat": "momentum", "score": score})
    return signals


def detect_ema_cross(df: pd.DataFrame) -> list[dict]:
    """#7 EMA 9/21 crossover + EMA 50 filter."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        ema_9 = _safe(row.get("ema_9"))
        ema_21 = _safe(row.get("ema_21"))
        ema_50 = _safe(row.get("ema_50"))
        prev_ema_9 = _safe(prev.get("ema_9"))
        prev_ema_21 = _safe(prev.get("ema_21"))

        if any(v is None for v in [ema_9, ema_21, ema_50, prev_ema_9, prev_ema_21]):
            continue

        if prev_ema_9 < prev_ema_21 and ema_9 > ema_21 and ema_21 > ema_50:
            signals.append({"bar": i, "dir": "LONG", "strat": "ema_cross", "score": 5})
        elif prev_ema_9 > prev_ema_21 and ema_9 < ema_21 and ema_21 < ema_50:
            signals.append({"bar": i, "dir": "SHORT", "strat": "ema_cross", "score": 5})
    return signals


def detect_bollinger_bounce(df: pd.DataFrame) -> list[dict]:
    """#8 Bounce at BB bands in ranging market."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        bb_pct = _safe(row.get("bb_pct"))
        adx = _safe(row.get("adx"))
        rsi = _safe(row.get("rsi_14"))

        if any(v is None for v in [bb_pct, adx, rsi]):
            continue

        if bb_pct < 0.1 and adx < 30 and rsi < 40:
            score = 5
            if bb_pct < 0.05: score += 1
            if rsi < 30: score += 1
            signals.append({"bar": i, "dir": "LONG", "strat": "bollinger_bounce", "score": score})
        elif bb_pct > 0.9 and adx < 30 and rsi > 60:
            score = 5
            if bb_pct > 0.95: score += 1
            if rsi > 70: score += 1
            signals.append({"bar": i, "dir": "SHORT", "strat": "bollinger_bounce", "score": score})
    return signals


def detect_volume_spike(df: pd.DataFrame) -> list[dict]:
    """#9 Volume spike + directional bias."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        vol_ratio = _safe(row.get("volume_ratio")) or 0
        adx = _safe(row.get("adx"))
        macd_sig = row.get("macd_signal")
        ema_align = _safe(row.get("ema_alignment"))

        if adx is None or macd_sig is None or ema_align is None:
            continue
        if vol_ratio <= 2.5 or adx <= 20:
            continue

        score = 5
        if vol_ratio > 3.5: score += 1
        if vol_ratio > 5.0: score += 1

        if macd_sig in ("bullish", "bullish_cross") and ema_align > 0:
            signals.append({"bar": i, "dir": "LONG", "strat": "volume_spike", "score": score})
        elif macd_sig in ("bearish", "bearish_cross") and ema_align < 0:
            signals.append({"bar": i, "dir": "SHORT", "strat": "volume_spike", "score": score})
    return signals


def detect_stoch_reversal(df: pd.DataFrame) -> list[dict]:
    """#10 StochRSI cross from extreme zones."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        stoch_k = _safe(row.get("stoch_rsi_k"))
        prev_stoch_k = _safe(prev.get("stoch_rsi_k"))
        bb_pct = _safe(row.get("bb_pct"))

        if any(v is None for v in [stoch_k, prev_stoch_k, bb_pct]):
            continue

        if prev_stoch_k < 20 and stoch_k > 20 and bb_pct < 0.3:
            signals.append({"bar": i, "dir": "LONG", "strat": "stoch_reversal", "score": 5})
        elif prev_stoch_k > 80 and stoch_k < 80 and bb_pct > 0.7:
            signals.append({"bar": i, "dir": "SHORT", "strat": "stoch_reversal", "score": 5})
    return signals


def detect_ichimoku_cloud(df: pd.DataFrame) -> list[dict]:
    """#11 Cloud breakout + TK cross."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        price = _safe(row.get("close"))
        senkou_a = _safe(row.get("ichimoku_senkou_a"))
        senkou_b = _safe(row.get("ichimoku_senkou_b"))
        tenkan = _safe(row.get("ichimoku_tenkan"))
        kijun = _safe(row.get("ichimoku_kijun"))

        if any(v is None for v in [price, senkou_a, senkou_b, tenkan, kijun]):
            continue

        if price > senkou_a and price > senkou_b and tenkan > kijun:
            score = 5
            cloud_dist = (price - max(senkou_a, senkou_b)) / price * 100
            if cloud_dist > 1: score += 1
            signals.append({"bar": i, "dir": "LONG", "strat": "ichimoku_cloud", "score": score})
        elif price < senkou_a and price < senkou_b and tenkan < kijun:
            score = 5
            signals.append({"bar": i, "dir": "SHORT", "strat": "ichimoku_cloud", "score": score})
    return signals


def detect_williams_cci_combo(df: pd.DataFrame) -> list[dict]:
    """#12 Williams %R + CCI extremes."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        wr = _safe(row.get("williams_r"))
        cci = _safe(row.get("cci"))
        vol_ratio = _safe(row.get("volume_ratio")) or 0

        if wr is None or cci is None:
            continue

        if wr < -80 and cci < -100 and vol_ratio > 0.8:
            score = 5
            if wr < -90: score += 1
            if cci < -200: score += 1
            signals.append({"bar": i, "dir": "LONG", "strat": "williams_cci_combo", "score": score})
        elif wr > -20 and cci > 100:
            score = 5
            if wr > -10: score += 1
            if cci > 200: score += 1
            signals.append({"bar": i, "dir": "SHORT", "strat": "williams_cci_combo", "score": score})
    return signals


# ===========================================================================
# NEW STRATEGIES (15)
# ===========================================================================

def detect_supertrend(df: pd.DataFrame) -> list[dict]:
    """#13 Supertrend flip — ATR-based trend follower."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        st_dir = _safe(row.get("supertrend_dir"))
        prev_dir = _safe(row.get("prev_supertrend_dir"))
        adx = _safe(row.get("adx"))

        if st_dir is None or prev_dir is None:
            continue

        score = 5
        if adx is not None and adx > 25: score += 1
        if adx is not None and adx > 35: score += 1

        if prev_dir == -1 and st_dir == 1:
            signals.append({"bar": i, "dir": "LONG", "strat": "supertrend", "score": score})
        elif prev_dir == 1 and st_dir == -1:
            signals.append({"bar": i, "dir": "SHORT", "strat": "supertrend", "score": score})
    return signals


def detect_donchian_breakout(df: pd.DataFrame) -> list[dict]:
    """#14 Donchian channel breakout."""
    signals = []
    for i in range(2, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        close = _safe(row.get("close"))
        prev_close = _safe(prev.get("close"))
        dc_upper = _safe(row.get("donchian_upper"))
        dc_lower = _safe(row.get("donchian_lower"))
        vol_ratio = _safe(row.get("volume_ratio")) or 0

        if any(v is None for v in [close, prev_close, dc_upper, dc_lower]):
            continue

        score = 5
        if vol_ratio > 1.5: score += 1
        if vol_ratio > 2.0: score += 1

        # Breakout above upper Donchian
        if close > dc_upper and prev_close <= dc_upper:
            signals.append({"bar": i, "dir": "LONG", "strat": "donchian_breakout", "score": score})
        # Breakout below lower Donchian
        elif close < dc_lower and prev_close >= dc_lower:
            signals.append({"bar": i, "dir": "SHORT", "strat": "donchian_breakout", "score": score})
    return signals


def detect_parabolic_sar(df: pd.DataFrame) -> list[dict]:
    """#15 PSAR flip + EMA confirmation."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        psar_dir = _safe(row.get("psar_dir"))
        prev_psar_dir = _safe(row.get("prev_psar_dir"))
        ema_align = _safe(row.get("ema_alignment"))

        if psar_dir is None or prev_psar_dir is None:
            continue

        score = 5
        if ema_align is not None:
            if abs(ema_align) > 0.3: score += 1
            if abs(ema_align) > 0.5: score += 1

        if prev_psar_dir == -1 and psar_dir == 1:
            if ema_align is None or ema_align >= 0:
                signals.append({"bar": i, "dir": "LONG", "strat": "parabolic_sar", "score": score})
        elif prev_psar_dir == 1 and psar_dir == -1:
            if ema_align is None or ema_align <= 0:
                signals.append({"bar": i, "dir": "SHORT", "strat": "parabolic_sar", "score": score})
    return signals


def detect_hull_trend(df: pd.DataFrame) -> list[dict]:
    """#16 Hull MA crossover (faster than EMA)."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        hull_9 = _safe(row.get("hull_9"))
        hull_21 = _safe(row.get("hull_21"))
        prev_hull_9 = _safe(row.get("prev_hull_9"))
        prev_hull_21 = _safe(row.get("prev_hull_21"))
        adx = _safe(row.get("adx"))

        if any(v is None for v in [hull_9, hull_21, prev_hull_9, prev_hull_21]):
            continue

        score = 5
        if adx is not None and adx > 25: score += 1

        if prev_hull_9 < prev_hull_21 and hull_9 > hull_21:
            signals.append({"bar": i, "dir": "LONG", "strat": "hull_trend", "score": score})
        elif prev_hull_9 > prev_hull_21 and hull_9 < hull_21:
            signals.append({"bar": i, "dir": "SHORT", "strat": "hull_trend", "score": score})
    return signals


def detect_roc_momentum(df: pd.DataFrame) -> list[dict]:
    """#17 Rate of Change exceeds threshold."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        roc_12 = _safe(row.get("roc_12"))
        prev_roc = _safe(prev.get("roc_12"))
        adx = _safe(row.get("adx"))
        ema_align = _safe(row.get("ema_alignment"))

        if roc_12 is None or prev_roc is None:
            continue

        score = 5
        if adx is not None and adx > 25: score += 1
        if ema_align is not None and abs(ema_align) > 0.3: score += 1

        # ROC crosses above threshold
        if roc_12 > 2.0 and prev_roc <= 2.0:
            if ema_align is None or ema_align >= 0:
                signals.append({"bar": i, "dir": "LONG", "strat": "roc_momentum", "score": score})
        elif roc_12 < -2.0 and prev_roc >= -2.0:
            if ema_align is None or ema_align <= 0:
                signals.append({"bar": i, "dir": "SHORT", "strat": "roc_momentum", "score": score})
    return signals


def detect_zscore_reversion(df: pd.DataFrame) -> list[dict]:
    """#18 Z-Score extreme -> reversion to mean."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        z = _safe(row.get("zscore_20"))
        adx = _safe(row.get("adx"))
        rsi = _safe(row.get("rsi_14"))

        if z is None:
            continue

        score = 5
        if adx is not None and adx < 25: score += 1  # ranging = better for reversion

        # Extremely oversold
        if z < -2.0:
            if rsi is None or rsi < 40:
                if z < -2.5: score += 1
                if z < -3.0: score += 1
                signals.append({"bar": i, "dir": "LONG", "strat": "zscore_reversion", "score": score})
        elif z > 2.0:
            if rsi is None or rsi > 60:
                if z > 2.5: score += 1
                if z > 3.0: score += 1
                signals.append({"bar": i, "dir": "SHORT", "strat": "zscore_reversion", "score": score})
    return signals


def detect_vol_squeeze_breakout(df: pd.DataFrame) -> list[dict]:
    """#19 BB squeeze inside Keltner -> explosion on release."""
    signals = []
    for i in range(2, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        prev2 = df.iloc[i - 2]

        kc_squeeze = row.get("kc_squeeze")
        prev_kc = prev.get("kc_squeeze")
        prev2_kc = prev2.get("kc_squeeze")
        macd_hist = _safe(row.get("macd_hist"))
        vol_ratio = _safe(row.get("volume_ratio")) or 0
        ema_align = _safe(row.get("ema_alignment"))

        if kc_squeeze is None or prev_kc is None:
            continue

        # Squeeze release: was in squeeze for 2+ bars, now released
        if prev_kc is True and prev2_kc is True and kc_squeeze is False:
            score = 5
            if vol_ratio > 1.5: score += 1
            if vol_ratio > 2.0: score += 1

            if macd_hist is not None and macd_hist > 0:
                if ema_align is None or ema_align >= 0:
                    signals.append({"bar": i, "dir": "LONG", "strat": "vol_squeeze_breakout", "score": score})
            elif macd_hist is not None and macd_hist < 0:
                if ema_align is None or ema_align <= 0:
                    signals.append({"bar": i, "dir": "SHORT", "strat": "vol_squeeze_breakout", "score": score})
    return signals


def detect_rsi_divergence_entry(df: pd.DataFrame) -> list[dict]:
    """#20 RSI divergence as entry signal."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        rsi_div = row.get("rsi_divergence")
        rsi = _safe(row.get("rsi_14"))
        stoch_k = _safe(row.get("stoch_rsi_k"))

        if rsi_div is None or rsi_div == "none":
            continue

        score = 5
        if rsi is not None:
            if rsi_div == "bullish_div" and rsi < 40: score += 1
            if rsi_div == "bearish_div" and rsi > 60: score += 1
        if stoch_k is not None:
            if rsi_div == "bullish_div" and stoch_k < 30: score += 1
            if rsi_div == "bearish_div" and stoch_k > 70: score += 1

        if rsi_div == "bullish_div":
            signals.append({"bar": i, "dir": "LONG", "strat": "rsi_divergence_entry", "score": score})
        elif rsi_div == "bearish_div":
            signals.append({"bar": i, "dir": "SHORT", "strat": "rsi_divergence_entry", "score": score})
    return signals


def detect_obv_divergence(df: pd.DataFrame) -> list[dict]:
    """#21 Price vs OBV diverge."""
    signals = []
    if "obv_slope" not in df.columns or "obv" not in df.columns:
        return signals

    for i in range(20, len(df)):
        row = df.iloc[i]
        close = _safe(row.get("close"))
        obv_sl = _safe(row.get("obv_slope"))
        roc_12 = _safe(row.get("roc_12"))

        if close is None or obv_sl is None or roc_12 is None:
            continue

        score = 5

        # Price falling but OBV rising -> bullish divergence
        if roc_12 < -1.0 and obv_sl > 0:
            signals.append({"bar": i, "dir": "LONG", "strat": "obv_divergence", "score": score})
        # Price rising but OBV falling -> bearish divergence
        elif roc_12 > 1.0 and obv_sl < 0:
            signals.append({"bar": i, "dir": "SHORT", "strat": "obv_divergence", "score": score})
    return signals


def detect_vwap_reversion(df: pd.DataFrame) -> list[dict]:
    """#22 Price deviates from rolling VWAP -> reversion."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        close = _safe(row.get("close"))
        vwap = _safe(row.get("rolling_vwap"))
        adx = _safe(row.get("adx"))

        if close is None or vwap is None or vwap == 0:
            continue

        deviation_pct = (close - vwap) / vwap * 100
        score = 5

        if adx is not None and adx < 30: score += 1  # ranging better

        # Price far below VWAP -> buy
        if deviation_pct < -2.0:
            if deviation_pct < -3.0: score += 1
            signals.append({"bar": i, "dir": "LONG", "strat": "vwap_reversion", "score": score})
        elif deviation_pct > 2.0:
            if deviation_pct > 3.0: score += 1
            signals.append({"bar": i, "dir": "SHORT", "strat": "vwap_reversion", "score": score})
    return signals


def detect_mtf_confirmation(df_1h: pd.DataFrame, df_4h: pd.DataFrame = None) -> list[dict]:
    """#23 Multi-timeframe: 1H signal + 4H trend alignment.

    If df_4h is None, uses 1H indicators only with stricter alignment.
    If df_4h is provided, maps 1H bars to 4H bars for confirmation.
    """
    signals = []

    # Build 4H context mapping if available
    context_4h = {}
    if df_4h is not None and len(df_4h) > 0 and "ema_alignment" in df_4h.columns:
        for _, row_4h in df_4h.iterrows():
            ts = row_4h.get("timestamp")
            if ts is not None:
                context_4h[ts] = {
                    "ema_align_4h": _safe(row_4h.get("ema_alignment")),
                    "adx_4h": _safe(row_4h.get("adx")),
                    "macd_sig_4h": row_4h.get("macd_signal"),
                }

    for i in range(1, len(df_1h)):
        row = df_1h.iloc[i]
        macd_sig = row.get("macd_signal")
        ema_align = _safe(row.get("ema_alignment"))
        adx = _safe(row.get("adx"))
        vol_ratio = _safe(row.get("volume_ratio")) or 0

        if macd_sig is None or ema_align is None:
            continue

        # Get 4H context
        ema_align_4h = None
        adx_4h = None
        macd_sig_4h = None

        if context_4h:
            ts_1h = row.get("timestamp")
            if ts_1h is not None:
                if not isinstance(ts_1h, pd.Timestamp):
                    ts_1h = pd.Timestamp(ts_1h)
                ts_4h = ts_1h.floor("4h")
                ctx = context_4h.get(ts_4h, {})
                ema_align_4h = ctx.get("ema_align_4h")
                adx_4h = ctx.get("adx_4h")
                macd_sig_4h = ctx.get("macd_sig_4h")

        score = 5

        # LONG: 1H bullish + 4H bullish
        if macd_sig in ("bullish", "bullish_cross") and ema_align > 0.3:
            if ema_align_4h is not None:
                if ema_align_4h > 0:
                    score += 2
                    if macd_sig_4h in ("bullish", "bullish_cross"): score += 1
                else:
                    continue  # 4H disagrees, skip
            else:
                # No 4H data: require stronger 1H signal
                if ema_align <= 0.5:
                    continue
                score += 1

            if adx is not None and adx > 25: score += 1
            if vol_ratio > 1.2: score += 1
            signals.append({"bar": i, "dir": "LONG", "strat": "mtf_confirmation", "score": score})

        # SHORT: mirror
        elif macd_sig in ("bearish", "bearish_cross") and ema_align < -0.3:
            if ema_align_4h is not None:
                if ema_align_4h < 0:
                    score += 2
                    if macd_sig_4h in ("bearish", "bearish_cross"): score += 1
                else:
                    continue
            else:
                if ema_align >= -0.5:
                    continue
                score += 1

            if adx is not None and adx > 25: score += 1
            if vol_ratio > 1.2: score += 1
            signals.append({"bar": i, "dir": "SHORT", "strat": "mtf_confirmation", "score": score})

    return signals


def detect_regime_adaptive(df: pd.DataFrame) -> list[dict]:
    """#24 Adaptive: trend follow if ADX>30, mean reversion if ADX<20."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        adx = _safe(row.get("adx"))
        if adx is None:
            continue

        if adx > 30:
            # Trend follow mode
            ema_align = _safe(row.get("ema_alignment"))
            plus_di = _safe(row.get("plus_di"))
            minus_di = _safe(row.get("minus_di"))
            macd_sig = row.get("macd_signal")

            if any(v is None for v in [ema_align, plus_di, minus_di]):
                continue

            score = 5
            if adx > 40: score += 1

            if ema_align > 0.3 and plus_di > minus_di and macd_sig in ("bullish", "bullish_cross"):
                signals.append({"bar": i, "dir": "LONG", "strat": "regime_adaptive", "score": score})
            elif ema_align < -0.3 and minus_di > plus_di and macd_sig in ("bearish", "bearish_cross"):
                signals.append({"bar": i, "dir": "SHORT", "strat": "regime_adaptive", "score": score})

        elif adx < 20:
            # Mean reversion mode
            rsi = _safe(row.get("rsi_14"))
            bb_pct = _safe(row.get("bb_pct"))
            stoch_k = _safe(row.get("stoch_rsi_k"))

            if any(v is None for v in [rsi, bb_pct]):
                continue

            score = 5

            if rsi < 30 and bb_pct < 0.15:
                if stoch_k is not None and stoch_k < 20: score += 1
                signals.append({"bar": i, "dir": "LONG", "strat": "regime_adaptive", "score": score})
            elif rsi > 70 and bb_pct > 0.85:
                if stoch_k is not None and stoch_k > 80: score += 1
                signals.append({"bar": i, "dir": "SHORT", "strat": "regime_adaptive", "score": score})
    return signals


def detect_fair_value_gap(df: pd.DataFrame) -> list[dict]:
    """#25 Fair Value Gap approximation (Smart Money concept).

    FVG bullish: candle[i-2].high < candle[i].low (gap between bodies).
    """
    signals = []
    for i in range(3, len(df)):
        row = df.iloc[i]
        c2 = df.iloc[i - 2]
        c1 = df.iloc[i - 1]  # the big candle that creates the gap

        high_2 = _safe(c2.get("high"))
        low_0 = _safe(row.get("low"))
        high_0 = _safe(row.get("high"))
        low_2 = _safe(c2.get("low"))
        close = _safe(row.get("close"))
        vol_ratio = _safe(row.get("volume_ratio")) or 0

        if any(v is None for v in [high_2, low_0, high_0, low_2, close]):
            continue

        score = 5
        if vol_ratio > 1.5: score += 1

        # Bullish FVG: gap up (candle 2 ago's high < current candle's low)
        if high_2 < low_0:
            gap_size = (low_0 - high_2) / close * 100
            if gap_size > 0.3:  # meaningful gap
                if gap_size > 0.5: score += 1
                signals.append({"bar": i, "dir": "LONG", "strat": "fair_value_gap", "score": score})

        # Bearish FVG: gap down
        elif low_2 > high_0:
            gap_size = (low_2 - high_0) / close * 100
            if gap_size > 0.3:
                if gap_size > 0.5: score += 1
                signals.append({"bar": i, "dir": "SHORT", "strat": "fair_value_gap", "score": score})
    return signals


def detect_consecutive_reversal(df: pd.DataFrame) -> list[dict]:
    """#26 After N consecutive red/green candles -> counter-trade."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        consec = _safe(row.get("consecutive_direction"))
        rsi = _safe(row.get("rsi_14"))

        if consec is None:
            continue

        score = 5

        # After 4+ consecutive red candles -> potential bounce (LONG)
        if consec <= -4:
            if rsi is not None and rsi < 40: score += 1
            if consec <= -5: score += 1
            if consec <= -6: score += 1
            signals.append({"bar": i, "dir": "LONG", "strat": "consecutive_reversal", "score": score})

        # After 4+ consecutive green candles -> potential pullback (SHORT)
        elif consec >= 4:
            if rsi is not None and rsi > 60: score += 1
            if consec >= 5: score += 1
            if consec >= 6: score += 1
            signals.append({"bar": i, "dir": "SHORT", "strat": "consecutive_reversal", "score": score})
    return signals


def detect_triple_ema_adx(df: pd.DataFrame) -> list[dict]:
    """#27 Triple EMA stack (9>21>50) + strong ADX."""
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        ema_9 = _safe(row.get("ema_9"))
        ema_21 = _safe(row.get("ema_21"))
        ema_50 = _safe(row.get("ema_50"))
        adx = _safe(row.get("adx"))
        plus_di = _safe(row.get("plus_di"))
        minus_di = _safe(row.get("minus_di"))

        prev_ema_align = _safe(prev.get("ema_alignment"))
        curr_ema_align = _safe(row.get("ema_alignment"))

        if any(v is None for v in [ema_9, ema_21, ema_50, adx]):
            continue

        score = 5
        if adx > 30: score += 1
        if adx > 40: score += 1

        # Bullish: perfect stack just formed (prev wasn't perfect, now it is)
        if ema_9 > ema_21 > ema_50 and adx > 25:
            if prev_ema_align is not None and prev_ema_align < 1.0 and curr_ema_align == 1.0:
                if plus_di is not None and minus_di is not None and plus_di > minus_di:
                    signals.append({"bar": i, "dir": "LONG", "strat": "triple_ema_adx", "score": score})

        # Bearish: perfect bearish stack
        elif ema_9 < ema_21 < ema_50 and adx > 25:
            if prev_ema_align is not None and prev_ema_align > -1.0 and curr_ema_align == -1.0:
                if plus_di is not None and minus_di is not None and minus_di > plus_di:
                    signals.append({"bar": i, "dir": "SHORT", "strat": "triple_ema_adx", "score": score})
    return signals


# ===========================================================================
# Registry: all 27 detectors
# ===========================================================================

STRATEGY_DETECTORS = {
    # Existing (12)
    "macd_cross_1h": detect_macd_cross_1h,
    "trend_follow_1h": detect_trend_follow_1h,
    "multi_confirm": detect_multi_confirm,
    "breakout": detect_breakout,
    "mean_reversion": detect_mean_reversion,
    "momentum": detect_momentum,
    "ema_cross": detect_ema_cross,
    "bollinger_bounce": detect_bollinger_bounce,
    "volume_spike": detect_volume_spike,
    "stoch_reversal": detect_stoch_reversal,
    "ichimoku_cloud": detect_ichimoku_cloud,
    "williams_cci_combo": detect_williams_cci_combo,
    # New (15)
    "supertrend": detect_supertrend,
    "donchian_breakout": detect_donchian_breakout,
    "parabolic_sar": detect_parabolic_sar,
    "hull_trend": detect_hull_trend,
    "roc_momentum": detect_roc_momentum,
    "zscore_reversion": detect_zscore_reversion,
    "vol_squeeze_breakout": detect_vol_squeeze_breakout,
    "rsi_divergence_entry": detect_rsi_divergence_entry,
    "obv_divergence": detect_obv_divergence,
    "vwap_reversion": detect_vwap_reversion,
    "mtf_confirmation": detect_mtf_confirmation,
    "regime_adaptive": detect_regime_adaptive,
    "fair_value_gap": detect_fair_value_gap,
    "consecutive_reversal": detect_consecutive_reversal,
    "triple_ema_adx": detect_triple_ema_adx,
}

# mtf_confirmation has special signature (needs df_4h), handled separately
MTF_STRATEGIES = {"mtf_confirmation"}


def detect_all_signals(df: pd.DataFrame, df_4h: pd.DataFrame = None) -> dict[str, list[dict]]:
    """Run all 27 detectors on a single pair's DataFrame.

    Returns: {strategy_name: [signal_dicts]}
    """
    results = {}
    for name, func in STRATEGY_DETECTORS.items():
        if name in MTF_STRATEGIES:
            results[name] = func(df, df_4h)
        else:
            results[name] = func(df)
    return results
