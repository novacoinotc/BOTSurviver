"""Backtesting strategy functions.

Each strategy takes a DataFrame row (pd.Series with indicator columns) and a
params dict, returning an Optional[dict] signal or None.

Signal format::

    {
        "direction": "LONG" | "SHORT",
        "sl_distance_pct": float,
        "tp_distance_pct": float,
        "trailing_pct": float,
    }

All strategies guard against NaN / None values on required indicators and
return None when data is insufficient.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(val) -> Optional[float]:
    """Return *val* as float if it is a valid finite number, else None."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _safe_str(val) -> Optional[str]:
    """Return *val* as str if it is a non-empty string, else None."""
    if val is None:
        return None
    s = str(val)
    return s if s and s != "nan" else None


def _build_signal(direction: str, params: dict, default_sl: float, default_tp: float) -> dict:
    """Construct a signal dict using params with sensible defaults."""
    return {
        "direction": direction,
        "sl_distance_pct": params.get("sl_pct", default_sl),
        "tp_distance_pct": params.get("tp_pct", default_tp),
        "trailing_pct": params.get("trailing_pct", 0.015),
    }


# ---------------------------------------------------------------------------
# 1. Trend Follow – pullback entry in established trend
# ---------------------------------------------------------------------------

def trend_follow(row: pd.Series, params: dict) -> Optional[dict]:
    """Pullback entry in an established trend.

    LONG: ADX > adx_min, ema_alignment > ema_align_min, plus_di > minus_di,
          rsi_low <= RSI(14) <= rsi_high, MACD bullish/bullish_cross,
          ema_trend_5m bullish.
    SHORT: mirror conditions.

    Default params: adx_min=30, ema_align_min=0.5, rsi_low=35, rsi_high=55
    """
    adx = _safe(row.get("adx"))
    ema_align = _safe(row.get("ema_alignment"))
    plus_di = _safe(row.get("plus_di"))
    minus_di = _safe(row.get("minus_di"))
    rsi = _safe(row.get("rsi_14"))
    macd_sig = _safe_str(row.get("macd_signal"))
    ema_trend = _safe_str(row.get("ema_trend_5m"))

    if any(v is None for v in [adx, ema_align, plus_di, minus_di, rsi]):
        return None
    if macd_sig is None or ema_trend is None:
        return None

    adx_min = params.get("adx_min", 30)
    ema_align_min = params.get("ema_align_min", 0.5)
    rsi_low = params.get("rsi_low", 35)
    rsi_high = params.get("rsi_high", 55)

    # LONG conditions
    if (
        adx > adx_min
        and ema_align > ema_align_min
        and plus_di > minus_di
        and rsi_low <= rsi <= rsi_high
        and macd_sig in ("bullish", "bullish_cross")
        and ema_trend in ("bullish", "strong_bullish")
    ):
        return _build_signal("LONG", params, default_sl=0.012, default_tp=0.030)

    # SHORT conditions (mirror)
    if (
        adx > adx_min
        and ema_align < -ema_align_min
        and minus_di > plus_di
        and (100 - rsi_high) <= rsi <= (100 - rsi_low)
        and macd_sig in ("bearish", "bearish_cross")
        and ema_trend in ("bearish", "strong_bearish")
    ):
        return _build_signal("SHORT", params, default_sl=0.012, default_tp=0.030)

    return None


# ---------------------------------------------------------------------------
# 2. Breakout – BB squeeze release with volume
# ---------------------------------------------------------------------------

def breakout(row: pd.Series, params: dict) -> Optional[dict]:
    """Bollinger Band squeeze release with volume confirmation.

    Requires previous bar's bb_squeeze=True, current bb_squeeze=False.
    volume_ratio > vol_min. Direction from MACD + EMA alignment.

    Pass previous bar's squeeze state via params["prev_bb_squeeze"].
    Default params: vol_min=1.5
    """
    bb_squeeze = row.get("bb_squeeze")
    vol_ratio = _safe(row.get("volume_ratio"))
    macd_sig = _safe_str(row.get("macd_signal"))
    ema_align = _safe(row.get("ema_alignment"))
    prev_bb_squeeze = params.get("prev_bb_squeeze")

    if bb_squeeze is None or vol_ratio is None or macd_sig is None or ema_align is None:
        return None
    if prev_bb_squeeze is None:
        return None

    vol_min = params.get("vol_min", 1.5)

    # Squeeze release: was squeezed, now not
    if not (prev_bb_squeeze is True and bb_squeeze is False):
        return None

    if vol_ratio < vol_min:
        return None

    # Determine direction
    if macd_sig in ("bullish", "bullish_cross") and ema_align > 0:
        return _build_signal("LONG", params, default_sl=0.012, default_tp=0.035)
    elif macd_sig in ("bearish", "bearish_cross") and ema_align < 0:
        return _build_signal("SHORT", params, default_sl=0.012, default_tp=0.035)

    return None


# ---------------------------------------------------------------------------
# 3. Mean Reversion – extreme oversold/overbought in ranging markets
# ---------------------------------------------------------------------------

def mean_reversion(row: pd.Series, params: dict) -> Optional[dict]:
    """Extreme oversold/overbought in ranging (low ADX) markets.

    LONG: RSI<rsi_oversold, bb_pct<bb_low, stoch_rsi_k<stoch_low, ADX<adx_max.
    SHORT: mirror.

    Default params: rsi_oversold=22, rsi_overbought=78, bb_low=0.05,
                    bb_high=0.95, stoch_low=10, stoch_high=90, adx_max=25
    """
    rsi = _safe(row.get("rsi_14"))
    bb_pct = _safe(row.get("bb_pct"))
    stoch_k = _safe(row.get("stoch_rsi_k"))
    adx = _safe(row.get("adx"))

    if any(v is None for v in [rsi, bb_pct, stoch_k, adx]):
        return None

    rsi_oversold = params.get("rsi_oversold", 22)
    rsi_overbought = params.get("rsi_overbought", 78)
    bb_low = params.get("bb_low", 0.05)
    bb_high = params.get("bb_high", 0.95)
    stoch_low = params.get("stoch_low", 10)
    stoch_high = params.get("stoch_high", 90)
    adx_max = params.get("adx_max", 25)

    # LONG
    if rsi < rsi_oversold and bb_pct < bb_low and stoch_k < stoch_low and adx < adx_max:
        return _build_signal("LONG", params, default_sl=0.010, default_tp=0.020)

    # SHORT
    if rsi > rsi_overbought and bb_pct > bb_high and stoch_k > stoch_high and adx < adx_max:
        return _build_signal("SHORT", params, default_sl=0.010, default_tp=0.020)

    return None


# ---------------------------------------------------------------------------
# 4. Momentum – strong momentum continuation
# ---------------------------------------------------------------------------

def momentum(row: pd.Series, params: dict) -> Optional[dict]:
    """Strong momentum continuation.

    LONG: RSI crosses above 50 (prev<50, curr>50), ADX > adx_min,
          MACD bullish, volume_ratio > 1.0.
    SHORT: mirror.

    Pass params["prev_rsi_14"] for crossover detection.
    Default params: adx_min=25
    """
    rsi = _safe(row.get("rsi_14"))
    adx = _safe(row.get("adx"))
    macd_sig = _safe_str(row.get("macd_signal"))
    vol_ratio = _safe(row.get("volume_ratio"))
    prev_rsi = _safe(params.get("prev_rsi_14"))

    if any(v is None for v in [rsi, adx, vol_ratio]):
        return None
    if macd_sig is None or prev_rsi is None:
        return None

    adx_min = params.get("adx_min", 25)

    # LONG: RSI crosses above 50
    if (
        prev_rsi < 50
        and rsi > 50
        and adx > adx_min
        and macd_sig in ("bullish", "bullish_cross")
        and vol_ratio > 1.0
    ):
        return _build_signal("LONG", params, default_sl=0.012, default_tp=0.030)

    # SHORT: RSI crosses below 50
    if (
        prev_rsi > 50
        and rsi < 50
        and adx > adx_min
        and macd_sig in ("bearish", "bearish_cross")
        and vol_ratio > 1.0
    ):
        return _build_signal("SHORT", params, default_sl=0.012, default_tp=0.030)

    return None


# ---------------------------------------------------------------------------
# 5. MACD Cross – pure MACD crossover
# ---------------------------------------------------------------------------

def macd_cross(row: pd.Series, params: dict) -> Optional[dict]:
    """Pure MACD crossover with EMA alignment and volume filters.

    LONG: macd_signal=="bullish_cross", ema_alignment>=0, volume_ratio>vol_min.
    SHORT: macd_signal=="bearish_cross", ema_alignment<=0.

    Default params: vol_min=0.8
    """
    macd_sig = _safe_str(row.get("macd_signal"))
    ema_align = _safe(row.get("ema_alignment"))
    vol_ratio = _safe(row.get("volume_ratio"))

    if macd_sig is None or ema_align is None or vol_ratio is None:
        return None

    vol_min = params.get("vol_min", 0.8)

    # LONG
    if macd_sig == "bullish_cross" and ema_align >= 0 and vol_ratio > vol_min:
        return _build_signal("LONG", params, default_sl=0.012, default_tp=0.025)

    # SHORT
    if macd_sig == "bearish_cross" and ema_align <= 0:
        return _build_signal("SHORT", params, default_sl=0.012, default_tp=0.025)

    return None


# ---------------------------------------------------------------------------
# 6. EMA Cross – EMA crossover strategy
# ---------------------------------------------------------------------------

def ema_cross(row: pd.Series, params: dict) -> Optional[dict]:
    """EMA 9/21 crossover with EMA 50 trend filter.

    LONG: ema_9 crosses above ema_21 (prev ema_9<ema_21, curr ema_9>ema_21),
          ema_21 > ema_50 (trend filter).
    SHORT: mirror.

    Pass params["prev_ema_9"] and params["prev_ema_21"] for crossover detection.
    """
    ema_9 = _safe(row.get("ema_9"))
    ema_21 = _safe(row.get("ema_21"))
    ema_50 = _safe(row.get("ema_50"))
    prev_ema_9 = _safe(params.get("prev_ema_9"))
    prev_ema_21 = _safe(params.get("prev_ema_21"))

    if any(v is None for v in [ema_9, ema_21, ema_50, prev_ema_9, prev_ema_21]):
        return None

    # LONG: ema_9 crosses above ema_21, trend filter ema_21 > ema_50
    if prev_ema_9 < prev_ema_21 and ema_9 > ema_21 and ema_21 > ema_50:
        return _build_signal("LONG", params, default_sl=0.012, default_tp=0.030)

    # SHORT: ema_9 crosses below ema_21, trend filter ema_21 < ema_50
    if prev_ema_9 > prev_ema_21 and ema_9 < ema_21 and ema_21 < ema_50:
        return _build_signal("SHORT", params, default_sl=0.012, default_tp=0.030)

    return None


# ---------------------------------------------------------------------------
# 7. Bollinger Bounce – buy lower band, sell upper band in ranging market
# ---------------------------------------------------------------------------

def bollinger_bounce(row: pd.Series, params: dict) -> Optional[dict]:
    """Buy at lower Bollinger Band, sell at upper band in ranging markets.

    LONG: bb_pct < bb_entry_low, ADX < adx_max, RSI < rsi_max.
    SHORT: bb_pct > bb_entry_high, ADX < adx_max, RSI > rsi_min.

    Default params: bb_entry_low=0.1, bb_entry_high=0.9, adx_max=30,
                    rsi_max=40, rsi_min=60
    """
    bb_pct = _safe(row.get("bb_pct"))
    adx = _safe(row.get("adx"))
    rsi = _safe(row.get("rsi_14"))

    if any(v is None for v in [bb_pct, adx, rsi]):
        return None

    bb_entry_low = params.get("bb_entry_low", 0.1)
    bb_entry_high = params.get("bb_entry_high", 0.9)
    adx_max = params.get("adx_max", 30)
    rsi_max = params.get("rsi_max", 40)
    rsi_min = params.get("rsi_min", 60)

    # LONG
    if bb_pct < bb_entry_low and adx < adx_max and rsi < rsi_max:
        return _build_signal("LONG", params, default_sl=0.010, default_tp=0.020)

    # SHORT
    if bb_pct > bb_entry_high and adx < adx_max and rsi > rsi_min:
        return _build_signal("SHORT", params, default_sl=0.010, default_tp=0.020)

    return None


# ---------------------------------------------------------------------------
# 8. Volume Spike – sudden volume spike + directional bias
# ---------------------------------------------------------------------------

def volume_spike(row: pd.Series, params: dict) -> Optional[dict]:
    """Sudden volume spike with directional confirmation.

    volume_ratio > vol_spike AND ADX > 20.
    Direction determined by MACD + EMA alignment.

    Default params: vol_spike=2.5
    """
    vol_ratio = _safe(row.get("volume_ratio"))
    adx = _safe(row.get("adx"))
    macd_sig = _safe_str(row.get("macd_signal"))
    ema_align = _safe(row.get("ema_alignment"))

    if any(v is None for v in [vol_ratio, adx]):
        return None
    if macd_sig is None or ema_align is None:
        return None

    vol_spike_threshold = params.get("vol_spike", 2.5)

    if vol_ratio <= vol_spike_threshold or adx <= 20:
        return None

    # Direction
    if macd_sig in ("bullish", "bullish_cross") and ema_align > 0:
        return _build_signal("LONG", params, default_sl=0.015, default_tp=0.035)
    elif macd_sig in ("bearish", "bearish_cross") and ema_align < 0:
        return _build_signal("SHORT", params, default_sl=0.015, default_tp=0.035)

    return None


# ---------------------------------------------------------------------------
# 9. Stochastic Reversal – StochRSI cross from extreme
# ---------------------------------------------------------------------------

def stoch_reversal(row: pd.Series, params: dict) -> Optional[dict]:
    """StochRSI cross from extreme zones.

    LONG: stoch_rsi_k crosses above stoch_low (prev<stoch_low, curr>stoch_low),
          bb_pct < 0.3.
    SHORT: stoch_rsi_k crosses below stoch_high, bb_pct > 0.7.

    Pass params["prev_stoch_k"] for crossover detection.
    Default params: stoch_low=20, stoch_high=80
    """
    stoch_k = _safe(row.get("stoch_rsi_k"))
    bb_pct = _safe(row.get("bb_pct"))
    prev_stoch_k = _safe(params.get("prev_stoch_k"))

    if any(v is None for v in [stoch_k, bb_pct, prev_stoch_k]):
        return None

    stoch_low = params.get("stoch_low", 20)
    stoch_high = params.get("stoch_high", 80)

    # LONG: stoch crosses above stoch_low from below
    if prev_stoch_k < stoch_low and stoch_k > stoch_low and bb_pct < 0.3:
        return _build_signal("LONG", params, default_sl=0.010, default_tp=0.025)

    # SHORT: stoch crosses below stoch_high from above
    if prev_stoch_k > stoch_high and stoch_k < stoch_high and bb_pct > 0.7:
        return _build_signal("SHORT", params, default_sl=0.010, default_tp=0.025)

    return None


# ---------------------------------------------------------------------------
# 10. Ichimoku Cloud – price crosses above/below cloud
# ---------------------------------------------------------------------------

def ichimoku_cloud(row: pd.Series, params: dict) -> Optional[dict]:
    """Price crosses above/below the Ichimoku cloud.

    LONG: price > senkou_a AND price > senkou_b AND tenkan > kijun.
    SHORT: price < senkou_a AND price < senkou_b AND tenkan < kijun.

    Requires columns: ichimoku_senkou_a, ichimoku_senkou_b, ichimoku_tenkan,
    ichimoku_kijun.
    """
    price = _safe(row.get("close"))
    senkou_a = _safe(row.get("ichimoku_senkou_a"))
    senkou_b = _safe(row.get("ichimoku_senkou_b"))
    tenkan = _safe(row.get("ichimoku_tenkan"))
    kijun = _safe(row.get("ichimoku_kijun"))

    if any(v is None for v in [price, senkou_a, senkou_b, tenkan, kijun]):
        return None

    # LONG
    if price > senkou_a and price > senkou_b and tenkan > kijun:
        return _build_signal("LONG", params, default_sl=0.012, default_tp=0.030)

    # SHORT
    if price < senkou_a and price < senkou_b and tenkan < kijun:
        return _build_signal("SHORT", params, default_sl=0.012, default_tp=0.030)

    return None


# ---------------------------------------------------------------------------
# 11. Triple Confirmation – requires 3+ indicators to agree
# ---------------------------------------------------------------------------

def triple_confirmation(row: pd.Series, params: dict) -> Optional[dict]:
    """Requires multiple indicators to agree before entering.

    Counts bullish signals: RSI<40, MACD bullish, ema_alignment>0,
    stoch_k<30, bb_pct<0.3, volume_ratio>1.0.
    If count >= min_confirms -> LONG.

    Counts bearish signals: RSI>60, MACD bearish, ema_alignment<0,
    stoch_k>70, bb_pct>0.7.
    If count >= min_confirms -> SHORT.

    Default params: min_confirms=4
    """
    rsi = _safe(row.get("rsi_14"))
    macd_sig = _safe_str(row.get("macd_signal"))
    ema_align = _safe(row.get("ema_alignment"))
    stoch_k = _safe(row.get("stoch_rsi_k"))
    bb_pct = _safe(row.get("bb_pct"))
    vol_ratio = _safe(row.get("volume_ratio"))

    # Need at least some of these to be present
    if all(v is None for v in [rsi, macd_sig, ema_align, stoch_k, bb_pct, vol_ratio]):
        return None

    min_confirms = params.get("min_confirms", 4)

    # Count bullish signals
    bull_count = 0
    if rsi is not None and rsi < 40:
        bull_count += 1
    if macd_sig in ("bullish", "bullish_cross"):
        bull_count += 1
    if ema_align is not None and ema_align > 0:
        bull_count += 1
    if stoch_k is not None and stoch_k < 30:
        bull_count += 1
    if bb_pct is not None and bb_pct < 0.3:
        bull_count += 1
    if vol_ratio is not None and vol_ratio > 1.0:
        bull_count += 1

    if bull_count >= min_confirms:
        return _build_signal("LONG", params, default_sl=0.010, default_tp=0.025)

    # Count bearish signals
    bear_count = 0
    if rsi is not None and rsi > 60:
        bear_count += 1
    if macd_sig in ("bearish", "bearish_cross"):
        bear_count += 1
    if ema_align is not None and ema_align < 0:
        bear_count += 1
    if stoch_k is not None and stoch_k > 70:
        bear_count += 1
    if bb_pct is not None and bb_pct > 0.7:
        bear_count += 1
    # (volume_ratio > 1.0 is direction-neutral, counted for both)
    if vol_ratio is not None and vol_ratio > 1.0:
        bear_count += 1

    if bear_count >= min_confirms:
        return _build_signal("SHORT", params, default_sl=0.010, default_tp=0.025)

    return None


# ---------------------------------------------------------------------------
# 12. Williams %R + CCI Combo
# ---------------------------------------------------------------------------

def williams_cci_combo(row: pd.Series, params: dict) -> Optional[dict]:
    """Williams %R + CCI combo for extreme conditions.

    LONG: williams_r < wr_oversold AND cci < cci_oversold AND volume_ratio>0.8.
    SHORT: williams_r > wr_overbought AND cci > cci_overbought.

    Requires columns: williams_r, cci.
    Default params: wr_oversold=-80, wr_overbought=-20,
                    cci_oversold=-100, cci_overbought=100
    """
    williams_r = _safe(row.get("williams_r"))
    cci = _safe(row.get("cci"))
    vol_ratio = _safe(row.get("volume_ratio"))

    if any(v is None for v in [williams_r, cci]):
        return None

    wr_oversold = params.get("wr_oversold", -80)
    wr_overbought = params.get("wr_overbought", -20)
    cci_oversold = params.get("cci_oversold", -100)
    cci_overbought = params.get("cci_overbought", 100)

    # LONG
    if williams_r < wr_oversold and cci < cci_oversold:
        if vol_ratio is None or vol_ratio > 0.8:
            return _build_signal("LONG", params, default_sl=0.010, default_tp=0.025)

    # SHORT
    if williams_r > wr_overbought and cci > cci_overbought:
        return _build_signal("SHORT", params, default_sl=0.010, default_tp=0.025)

    return None


# ---------------------------------------------------------------------------
# Registry – maps strategy name to (function, default_params)
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, tuple] = {
    "trend_follow": (trend_follow, {
        "adx_min": 30, "ema_align_min": 0.5, "rsi_low": 35, "rsi_high": 55,
        "sl_pct": 0.012, "tp_pct": 0.030, "trailing_pct": 0.015,
    }),
    "breakout": (breakout, {
        "vol_min": 1.5,
        "sl_pct": 0.012, "tp_pct": 0.035, "trailing_pct": 0.015,
    }),
    "mean_reversion": (mean_reversion, {
        "rsi_oversold": 22, "rsi_overbought": 78,
        "bb_low": 0.05, "bb_high": 0.95,
        "stoch_low": 10, "stoch_high": 90,
        "adx_max": 25,
        "sl_pct": 0.010, "tp_pct": 0.020, "trailing_pct": 0.015,
    }),
    "momentum": (momentum, {
        "adx_min": 25,
        "sl_pct": 0.012, "tp_pct": 0.030, "trailing_pct": 0.015,
    }),
    "macd_cross": (macd_cross, {
        "vol_min": 0.8,
        "sl_pct": 0.012, "tp_pct": 0.025, "trailing_pct": 0.015,
    }),
    "ema_cross": (ema_cross, {
        "sl_pct": 0.012, "tp_pct": 0.030, "trailing_pct": 0.015,
    }),
    "bollinger_bounce": (bollinger_bounce, {
        "bb_entry_low": 0.1, "bb_entry_high": 0.9, "adx_max": 30,
        "rsi_max": 40, "rsi_min": 60,
        "sl_pct": 0.010, "tp_pct": 0.020, "trailing_pct": 0.015,
    }),
    "volume_spike": (volume_spike, {
        "vol_spike": 2.5,
        "sl_pct": 0.015, "tp_pct": 0.035, "trailing_pct": 0.015,
    }),
    "stoch_reversal": (stoch_reversal, {
        "stoch_low": 20, "stoch_high": 80,
        "sl_pct": 0.010, "tp_pct": 0.025, "trailing_pct": 0.015,
    }),
    "ichimoku_cloud": (ichimoku_cloud, {
        "sl_pct": 0.012, "tp_pct": 0.030, "trailing_pct": 0.015,
    }),
    "triple_confirmation": (triple_confirmation, {
        "min_confirms": 4,
        "sl_pct": 0.010, "tp_pct": 0.025, "trailing_pct": 0.015,
    }),
    "williams_cci_combo": (williams_cci_combo, {
        "wr_oversold": -80, "wr_overbought": -20,
        "cci_oversold": -100, "cci_overbought": 100,
        "sl_pct": 0.010, "tp_pct": 0.025, "trailing_pct": 0.015,
    }),
}


def get_strategy(name: str):
    """Return (strategy_func, default_params) by name, or raise KeyError."""
    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name]


def list_strategies() -> list[str]:
    """Return all registered strategy names."""
    return list(STRATEGY_REGISTRY.keys())
