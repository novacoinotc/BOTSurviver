"""18 combo strategies: base detectors + confirmation filters.

Each combo wraps a base strategy's signals and applies an additional
indicator filter to reduce false entries.  Re-uses signal lists from
strategies_mega.detect_all_signals() — no duplicate detection loops.

Signal format is identical to base strategies:
    {"bar": int, "dir": "LONG"|"SHORT", "strat": str, "score": float}
"""

import numpy as np
import pandas as pd

from backtest.mega.strategies_mega import _safe


# ---------------------------------------------------------------------------
# Generic filter helpers
# ---------------------------------------------------------------------------

def _filter_ema_aligned(signals: list[dict], df: pd.DataFrame,
                        threshold: float = 0.3) -> list[dict]:
    """Keep signals where EMA alignment confirms direction."""
    out = []
    for s in signals:
        ea = _safe(df.iloc[s["bar"]].get("ema_alignment"))
        if ea is None:
            continue
        if s["dir"] == "LONG" and ea > threshold:
            out.append({**s, "strat": s["strat"] + "_ema_aligned"})
        elif s["dir"] == "SHORT" and ea < -threshold:
            out.append({**s, "strat": s["strat"] + "_ema_aligned"})
    return out


def _filter_adx_trending(signals: list[dict], df: pd.DataFrame,
                         threshold: float = 25) -> list[dict]:
    """Keep signals where ADX > threshold (trending market)."""
    out = []
    for s in signals:
        adx = _safe(df.iloc[s["bar"]].get("adx"))
        if adx is not None and adx > threshold:
            out.append({**s, "strat": s["strat"] + "_adx_trending"})
    return out


def _filter_supertrend(signals: list[dict], df: pd.DataFrame) -> list[dict]:
    """Keep signals where supertrend confirms direction."""
    out = []
    for s in signals:
        st_dir = _safe(df.iloc[s["bar"]].get("supertrend_dir"))
        if st_dir is None:
            continue
        if (s["dir"] == "LONG" and st_dir > 0) or \
           (s["dir"] == "SHORT" and st_dir < 0):
            out.append({**s, "strat": s["strat"] + "_supertrend"})
    return out


def _filter_rsi_neutral(signals: list[dict], df: pd.DataFrame,
                        lo: float = 30, hi: float = 70) -> list[dict]:
    """Keep signals where RSI is in neutral zone (not overbought/oversold)."""
    out = []
    for s in signals:
        rsi = _safe(df.iloc[s["bar"]].get("rsi_14"))
        if rsi is not None and lo <= rsi <= hi:
            out.append({**s, "strat": s["strat"] + "_rsi_neutral"})
    return out


def _filter_macd_confirms(signals: list[dict], df: pd.DataFrame) -> list[dict]:
    """Keep signals where MACD histogram confirms direction."""
    out = []
    for s in signals:
        row = df.iloc[s["bar"]]
        macd_sig = row.get("macd_signal")
        if macd_sig is None:
            continue
        if s["dir"] == "LONG" and macd_sig in ("bullish", "bullish_cross"):
            out.append({**s, "strat": s["strat"] + "_macd_confirms"})
        elif s["dir"] == "SHORT" and macd_sig in ("bearish", "bearish_cross"):
            out.append({**s, "strat": s["strat"] + "_macd_confirms"})
    return out


def _filter_volume_above(signals: list[dict], df: pd.DataFrame,
                         threshold: float = 1.5) -> list[dict]:
    """Keep signals where volume_ratio > threshold."""
    out = []
    suffix = f"_vol{threshold:.0f}" if threshold == int(threshold) else f"_vol{threshold}"
    for s in signals:
        vr = _safe(df.iloc[s["bar"]].get("volume_ratio")) or 0
        if vr > threshold:
            out.append({**s, "strat": s["strat"] + suffix})
    return out


def _filter_ichimoku(signals: list[dict], df: pd.DataFrame) -> list[dict]:
    """Keep signals where price is on correct side of ichimoku cloud."""
    out = []
    for s in signals:
        row = df.iloc[s["bar"]]
        close = _safe(row.get("close"))
        sa = _safe(row.get("ichimoku_senkou_a"))
        sb = _safe(row.get("ichimoku_senkou_b"))
        if any(v is None for v in [close, sa, sb]):
            continue
        cloud_top = max(sa, sb)
        cloud_bot = min(sa, sb)
        if s["dir"] == "LONG" and close > cloud_top:
            out.append({**s, "strat": s["strat"] + "_ichimoku"})
        elif s["dir"] == "SHORT" and close < cloud_bot:
            out.append({**s, "strat": s["strat"] + "_ichimoku"})
    return out


def _filter_all(signals: list[dict], df: pd.DataFrame) -> list[dict]:
    """Apply ALL filters (most strict). For volume_spike combos."""
    filtered = signals
    # Apply each filter in sequence, resetting strat name at each step
    # to avoid compounding suffixes
    result = []
    for s in filtered:
        bar = s["bar"]
        row = df.iloc[bar]

        ea = _safe(row.get("ema_alignment"))
        adx = _safe(row.get("adx"))
        st_dir = _safe(row.get("supertrend_dir"))
        rsi = _safe(row.get("rsi_14"))
        macd_sig = row.get("macd_signal")

        if any(v is None for v in [ea, adx, st_dir, rsi, macd_sig]):
            continue

        if s["dir"] == "LONG":
            if not (ea > 0.3 and adx > 25 and st_dir > 0
                    and 30 <= rsi <= 70
                    and macd_sig in ("bullish", "bullish_cross")):
                continue
        else:
            if not (ea < -0.3 and adx > 25 and st_dir < 0
                    and 30 <= rsi <= 70
                    and macd_sig in ("bearish", "bearish_cross")):
                continue

        result.append({**s, "strat": s["strat"] + "_all_filters"})
    return result


# ---------------------------------------------------------------------------
# Combo definitions
# ---------------------------------------------------------------------------

# Map: combo_name -> (base_strategy, filter_function)
COMBO_DEFS: dict[str, tuple[str, callable]] = {
    # volume_spike (6 combos)
    "vs_ema_aligned":    ("volume_spike", _filter_ema_aligned),
    "vs_adx_trending":   ("volume_spike", _filter_adx_trending),
    "vs_supertrend":     ("volume_spike", _filter_supertrend),
    "vs_rsi_neutral":    ("volume_spike", _filter_rsi_neutral),
    "vs_macd_confirms":  ("volume_spike", _filter_macd_confirms),
    "vs_all_filters":    ("volume_spike", _filter_all),

    # mtf_confirmation (3 combos)
    "mtf_adx_trending":  ("mtf_confirmation", _filter_adx_trending),
    "mtf_rsi_neutral":   ("mtf_confirmation", _filter_rsi_neutral),
    "mtf_supertrend":    ("mtf_confirmation", _filter_supertrend),

    # macd_cross_1h (3 combos)
    "macd_adx_trending": ("macd_cross_1h", _filter_adx_trending),
    "macd_rsi_neutral":  ("macd_cross_1h", _filter_rsi_neutral),
    "macd_supertrend":   ("macd_cross_1h", _filter_supertrend),

    # regime_adaptive (3 combos)
    "regime_vol15":      ("regime_adaptive", lambda s, df: _filter_volume_above(s, df, 1.5)),
    "regime_supertrend": ("regime_adaptive", _filter_supertrend),
    "regime_rsi_neutral":("regime_adaptive", _filter_rsi_neutral),

    # trend_follow_1h (3 combos)
    "tf_supertrend":     ("trend_follow_1h", _filter_supertrend),
    "tf_vol20":          ("trend_follow_1h", lambda s, df: _filter_volume_above(s, df, 2.0)),
    "tf_ichimoku":       ("trend_follow_1h", _filter_ichimoku),
}


def detect_combo_signals(base_signals: dict[str, list[dict]],
                         df: pd.DataFrame) -> dict[str, list[dict]]:
    """Generate all 18 combo signal lists from pre-computed base signals.

    Args:
        base_signals: {strategy_name: [signal_dicts]} from detect_all_signals()
        df: DataFrame with all indicators

    Returns:
        {combo_name: [signal_dicts]} — 18 entries
    """
    combos = {}
    for combo_name, (base_strat, filter_fn) in COMBO_DEFS.items():
        sigs = base_signals.get(base_strat, [])
        if not sigs:
            combos[combo_name] = []
            continue
        # Reset strat name to base before filtering (filter will set combo suffix)
        # We override strat to the combo_name directly for clarity
        filtered = filter_fn(sigs, df)
        # Normalize strat name to combo_name
        for s in filtered:
            s["strat"] = combo_name
        combos[combo_name] = filtered
    return combos


# Number of combos for verification
N_COMBOS = len(COMBO_DEFS)
assert N_COMBOS == 18, f"Expected 18 combos, got {N_COMBOS}"
