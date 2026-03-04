"""Session filter: restrict signals to specific trading sessions.

Filters signals by the UTC hour of the bar's timestamp, allowing analysis
of which trading session produces the best results.

Session definitions (UTC hours):
    asian_only:   00-08
    london_only:  08-16
    ny_only:      16-24
    no_asian:     08-24
    london_ny:    08-24 (same as no_asian)
    all_sessions: 00-24 (baseline, no filter)
"""

import pandas as pd
from backtest.mega.strategies_mega import _safe


# ---------------------------------------------------------------------------
# Session definitions: name -> (start_hour, end_hour) in UTC
# ---------------------------------------------------------------------------

SESSION_CONFIGS = {
    "asian_only":   (0, 8),
    "london_only":  (8, 16),
    "ny_only":      (16, 24),
    "no_asian":     (8, 24),
    "london_ny":    (8, 24),
    "all_sessions": (0, 24),
}


def filter_signals_by_session(signals: list[dict], df: pd.DataFrame,
                              session: str) -> list[dict]:
    """Filter signals to only keep those within the given session hours.

    Args:
        signals: list of signal dicts with "bar" index
        df: DataFrame with "timestamp" column
        session: one of SESSION_CONFIGS keys

    Returns:
        Filtered signal list (new list, originals unchanged)
    """
    if session == "all_sessions":
        return signals  # no filtering

    start_h, end_h = SESSION_CONFIGS[session]
    has_ts = "timestamp" in df.columns

    if not has_ts:
        return signals  # can't filter without timestamps

    out = []
    for s in signals:
        bar = s["bar"]
        if bar < 0 or bar >= len(df):
            continue
        ts = df.iloc[bar].get("timestamp")
        if ts is None:
            continue
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        hour = ts.hour
        if start_h <= hour < end_h:
            out.append(s)
    return out


def run_session_analysis(signals: list[dict], df: pd.DataFrame,
                         simulate_fn, params: dict, pair: str,
                         strategy: str) -> list[dict]:
    """Run simulation for each session config and return results.

    Args:
        signals: base signal list for one strategy+pair
        df: DataFrame for simulation
        simulate_fn: simulator.simulate function
        params: best params dict for this strategy
        pair: symbol string
        strategy: strategy name

    Returns:
        List of result dicts, one per session config
    """
    results = []
    for session_name in SESSION_CONFIGS:
        filtered = filter_signals_by_session(signals, df, session_name)
        if not filtered:
            continue

        result = simulate_fn(df, filtered, params, pair)
        if result and result["total_trades"] >= 3:
            result["pair"] = pair
            result["strategy"] = strategy
            result["session"] = session_name
            result["params"] = {k: v for k, v in params.items()}
            result["n_signals_filtered"] = len(filtered)
            result["n_signals_total"] = len(signals)
            results.append(result)
    return results
