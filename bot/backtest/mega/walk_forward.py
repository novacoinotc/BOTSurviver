"""Walk-forward validation: detect overfitting via train/test splits.

Three split methods:
    A (50/50):     Year 1 train -> Year 2 test (1 window)
    B (Rolling):   12-month train -> 3-month test, sliding (~5 windows)
    C (Quarterly): All prior data train -> next quarter test (~6 windows)

For each split: filter signals by bar range, simulate train and test
separately, compute degradation metric.

Degradation = (train_PF - test_PF) / train_PF
    < 20%  -> ROBUST
    20-40% -> MARGINAL
    > 40%  -> OVERFIT

Optimisation: uses pre-computed indicators from full DF, filters signals
by bar range (no recomputing indicators per split — minimal bias at
window edges, 20x faster).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Split generators
# ---------------------------------------------------------------------------

def _get_total_bars(df: pd.DataFrame) -> int:
    return len(df)


def splits_5050(df: pd.DataFrame) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Method A: simple 50/50 split."""
    n = _get_total_bars(df)
    mid = n // 2
    return [((0, mid), (mid, n))]


def splits_rolling(df: pd.DataFrame, train_bars: int = 720,
                   test_bars: int = 180, step_bars: int = 180) -> list[tuple]:
    """Method B: rolling 12-month train -> 3-month test.

    720 bars ≈ 30 days/month * 24h = actually 720 1H bars = 30 days.
    For 12 months: 365*24=8760, but we use approximate 720*12=8640 ≈ 12mo.
    Simpler: train_bars=8760//2=4380 is too much. Use months:
    12 months = ~8760 bars, 3 months = ~2190 bars.
    But typical 2-year data = ~17520 bars. Let's use sensible defaults.
    """
    n = _get_total_bars(df)
    windows = []
    start = 0
    while start + train_bars + test_bars <= n:
        train_range = (start, start + train_bars)
        test_range = (start + train_bars, start + train_bars + test_bars)
        windows.append((train_range, test_range))
        start += step_bars
    return windows


def splits_quarterly(df: pd.DataFrame) -> list[tuple]:
    """Method C: expanding window, test on next quarter.

    Quarter ≈ 90 days * 24 = 2160 bars (1H).
    Min train = 2 quarters before first test.
    """
    n = _get_total_bars(df)
    quarter = 2160  # ~90 days of 1H bars
    min_train = 2 * quarter  # at least 6 months for training

    windows = []
    test_start = min_train
    while test_start + quarter <= n:
        train_range = (0, test_start)
        test_range = (test_start, min(test_start + quarter, n))
        windows.append((train_range, test_range))
        test_start += quarter
    return windows


SPLIT_METHODS = {
    "5050": splits_5050,
    "rolling": lambda df: splits_rolling(df, train_bars=8760, test_bars=2190, step_bars=2190),
    "quarterly": splits_quarterly,
}


# ---------------------------------------------------------------------------
# Signal filtering by bar range
# ---------------------------------------------------------------------------

def _filter_signals_by_range(signals: list[dict],
                             bar_start: int, bar_end: int) -> list[dict]:
    """Keep only signals with bar in [bar_start, bar_end)."""
    return [s for s in signals if bar_start <= s["bar"] < bar_end]


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward_validate(df: pd.DataFrame, signals: list[dict],
                          params: dict, pair: str, strategy: str,
                          simulate_fn, method: str = "all") -> list[dict]:
    """Run walk-forward validation on a single strategy+pair+params.

    Args:
        df: full DataFrame with indicators
        signals: full signal list for this strategy+pair
        params: parameter dict
        pair: symbol
        strategy: strategy name
        simulate_fn: simulator.simulate function
        method: "5050", "rolling", "quarterly", or "all"

    Returns:
        List of validation result dicts, one per split window
    """
    methods_to_run = list(SPLIT_METHODS.keys()) if method == "all" else [method]
    results = []

    for method_name in methods_to_run:
        split_fn = SPLIT_METHODS[method_name]
        windows = split_fn(df)

        for w_idx, (train_range, test_range) in enumerate(windows):
            train_sigs = _filter_signals_by_range(signals, *train_range)
            test_sigs = _filter_signals_by_range(signals, *test_range)

            # Need minimum signals to be meaningful
            if len(train_sigs) < 3 or len(test_sigs) < 2:
                continue

            train_result = simulate_fn(df, train_sigs, params, pair)
            test_result = simulate_fn(df, test_sigs, params, pair)

            if not train_result or not test_result:
                continue
            if train_result["total_trades"] < 3 or test_result["total_trades"] < 2:
                continue

            train_pf = train_result["profit_factor"]
            test_pf = test_result["profit_factor"]

            # Degradation: how much worse is test vs train
            if train_pf > 0:
                degradation = (train_pf - test_pf) / train_pf
            else:
                degradation = 1.0  # train already bad

            if degradation < 0.20:
                verdict = "ROBUST"
            elif degradation < 0.40:
                verdict = "MARGINAL"
            else:
                verdict = "OVERFIT"

            results.append({
                "pair": pair,
                "strategy": strategy,
                "params": {k: v for k, v in params.items()},
                "method": method_name,
                "window": w_idx,
                "train_bars": train_range[1] - train_range[0],
                "test_bars": test_range[1] - test_range[0],
                "train_trades": train_result["total_trades"],
                "test_trades": test_result["total_trades"],
                "train_pf": round(train_pf, 4),
                "test_pf": round(test_pf, 4),
                "train_pnl": round(train_result["total_pnl"], 2),
                "test_pnl": round(test_result["total_pnl"], 2),
                "train_wr": round(train_result["win_rate"] * 100, 1),
                "test_wr": round(test_result["win_rate"] * 100, 1),
                "degradation": round(degradation, 4),
                "verdict": verdict,
            })

    return results


def aggregate_walk_forward(wf_results: list[dict]) -> list[dict]:
    """Aggregate walk-forward results per strategy+params config.

    Groups by (strategy, params_key), computes average degradation and
    overall verdict across all windows and methods.

    Returns sorted list (best = lowest avg degradation).
    """
    from collections import defaultdict
    import json

    config_map = defaultdict(list)
    for r in wf_results:
        key = (r["strategy"], json.dumps(r["params"], sort_keys=True))
        config_map[key].append(r)

    summaries = []
    for (strat, params_key), entries in config_map.items():
        degradations = [e["degradation"] for e in entries]
        avg_deg = float(np.mean(degradations))
        max_deg = float(np.max(degradations))
        n_robust = sum(1 for e in entries if e["verdict"] == "ROBUST")
        n_marginal = sum(1 for e in entries if e["verdict"] == "MARGINAL")
        n_overfit = sum(1 for e in entries if e["verdict"] == "OVERFIT")

        if avg_deg < 0.20:
            overall = "ROBUST"
        elif avg_deg < 0.40:
            overall = "MARGINAL"
        else:
            overall = "OVERFIT"

        # Aggregate PnL
        avg_train_pnl = float(np.mean([e["train_pnl"] for e in entries]))
        avg_test_pnl = float(np.mean([e["test_pnl"] for e in entries]))
        avg_train_pf = float(np.mean([e["train_pf"] for e in entries]))
        avg_test_pf = float(np.mean([e["test_pf"] for e in entries]))

        pairs_tested = sorted(set(e["pair"] for e in entries))

        summaries.append({
            "strategy": strat,
            "params": json.loads(params_key),
            "n_windows": len(entries),
            "n_pairs": len(pairs_tested),
            "pairs": pairs_tested,
            "avg_degradation": round(avg_deg, 4),
            "max_degradation": round(max_deg, 4),
            "n_robust": n_robust,
            "n_marginal": n_marginal,
            "n_overfit": n_overfit,
            "overall_verdict": overall,
            "avg_train_pf": round(avg_train_pf, 4),
            "avg_test_pf": round(avg_test_pf, 4),
            "avg_train_pnl": round(avg_train_pnl, 2),
            "avg_test_pnl": round(avg_test_pnl, 2),
        })

    summaries.sort(key=lambda x: x["avg_degradation"])
    return summaries
