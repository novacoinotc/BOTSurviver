"""Grid-search parameter optimizer for backtesting strategies.

Generates all parameter combinations per strategy, runs backtests in parallel
using ProcessPoolExecutor, and returns results sorted by profit_factor.

Usage::

    from backtest.optimizer import optimize, optimize_all
    from backtest.data_loader import download_klines

    df = download_klines("BTCUSDT", "1m", days=14)
    # ... add indicator columns to df ...

    results = optimize(df, "BTCUSDT", "trend_follow", n_workers=8)
    # results is a list of dicts sorted by profit_factor descending

    # Or run everything:
    all_results = optimize_all({"BTCUSDT": df_btc, "ETHUSDT": df_eth})
"""

from __future__ import annotations

import math
import os
import pickle
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd

from backtest.backtester import Backtester
from backtest.strategies import STRATEGY_REGISTRY, get_strategy


# ---------------------------------------------------------------------------
# Common parameter grids
# ---------------------------------------------------------------------------

SL_VALUES = [0.008, 0.010, 0.012, 0.015, 0.020, 0.025]
TP_VALUES = [0.020, 0.025, 0.030, 0.040, 0.050, 0.060]
TRAILING_VALUES = [0.010, 0.012, 0.015, 0.020, 0.025]
LEVERAGE_VALUES = [2, 3]
POSITION_PCT_VALUES = [0.005, 0.008]
COOLDOWN_VALUES = [10, 15, 20, 30]


def _cart_product(grid: dict[str, list]) -> list[dict]:
    """Cartesian product of parameter lists into list of dicts."""
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*vals)]


# ---------------------------------------------------------------------------
# Per-strategy parameter grids
# ---------------------------------------------------------------------------

def generate_param_grid(strategy_name: str) -> list[dict]:
    """Generate all parameter combinations for the given strategy.

    Returns a list of param dicts ready to pass to the backtester.
    """
    if strategy_name == "trend_follow":
        grid = {
            "adx_min": [25, 30, 35],
            "ema_align_min": [0.3, 0.5, 1.0],
            "rsi_low": [30, 35, 40],
            "rsi_high": [50, 55, 60],
            "sl_pct": [0.010, 0.012, 0.015, 0.020],
            "tp_pct": [0.025, 0.030, 0.040, 0.050],
            "trailing_pct": [0.012, 0.015, 0.020],
            "cooldown_bars": [10, 15, 20],
        }

    elif strategy_name == "mean_reversion":
        grid = {
            "rsi_oversold": [20, 25, 30],
            "rsi_overbought": [70, 75, 80],
            "bb_low": [0.05, 0.10],
            "bb_high": [0.90, 0.95],
            "stoch_low": [10, 15],
            "stoch_high": [85, 90],
            "adx_max": [20, 25],
            "sl_pct": [0.010, 0.015, 0.020],
            "tp_pct": [0.020, 0.030, 0.050],
            "trailing_pct": [0.012, 0.015],
        }

    elif strategy_name == "breakout":
        grid = {
            "vol_min": [1.2, 1.5, 2.0, 2.5],
            "sl_pct": [0.010, 0.012, 0.015, 0.020],
            "tp_pct": [0.025, 0.030, 0.040, 0.050],
            "trailing_pct": [0.012, 0.015, 0.020],
            "cooldown_bars": [10, 15, 20, 30],
        }

    elif strategy_name == "momentum":
        grid = {
            "adx_min": [20, 25, 30],
            "sl_pct": [0.010, 0.012, 0.015, 0.020],
            "tp_pct": [0.025, 0.030, 0.040, 0.050],
            "trailing_pct": [0.012, 0.015, 0.020],
            "cooldown_bars": [10, 15, 20, 30],
        }

    elif strategy_name == "macd_cross":
        grid = {
            "vol_min": [0.5, 0.8, 1.0, 1.5],
            "sl_pct": [0.010, 0.012, 0.015, 0.020],
            "tp_pct": [0.025, 0.030, 0.040, 0.050],
            "trailing_pct": [0.012, 0.015, 0.020],
        }

    elif strategy_name == "ema_cross":
        grid = {
            "sl_pct": [0.010, 0.012, 0.015, 0.020],
            "tp_pct": [0.025, 0.030, 0.040, 0.050],
            "trailing_pct": [0.012, 0.015, 0.020],
            "cooldown_bars": [10, 15, 20, 30],
        }

    elif strategy_name == "bollinger_bounce":
        grid = {
            "bb_entry_low": [0.05, 0.10, 0.15],
            "bb_entry_high": [0.85, 0.90, 0.95],
            "adx_max": [20, 25, 30],
            "rsi_max": [35, 40],
            "rsi_min": [60, 65],
            "sl_pct": [0.010, 0.015, 0.020],
            "tp_pct": [0.020, 0.030, 0.050],
            "trailing_pct": [0.012, 0.015],
        }

    elif strategy_name == "volume_spike":
        grid = {
            "vol_spike": [2.0, 2.5, 3.0, 4.0],
            "sl_pct": [0.010, 0.012, 0.015, 0.020],
            "tp_pct": [0.025, 0.030, 0.040, 0.050],
            "trailing_pct": [0.012, 0.015, 0.020],
        }

    elif strategy_name == "stoch_reversal":
        grid = {
            "stoch_low": [15, 20, 25],
            "stoch_high": [75, 80, 85],
            "sl_pct": [0.008, 0.010, 0.012, 0.015],
            "tp_pct": [0.020, 0.025, 0.030, 0.040],
            "trailing_pct": [0.010, 0.012, 0.015],
        }

    elif strategy_name == "ichimoku_cloud":
        grid = {
            "sl_pct": [0.010, 0.012, 0.015, 0.020],
            "tp_pct": [0.025, 0.030, 0.040, 0.050],
            "trailing_pct": [0.012, 0.015, 0.020],
            "cooldown_bars": [10, 15, 20, 30],
        }

    elif strategy_name == "triple_confirmation":
        grid = {
            "min_confirms": [3, 4, 5],
            "sl_pct": [0.008, 0.010, 0.012, 0.015],
            "tp_pct": [0.020, 0.025, 0.030, 0.040],
            "trailing_pct": [0.010, 0.012, 0.015],
        }

    elif strategy_name == "williams_cci_combo":
        grid = {
            "wr_oversold": [-85, -80, -75],
            "wr_overbought": [-25, -20, -15],
            "cci_oversold": [-150, -100, -50],
            "cci_overbought": [50, 100, 150],
            "sl_pct": [0.008, 0.010, 0.012, 0.015],
            "tp_pct": [0.020, 0.025, 0.030, 0.040],
            "trailing_pct": [0.010, 0.012, 0.015],
        }

    else:
        raise KeyError(f"No parameter grid defined for strategy: {strategy_name}")

    combos = _cart_product(grid)

    # Tag each combo with strategy name (used by backtester result)
    for c in combos:
        c["strategy_name"] = strategy_name

    return combos


# ---------------------------------------------------------------------------
# Strategy wrapper that injects previous-bar values into params
# ---------------------------------------------------------------------------

# Strategies that need previous-bar values and the columns they need
_PREV_BAR_COLUMNS = {
    "breakout": {"prev_bb_squeeze": "bb_squeeze"},
    "momentum": {"prev_rsi_14": "rsi_14"},
    "ema_cross": {"prev_ema_9": "ema_9", "prev_ema_21": "ema_21"},
    "stoch_reversal": {"prev_stoch_k": "stoch_rsi_k"},
}


def _make_strategy_wrapper(strategy_name: str):
    """Return a wrapper function that handles previous-bar injection.

    The wrapper is called with (row, params) by the Backtester.  For strategies
    that need previous-bar values, the wrapper stores them between calls via a
    mutable closure dict.
    """
    strategy_func, _ = get_strategy(strategy_name)
    prev_cols = _PREV_BAR_COLUMNS.get(strategy_name)

    if prev_cols is None:
        # No previous-bar values needed -- use the raw function
        return strategy_func

    # Closure state for previous bar values
    state = {"prev": {}}

    def wrapper(row: pd.Series, params: dict) -> Optional[dict]:
        # Inject previous bar values into params
        merged = dict(params)
        merged.update(state["prev"])

        # Call underlying strategy
        signal = strategy_func(row, merged)

        # Store current bar values for next call
        state["prev"] = {}
        for param_key, col_name in prev_cols.items():
            val = row.get(col_name)
            state["prev"][param_key] = val

        return signal

    return wrapper


# ---------------------------------------------------------------------------
# Single-backtest worker (for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def run_single_backtest(args: tuple) -> dict:
    """Worker function for parallel execution.

    args = (df_pickle_path, pair, strategy_name, params, initial_equity, fee_pct)

    Returns a result dict or None if the backtest produced no trades.
    """
    df_pickle_path, pair, strategy_name, params, initial_equity, fee_pct = args

    # Load data from temp file (avoids pickling large DataFrames per worker)
    with open(df_pickle_path, "rb") as f:
        df = pickle.load(f)

    # Build strategy wrapper (handles prev-bar injection)
    wrapper = _make_strategy_wrapper(strategy_name)

    # Run backtest
    bt = Backtester(df, pair, fee_pct=fee_pct, initial_equity=initial_equity)
    result = bt.run(wrapper, params)

    return {
        "strategy": strategy_name,
        "params": {k: v for k, v in params.items() if k != "strategy_name"},
        "pair": pair,
        "total_trades": result.total_trades,
        "wins": result.wins,
        "win_rate": round(result.win_rate, 4),
        "total_pnl": round(result.total_pnl, 4),
        "profit_factor": round(result.profit_factor, 4),
        "max_drawdown_pct": round(result.max_drawdown_pct, 6),
        "sharpe_ratio": round(result.sharpe_ratio, 4),
        "avg_win_pct": round(result.avg_win_pct, 6),
        "avg_loss_pct": round(result.avg_loss_pct, 6),
    }


# ---------------------------------------------------------------------------
# optimize() – single strategy on a single pair
# ---------------------------------------------------------------------------

def optimize(
    df: pd.DataFrame,
    pair: str,
    strategy_name: str,
    n_workers: int = None,
    initial_equity: float = 5000.0,
    fee_pct: float = 0.0005,
    top_n: int = 20,
) -> list[dict]:
    """Run grid search for a single strategy on a single pair.

    Parameters
    ----------
    df : pd.DataFrame
        Historical OHLCV data with indicator columns pre-computed.
    pair : str
        Trading pair symbol (e.g. "BTCUSDT").
    strategy_name : str
        Name of the strategy to optimize (must be in STRATEGY_REGISTRY).
    n_workers : int, optional
        Number of parallel workers.  Defaults to ``os.cpu_count()``.
    initial_equity : float
        Starting equity for each backtest run.
    fee_pct : float
        One-way trading fee as a fraction.
    top_n : int
        Number of top results to return.

    Returns
    -------
    list[dict]
        Top results sorted by profit_factor descending.
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 4

    param_combos = generate_param_grid(strategy_name)
    total = len(param_combos)
    print(f"\n[Optimizer] {strategy_name} on {pair}: {total} parameter combinations, {n_workers} workers")

    # Serialize DataFrame to temp file once
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    try:
        pickle.dump(df, tmp)
        tmp.close()
        tmp_path = tmp.name

        # Build task list
        tasks = [
            (tmp_path, pair, strategy_name, params, initial_equity, fee_pct)
            for params in param_combos
        ]

        results: list[dict] = []
        completed = 0
        t0 = time.time()
        last_report = 0.0

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(run_single_backtest, task): task for task in tasks}

            for future in as_completed(futures):
                completed += 1
                try:
                    result = future.result()
                    if result is not None and result["total_trades"] > 0:
                        results.append(result)
                except Exception as e:
                    print(f"  [Error] Worker failed: {e}")

                # Progress report every 10% or every 5 seconds
                elapsed = time.time() - t0
                pct = completed / total * 100
                if pct - last_report >= 10 or elapsed - last_report >= 5:
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    print(f"  [{strategy_name}] {completed}/{total} ({pct:.0f}%) - "
                          f"{rate:.0f} bt/s - ETA {eta:.0f}s")
                    last_report = pct

        elapsed_total = time.time() - t0
        print(f"  [{strategy_name}] Done: {completed} runs in {elapsed_total:.1f}s, "
              f"{len(results)} with trades")

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Sort by profit_factor descending, then by total_pnl
    results.sort(key=lambda r: (r["profit_factor"], r["total_pnl"]), reverse=True)

    return results[:top_n]


# ---------------------------------------------------------------------------
# optimize_all() – all strategies across all pairs
# ---------------------------------------------------------------------------

def optimize_all(
    data: dict[str, pd.DataFrame],
    n_workers: int = None,
    initial_equity: float = 5000.0,
    fee_pct: float = 0.0005,
    top_n: int = 20,
    strategies: list[str] = None,
) -> dict[str, list[dict]]:
    """Run optimization across all strategies and all pairs.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Mapping of pair -> DataFrame with OHLCV + indicator columns.
    n_workers : int, optional
        Number of parallel workers.
    initial_equity : float
        Starting equity.
    fee_pct : float
        One-way fee fraction.
    top_n : int
        Top results to keep per strategy.
    strategies : list[str], optional
        Subset of strategies to optimize. Defaults to all registered.

    Returns
    -------
    dict[str, list[dict]]
        ``{strategy_name: [top_results]}``
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 4

    if strategies is None:
        strategies = list(STRATEGY_REGISTRY.keys())

    all_results: dict[str, list[dict]] = {}
    total_strategies = len(strategies)
    total_pairs = len(data)

    print(f"\n{'='*70}")
    print(f"[Optimizer] Starting full optimization")
    print(f"  Strategies: {total_strategies}")
    print(f"  Pairs: {total_pairs} ({', '.join(data.keys())})")
    print(f"  Workers: {n_workers}")
    print(f"{'='*70}")

    t0_global = time.time()

    for s_idx, strategy_name in enumerate(strategies, 1):
        print(f"\n--- Strategy {s_idx}/{total_strategies}: {strategy_name} ---")

        strategy_results: list[dict] = []

        for pair, df in data.items():
            if df.empty or len(df) < 50:
                print(f"  Skipping {pair}: insufficient data ({len(df)} bars)")
                continue

            pair_results = optimize(
                df=df,
                pair=pair,
                strategy_name=strategy_name,
                n_workers=n_workers,
                initial_equity=initial_equity,
                fee_pct=fee_pct,
                top_n=top_n * 2,  # keep more per pair, trim later
            )
            strategy_results.extend(pair_results)

        # Sort combined results and take top N
        strategy_results.sort(key=lambda r: (r["profit_factor"], r["total_pnl"]), reverse=True)
        all_results[strategy_name] = strategy_results[:top_n]

        if strategy_results:
            best = strategy_results[0]
            print(f"  Best for {strategy_name}: PF={best['profit_factor']:.2f} "
                  f"PnL=${best['total_pnl']:+.2f} WR={best['win_rate']:.1%} "
                  f"({best['total_trades']} trades) [{best['pair']}]")
        else:
            print(f"  No profitable results for {strategy_name}")

    elapsed_global = time.time() - t0_global
    total_results = sum(len(v) for v in all_results.values())

    print(f"\n{'='*70}")
    print(f"[Optimizer] Complete in {elapsed_global:.1f}s")
    print(f"  Total results retained: {total_results}")
    print(f"{'='*70}")

    return all_results


# ---------------------------------------------------------------------------
# Utility: print formatted results
# ---------------------------------------------------------------------------

def print_results(results: dict[str, list[dict]], top: int = 5):
    """Pretty-print the top results for each strategy."""
    for strategy_name, entries in results.items():
        print(f"\n{'='*60}")
        print(f"  {strategy_name.upper()}")
        print(f"{'='*60}")

        if not entries:
            print("  (no results)")
            continue

        for i, r in enumerate(entries[:top], 1):
            print(f"\n  #{i}  {r['pair']}")
            print(f"    PF={r['profit_factor']:.2f}  PnL=${r['total_pnl']:+.2f}  "
                  f"WR={r['win_rate']:.1%}  Trades={r['total_trades']}")
            print(f"    MDD={r['max_drawdown_pct']:.2%}  Sharpe={r['sharpe_ratio']:.2f}")
            print(f"    AvgWin={r['avg_win_pct']:.4%}  AvgLoss={r['avg_loss_pct']:.4%}")
            # Print strategy-specific params (exclude common keys)
            common_keys = {"strategy_name", "sl_pct", "tp_pct", "trailing_pct",
                           "cooldown_bars", "leverage", "position_pct"}
            specific = {k: v for k, v in r["params"].items() if k not in common_keys}
            common = {k: v for k, v in r["params"].items() if k in common_keys and k != "strategy_name"}
            if specific:
                print(f"    Strategy params: {specific}")
            if common:
                print(f"    Trade params: {common}")


# ---------------------------------------------------------------------------
# Utility: save / load results
# ---------------------------------------------------------------------------

def save_results(results: dict[str, list[dict]], path: str):
    """Save optimization results to a JSON file."""
    import json
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {path}")


def load_results(path: str) -> dict[str, list[dict]]:
    """Load optimization results from a JSON file."""
    import json
    with open(path) as f:
        return json.load(f)
