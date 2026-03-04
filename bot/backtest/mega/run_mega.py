#!/usr/bin/env python3
"""
MEGA BACKTEST: 27 Strategies x 19 Pairs x Multi-Timeframe
==========================================================

Entry point for the mega parameter sweep. Uses multiprocessing for
parallel simulation across all CPU cores.

Usage:
    python bot/backtest/mega/run_mega.py              # All 19 pairs
    python bot/backtest/mega/run_mega.py BTCUSDT ETHUSDT  # Specific pairs
    python bot/backtest/mega/run_mega.py --days 730    # 2 years
    python bot/backtest/mega/run_mega.py --tier1-only  # Skip Tier 2
"""

import sys
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.data_loader import download_klines, load_proxy_url
from backtest.indicators import compute_all_indicators
from backtest.mega.indicators_extended import compute_extended_indicators, resample_1h_to_4h
from backtest.mega.strategies_mega import (
    STRATEGY_DETECTORS, MTF_STRATEGIES, detect_all_signals
)
from backtest.mega.simulator import simulate
from backtest.mega.param_grids import get_tier1_grid, get_tier2_grid
from backtest.mega import aggregator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ALL_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "SUIUSDT", "ARBUSDT", "OPUSDT", "APTUSDT",
    "NEARUSDT", "LTCUSDT", "ATOMUSDT", "FILUSDT", "INJUSDT",
]

CACHE_DIR = Path(__file__).parent / "cache"


# ---------------------------------------------------------------------------
# Phase 1: Data loading + indicator computation
# ---------------------------------------------------------------------------

def load_pair_data(pair: str, days: int, proxy_url: str) -> tuple[str, pd.DataFrame | None, pd.DataFrame | None]:
    """Load data for a single pair, compute all indicators.

    Returns: (pair, df_1h_with_indicators, df_4h_with_indicators)
    """
    try:
        df = download_klines(pair, interval="1h", days=days, proxy_url=proxy_url)
        if df.empty or len(df) < 200:
            return pair, None, None

        # Base indicators
        df = compute_all_indicators(df)
        # Extended indicators
        df = compute_extended_indicators(df)

        # 4H resample + indicators
        df_4h = resample_1h_to_4h(df)
        if len(df_4h) >= 60:
            df_4h = compute_all_indicators(df_4h)
        else:
            df_4h = None

        return pair, df, df_4h
    except Exception as e:
        print(f"  ERROR loading {pair}: {e}", flush=True)
        return pair, None, None


# ---------------------------------------------------------------------------
# Phase 2: Signal detection (per pair)
# ---------------------------------------------------------------------------

def detect_pair_signals(pair: str, df: pd.DataFrame, df_4h: pd.DataFrame | None) -> dict[str, list[dict]]:
    """Detect signals for all 27 strategies on a single pair."""
    return detect_all_signals(df, df_4h)


# ---------------------------------------------------------------------------
# Phase 3: Simulation (parallelized work units)
# ---------------------------------------------------------------------------

def _sim_job(args):
    """Single simulation job for multiprocessing.

    Args is a tuple: (pair, strategy, params, df_values_dict, signal_list)
    We pass pre-extracted numpy arrays to avoid pickling full DataFrames.
    """
    pair, strategy, params, df_dict, signals = args
    if not signals:
        return None

    # Reconstruct minimal DataFrame from dict of arrays
    df = pd.DataFrame(df_dict)

    result = simulate(df, signals, params, pair)
    if result and result["total_trades"] >= 5:
        result["pair"] = pair
        result["strategy"] = strategy
        result["params"] = {k: v for k, v in params.items()}
        return result
    return None


def _prepare_df_dict(df: pd.DataFrame) -> dict:
    """Extract minimal columns needed for simulation as a serializable dict."""
    cols = ["high", "low", "close"]
    if "timestamp" in df.columns:
        cols.append("timestamp")
    return {c: df[c].values for c in cols if c in df.columns}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mega Backtest")
    parser.add_argument("pairs", nargs="*", default=ALL_PAIRS, help="Pairs to test")
    parser.add_argument("--days", type=int, default=730, help="Days of history (default: 730 = 2 years)")
    parser.add_argument("--tier1-only", action="store_true", help="Skip Tier 2")
    parser.add_argument("--workers", type=int, default=0, help="Max workers (0=auto)")
    args = parser.parse_args()

    pairs = args.pairs
    days = args.days
    n_workers = args.workers or mp.cpu_count()
    t0 = time.time()

    proxy_url = load_proxy_url()

    print("=" * 80, flush=True)
    print("MEGA BACKTEST — 27 Strategies x 19 Pairs x Multi-Timeframe", flush=True)
    print(f"$50K | {len(pairs)} pairs | {days} days (~{days/365:.1f} years) | {n_workers} cores", flush=True)
    print("=" * 80, flush=True)

    tier1_grid = get_tier1_grid()
    tier2_grid = get_tier2_grid()
    n_strats = len(STRATEGY_DETECTORS)
    print(f"Strategies: {n_strats}", flush=True)
    print(f"Tier 1 grid: {len(tier1_grid)} combos/strategy", flush=True)
    print(f"Tier 2 grid: {len(tier2_grid)} combos/strategy", flush=True)
    print(flush=True)

    # ===================================================================
    # PHASE 1: Load data + compute indicators
    # ===================================================================
    print("PHASE 1: Loading data + computing indicators...", flush=True)
    t1 = time.time()

    data_1h = {}
    data_4h = {}

    for pair in pairs:
        print(f"  {pair}...", end=" ", flush=True)
        _, df, df_4h = load_pair_data(pair, days, proxy_url)
        if df is not None:
            data_1h[pair] = df
            if df_4h is not None:
                data_4h[pair] = df_4h
            ts_first = pd.Timestamp(df["timestamp"].iloc[0])
            ts_last = pd.Timestamp(df["timestamp"].iloc[-1])
            n_4h = len(df_4h) if df_4h is not None else 0
            print(f"{len(df)} 1H + {n_4h} 4H candles "
                  f"({ts_first.strftime('%Y-%m-%d')} to {ts_last.strftime('%Y-%m-%d')})", flush=True)
        else:
            print("FAILED", flush=True)
        time.sleep(0.5)  # Rate limit

    print(f"\n  Loaded: {len(data_1h)} pairs ({time.time()-t1:.1f}s)", flush=True)

    if not data_1h:
        print("ERROR: No data loaded. Exiting.", flush=True)
        return

    # ===================================================================
    # PHASE 2: Detect signals for all strategies
    # ===================================================================
    print(f"\nPHASE 2: Detecting signals (27 strategies x {len(data_1h)} pairs)...", flush=True)
    t2 = time.time()

    # {pair: {strategy: [signals]}}
    pair_signals = {}
    total_signals = 0

    for pair in data_1h:
        df = data_1h[pair]
        df_4h = data_4h.get(pair)
        strat_signals = detect_pair_signals(pair, df, df_4h)
        pair_signals[pair] = strat_signals

        # Count signals
        pair_total = sum(len(sigs) for sigs in strat_signals.values())
        total_signals += pair_total
        active = [f"{s}:{len(v)}" for s, v in strat_signals.items() if len(v) > 0]
        print(f"  {pair}: {pair_total} signals ({len(active)} active strategies)", flush=True)

    print(f"\n  Total signals: {total_signals:,} ({time.time()-t2:.1f}s)", flush=True)

    # ===================================================================
    # PHASE 3a: Tier 1 Simulation
    # ===================================================================
    print(f"\nPHASE 3a: Tier 1 Simulation...", flush=True)
    t3a = time.time()

    # Build job list
    jobs = []
    for pair in data_1h:
        df_dict = _prepare_df_dict(data_1h[pair])
        for strat, sigs in pair_signals[pair].items():
            if not sigs:
                continue
            for params in tier1_grid:
                jobs.append((pair, strat, params, df_dict, sigs))

    print(f"  Jobs: {len(jobs):,} simulations across {n_workers} cores", flush=True)

    # Run in parallel
    tier1_results = []
    with mp.Pool(n_workers) as pool:
        chunk_size = max(1, len(jobs) // (n_workers * 4))
        done = 0
        for result in pool.imap_unordered(_sim_job, jobs, chunksize=chunk_size):
            done += 1
            if result:
                tier1_results.append(result)
            if done % 10000 == 0:
                elapsed = time.time() - t3a
                speed = done / max(elapsed, 0.1)
                eta = (len(jobs) - done) / max(speed, 1)
                print(f"    {done:,}/{len(jobs):,} ({done/len(jobs)*100:.0f}%) | "
                      f"{speed:,.0f} sims/s | ETA: {eta:.0f}s | "
                      f"qualified: {len(tier1_results):,}", flush=True)

    elapsed_3a = time.time() - t3a
    print(f"\n  Tier 1 done: {len(tier1_results):,} qualified results from "
          f"{len(jobs):,} sims ({elapsed_3a:.1f}s, "
          f"{len(jobs)/max(elapsed_3a,0.1):,.0f} sims/s)", flush=True)

    # ===================================================================
    # PHASE 3b: Tier 1 Analysis + select top strategies
    # ===================================================================
    print(f"\nPHASE 3b: Tier 1 Analysis...", flush=True)

    tier1_rankings = aggregator.rank_strategies(tier1_results)
    tier1_cross = aggregator.find_cross_pair_configs(tier1_results, min_pairs=3)
    tier1_pair_best = aggregator.per_pair_best(tier1_results)

    aggregator.print_summary(tier1_rankings, tier1_cross, tier1_pair_best,
                             "Tier 1", len(jobs), elapsed_3a)

    # Save Tier 1
    out_dir = aggregator.save_results(tier1_results, tier1_rankings, tier1_cross,
                                       tier1_pair_best, "tier1", elapsed_3a)
    print(f"\n  Tier 1 results saved to: {out_dir}", flush=True)

    # Select top 10 for Tier 2
    top_strategies = aggregator.select_top_strategies(tier1_rankings, 10)
    print(f"\n  Top 10 for Tier 2: {', '.join(top_strategies)}", flush=True)

    if args.tier1_only:
        total_elapsed = time.time() - t0
        print(f"\n{'='*80}")
        print(f"MEGA BACKTEST COMPLETE (Tier 1 only) — {total_elapsed:.1f}s total")
        print(f"{'='*80}")
        return

    # ===================================================================
    # PHASE 3c: Tier 2 Simulation (expanded grid on top 10)
    # ===================================================================
    print(f"\nPHASE 3c: Tier 2 Simulation (top 10 strategies, expanded grid)...", flush=True)
    t3c = time.time()

    jobs_t2 = []
    for pair in data_1h:
        df_dict = _prepare_df_dict(data_1h[pair])
        for strat in top_strategies:
            sigs = pair_signals[pair].get(strat, [])
            if not sigs:
                continue
            for params in tier2_grid:
                jobs_t2.append((pair, strat, params, df_dict, sigs))

    print(f"  Jobs: {len(jobs_t2):,} simulations", flush=True)

    tier2_results = []
    with mp.Pool(n_workers) as pool:
        chunk_size = max(1, len(jobs_t2) // (n_workers * 4))
        done = 0
        for result in pool.imap_unordered(_sim_job, jobs_t2, chunksize=chunk_size):
            done += 1
            if result:
                tier2_results.append(result)
            if done % 10000 == 0:
                elapsed = time.time() - t3c
                speed = done / max(elapsed, 0.1)
                eta = (len(jobs_t2) - done) / max(speed, 1)
                print(f"    {done:,}/{len(jobs_t2):,} ({done/len(jobs_t2)*100:.0f}%) | "
                      f"{speed:,.0f} sims/s | ETA: {eta:.0f}s | "
                      f"qualified: {len(tier2_results):,}", flush=True)

    elapsed_3c = time.time() - t3c
    print(f"\n  Tier 2 done: {len(tier2_results):,} qualified results from "
          f"{len(jobs_t2):,} sims ({elapsed_3c:.1f}s)", flush=True)

    # ===================================================================
    # PHASE 4: Final aggregation (Tier 1 + Tier 2 combined)
    # ===================================================================
    print(f"\nPHASE 4: Final Aggregation...", flush=True)
    t4 = time.time()

    all_results = tier1_results + tier2_results

    final_rankings = aggregator.rank_strategies(all_results)
    final_cross = aggregator.find_cross_pair_configs(all_results, min_pairs=5)
    final_pair_best = aggregator.per_pair_best(all_results)

    total_sims = len(jobs) + len(jobs_t2)
    total_elapsed = time.time() - t0

    aggregator.print_summary(final_rankings, final_cross, final_pair_best,
                             "FINAL (Tier1+Tier2)", total_sims, total_elapsed)

    # Save final
    out_dir = aggregator.save_results(all_results, final_rankings, final_cross,
                                       final_pair_best, "final", total_elapsed)

    # ===================================================================
    # Verification: compare macd_cross_1h with previous backtest
    # ===================================================================
    macd_results = [r for r in all_results
                    if r["strategy"] == "macd_cross_1h"
                    and r["params"].get("sl_pct") == 0.03
                    and r["params"].get("tp_pct") == 0.06
                    and r["params"].get("trailing_pct") == 0.06
                    and r["params"].get("leverage") == 10]

    if macd_results:
        total_macd_pnl = sum(r["total_pnl"] for r in macd_results)
        n_profitable = sum(1 for r in macd_results if r["profit_factor"] > 1.0)
        print(f"\n--- VERIFICATION: macd_cross_1h (SL=3% TP=6% Trail=6% Lev=10x) ---")
        print(f"  Total PnL across {len(macd_results)} pairs: ${total_macd_pnl:+,.2f}")
        print(f"  Profitable: {n_profitable}/{len(macd_results)} pairs")
        print(f"  (Previous 2-year backtest reference: +$86,964)")

    print(f"\n{'='*80}")
    print(f"MEGA BACKTEST COMPLETE")
    print(f"{'='*80}")
    print(f"Total simulations: {total_sims:,}")
    print(f"Total time: {total_elapsed:.1f}s ({total_sims/max(total_elapsed,0.1):,.0f} sims/s)")
    print(f"Results saved to: {out_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
