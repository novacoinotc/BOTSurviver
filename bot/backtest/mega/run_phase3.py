#!/usr/bin/env python3
"""
MEGA BACKTEST PHASE 3: Multi-Timeframe + New Strategies
=======================================================

Explores everything pending before closing the research:
1. New strategies: BTC Correlation, Fibonacci, Heikin-Ashi, Range Compression,
   Momentum Exhaustion, EMA Ribbon
2. Failed 1H strategies tested on 15m and 4h
3. Top strategies from Phase 1 on multi-timeframe
4. Walk-forward of new winners
5. Final verdict: Phase 1 vs Phase 2 vs Phase 3

Usage:
    python bot/backtest/mega/run_phase3.py
    python bot/backtest/mega/run_phase3.py --pairs BTCUSDT ETHUSDT
    python bot/backtest/mega/run_phase3.py --skip-walkforward
"""

import sys
import time
import json
import argparse
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.data_loader import download_klines, load_proxy_url
from backtest.indicators import compute_all_indicators
from backtest.mega.indicators_extended import compute_extended_indicators
from backtest.mega.strategies_mega import STRATEGY_DETECTORS, MTF_STRATEGIES
from backtest.mega.strategies_phase3 import (
    PHASE3_STRATEGIES, BTC_STRATEGIES, STANDARD_PHASE3,
    precompute_btc_beta, detect_btc_correlation_lag,
)
from backtest.mega.simulator import simulate
from backtest.mega.param_grids import get_tier1_grid, get_tier2_grid
from backtest.mega import aggregator
from backtest.mega.walk_forward import walk_forward_validate, aggregate_walk_forward

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ALL_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "SUIUSDT", "ARBUSDT", "OPUSDT", "APTUSDT",
    "NEARUSDT", "LTCUSDT", "ATOMUSDT", "FILUSDT", "INJUSDT",
]

OUTPUT_DIR = Path(__file__).parent / "results"

# Timeframe configs: (interval, bar_hours, cooldown_scale)
TIMEFRAMES = {
    "15m": {"interval": "15m", "bar_hours": 0.25, "cd_scale": 4},
    "1h":  {"interval": "1h",  "bar_hours": 1.0,  "cd_scale": 1},
    "4h":  {"interval": "4h",  "bar_hours": 4.0,  "cd_scale": 0.25},
}

# Phase 1 strategies that failed on 1H — test on other TFs
FAILED_1H_STRATEGIES = [
    "zscore_reversion", "rsi_divergence_entry", "mean_reversion",
    "bollinger_bounce", "stoch_reversal",
]

# Extra strategies for 4H (may work better on higher TF)
EXTRA_4H_STRATEGIES = [
    "regime_adaptive", "momentum",
]

# Phase 3 new strategy names
NEW_STRATEGY_NAMES = list(PHASE3_STRATEGIES.keys())


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data_for_timeframe(pairs: list[str], tf: str, days: int,
                            proxy_url: str) -> dict[str, pd.DataFrame]:
    """Load + compute indicators for all pairs on a given timeframe."""
    interval = TIMEFRAMES[tf]["interval"]
    data = {}
    for pair in pairs:
        try:
            df = download_klines(pair, interval=interval, days=days,
                                proxy_url=proxy_url)
            if df is None or df.empty or len(df) < 100:
                print(f"    {pair}: SKIP (insufficient data)", flush=True)
                continue
            df = compute_all_indicators(df)
            df = compute_extended_indicators(df)
            data[pair] = df
            print(f"    {pair}: {len(df)} bars", flush=True)
        except Exception as e:
            print(f"    {pair}: ERROR {e}", flush=True)
        time.sleep(0.3)
    return data


# ---------------------------------------------------------------------------
# Signal Detection
# ---------------------------------------------------------------------------

def detect_phase3_signals(df: pd.DataFrame, btc_data: pd.DataFrame = None) -> dict:
    """Detect signals for all Phase 3 new strategies."""
    results = {}
    for name, func in STANDARD_PHASE3.items():
        results[name] = func(df)

    # BTC correlation (special handling)
    if btc_data is not None:
        results["btc_correlation_lag"] = detect_btc_correlation_lag(df, btc_data)
    else:
        results["btc_correlation_lag"] = []

    return results


def detect_existing_signals(df: pd.DataFrame, strategy_names: list[str]) -> dict:
    """Detect signals for existing Phase 1 strategies."""
    results = {}
    for name in strategy_names:
        func = STRATEGY_DETECTORS.get(name)
        if func is None:
            continue
        if name in MTF_STRATEGIES:
            results[name] = func(df, None)
        else:
            results[name] = func(df)
    return results


# ---------------------------------------------------------------------------
# Simulation Jobs
# ---------------------------------------------------------------------------

def _sim_job(args):
    """Single simulation job for multiprocessing."""
    pair, strategy, params, df_dict, signals, bar_hours = args
    if not signals:
        return None
    df = pd.DataFrame(df_dict)
    result = simulate(df, signals, params, pair, bar_hours=bar_hours)
    if result and result["total_trades"] >= 5:
        result["pair"] = pair
        result["strategy"] = strategy
        result["params"] = {k: v for k, v in params.items()}
        return result
    return None


def _prepare_df_dict(df: pd.DataFrame) -> dict:
    """Extract minimal columns for simulation."""
    cols = ["high", "low", "close"]
    if "timestamp" in df.columns:
        cols.append("timestamp")
    return {c: df[c].values for c in cols if c in df.columns}


def scale_cooldown_grid(grid: list[dict], cd_scale: float) -> list[dict]:
    """Scale cooldown_bars in parameter grid for different timeframes."""
    if cd_scale == 1:
        return grid
    scaled = []
    for p in grid:
        sp = dict(p)
        sp["cooldown_bars"] = max(1, int(p["cooldown_bars"] * cd_scale))
        scaled.append(sp)
    return scaled


# ---------------------------------------------------------------------------
# Run simulation batch
# ---------------------------------------------------------------------------

def run_simulations(jobs: list, n_workers: int, label: str) -> list[dict]:
    """Run sim jobs in parallel and return qualified results."""
    if not jobs:
        print(f"  {label}: No jobs to run", flush=True)
        return []

    print(f"  {label}: {len(jobs):,} simulations across {n_workers} cores",
          flush=True)
    t0 = time.time()
    results = []

    with mp.Pool(n_workers) as pool:
        chunk_size = max(1, len(jobs) // (n_workers * 4))
        done = 0
        for result in pool.imap_unordered(_sim_job, jobs, chunksize=chunk_size):
            done += 1
            if result:
                results.append(result)
            if done % 10000 == 0:
                elapsed = time.time() - t0
                speed = done / max(elapsed, 0.1)
                eta = (len(jobs) - done) / max(speed, 1)
                print(f"    {done:,}/{len(jobs):,} ({done/len(jobs)*100:.0f}%) "
                      f"| {speed:,.0f} sims/s | ETA: {eta:.0f}s | "
                      f"qualified: {len(results):,}", flush=True)

    elapsed = time.time() - t0
    print(f"  {label}: {len(results):,} qualified from {len(jobs):,} sims "
          f"({elapsed:.1f}s, {len(jobs)/max(elapsed,0.1):,.0f} sims/s)",
          flush=True)
    return results


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------

def run_walk_forward(data: dict, pair_signals: dict, top_configs: list[dict],
                     bar_hours: float, n_top: int = 10) -> list[dict]:
    """Run walk-forward on top N configs."""
    if not top_configs:
        return []

    configs_to_test = top_configs[:n_top]
    all_wf = []

    for cfg in configs_to_test:
        strat = cfg["strategy"]
        params = cfg["params"]
        for pair, df in data.items():
            sigs = pair_signals.get(pair, {}).get(strat, [])
            if len(sigs) < 5:
                continue
            wf = walk_forward_validate(df, sigs, params, pair, strat,
                                       simulate, method="all")
            all_wf.extend(wf)

    return all_wf


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_phase3_report(all_results: list[dict], wf_summaries: list[dict],
                           elapsed: float) -> str:
    """Generate Phase 3 summary markdown report."""
    lines = []
    lines.append("# Mega Backtest Phase 3: Multi-Timeframe + New Strategies")
    lines.append("")
    lines.append(f"**Runtime:** {elapsed:.1f}s | **Initial equity:** $50,000")
    lines.append("")

    # Group results by timeframe
    tf_groups = defaultdict(list)
    for r in all_results:
        tf = r.get("timeframe", "1h")
        tf_groups[tf].append(r)

    for tf in ["15m", "1h", "4h"]:
        results = tf_groups.get(tf, [])
        if not results:
            continue
        lines.append(f"## {tf.upper()} Results ({len(results):,} qualified configs)")
        lines.append("")
        rankings = aggregator.rank_strategies(results)
        lines.append("| # | Strategy | Profit Rate | Pairs | Best PnL | "
                     "Median PnL | Avg PF | Avg WR |")
        lines.append("|---|----------|------------|-------|----------|"
                     "-----------|--------|--------|")
        for i, r in enumerate(rankings[:20], 1):
            lines.append(
                f"| {i} | {r['strategy']} | {r['profit_rate']}% | "
                f"{r['pairs_profitable']}/{r['pairs_total']} | "
                f"${r['best_pnl']:+,.0f} | ${r['median_pnl']:+,.0f} | "
                f"{r['avg_pf']:.2f} | {r['avg_win_rate']:.1f}% |"
            )
        lines.append("")

    # Cross-pair configs (all TFs combined)
    lines.append("## Top Cross-Pair Configs (all timeframes)")
    lines.append("")
    cross = aggregator.find_cross_pair_configs(all_results, min_pairs=3)
    if cross:
        lines.append("| # | Strategy | TF | Pairs | Total PnL | Avg PF | "
                     "WR | Trades |")
        lines.append("|---|----------|-----|-------|-----------|--------|"
                     "-----|--------|")
        for i, c in enumerate(cross[:20], 1):
            p = c["params"]
            lines.append(
                f"| {i} | {c['strategy']} | — | {c['n_pairs']} | "
                f"${c['total_pnl']:+,.0f} | {c['avg_pf']:.2f} | "
                f"{c['avg_win_rate']:.1f}% | {c['total_trades']} |"
            )
            lines.append(
                f"|   | SL={p['sl_pct']*100:.0f}% TP={p['tp_pct']*100:.0f}% "
                f"Trail={p['trailing_pct']*100:.0f}% Lev={p['leverage']}x "
                f"CD={p['cooldown_bars']}bars | | | | | | |"
            )
    lines.append("")

    # Walk-forward
    if wf_summaries:
        lines.append("## Walk-Forward Validation")
        lines.append("")
        lines.append("| # | Strategy | Verdict | Avg Degrad | Windows | "
                     "Train PF | Test PF |")
        lines.append("|---|----------|---------|-----------|---------|"
                     "---------|---------|")
        for i, w in enumerate(wf_summaries[:15], 1):
            lines.append(
                f"| {i} | {w['strategy']} | {w['overall_verdict']} | "
                f"{w['avg_degradation']*100:.1f}% | {w['n_windows']} | "
                f"{w['avg_train_pf']:.2f} | {w['avg_test_pf']:.2f} |"
            )
        lines.append("")

    return "\n".join(lines)


def generate_final_verdict(all_p3_results: list[dict],
                           wf_summaries: list[dict]) -> str:
    """Generate FINAL_VERDICT.md comparing all phases."""
    lines = []
    lines.append("# FINAL VERDICT: Which Strategy to Deploy in Production")
    lines.append("")
    lines.append("## Phase Winners")
    lines.append("")
    lines.append("| Phase | Strategy | Total PnL | PF | WR | Timeframe | "
                "Walk-Forward |")
    lines.append("|-------|----------|-----------|----|----|-----------|"
                "-------------|")
    lines.append("| Phase 1 | volume_spike | $231,069 | 1.29 | 48% | 1H | "
                "ROBUST |")
    lines.append("| Phase 2 | vs_ema_aligned (combo) | $69,000 (best pair) | "
                "— | — | 1H | ROBUST |")

    # Find Phase 3 best cross-pair
    if all_p3_results:
        cross = aggregator.find_cross_pair_configs(all_p3_results, min_pairs=3)
        if cross:
            best = cross[0]
            p = best["params"]
            # Find walk-forward verdict if available
            wf_verdict = "N/A"
            for w in wf_summaries:
                if w["strategy"] == best["strategy"]:
                    wf_verdict = w["overall_verdict"]
                    break
            lines.append(
                f"| Phase 3 | {best['strategy']} | "
                f"${best['total_pnl']:+,.0f} | {best['avg_pf']:.2f} | "
                f"{best['avg_win_rate']:.1f}% | — | {wf_verdict} |"
            )
            lines.append(
                f"|         | SL={p['sl_pct']*100:.0f}% TP={p['tp_pct']*100:.0f}% "
                f"Trail={p['trailing_pct']*100:.0f}% Lev={p['leverage']}x "
                f"CD={p['cooldown_bars']}bars | | | | | |"
            )

    lines.append("")
    lines.append("## Analysis")
    lines.append("")

    # Strategy ranking across all Phase 3
    if all_p3_results:
        rankings = aggregator.rank_strategies(all_p3_results)
        lines.append("### Phase 3 Strategy Ranking")
        lines.append("")
        for i, r in enumerate(rankings[:10], 1):
            lines.append(
                f"{i}. **{r['strategy']}**: {r['pairs_profitable']}/{r['pairs_total']} "
                f"pairs profitable, median PnL=${r['median_pnl']:+,.0f}, "
                f"avg PF={r['avg_pf']:.2f}"
            )
        lines.append("")

    # Multi-TF analysis
    tf_data = defaultdict(list)
    for r in all_p3_results:
        tf_data[r.get("timeframe", "1h")].append(r)

    lines.append("### Multi-Timeframe Comparison")
    lines.append("")
    for tf in ["15m", "1h", "4h"]:
        results = tf_data.get(tf, [])
        if not results:
            continue
        profitable = [r for r in results if r["profit_factor"] > 1.0]
        total_pnl = sum(r["total_pnl"] for r in results)
        lines.append(f"- **{tf}**: {len(results)} configs tested, "
                    f"{len(profitable)} profitable ({len(profitable)/max(len(results),1)*100:.0f}%), "
                    f"total PnL=${total_pnl:+,.0f}")
    lines.append("")

    # Walk-forward summary
    if wf_summaries:
        robust = [w for w in wf_summaries if w["overall_verdict"] == "ROBUST"]
        marginal = [w for w in wf_summaries if w["overall_verdict"] == "MARGINAL"]
        overfit = [w for w in wf_summaries if w["overall_verdict"] == "OVERFIT"]
        lines.append("### Walk-Forward Results")
        lines.append("")
        lines.append(f"- ROBUST: {len(robust)} configs")
        lines.append(f"- MARGINAL: {len(marginal)} configs")
        lines.append(f"- OVERFIT: {len(overfit)} configs")
        lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    lines.append("Based on cross-phase analysis:")
    lines.append("")
    lines.append("1. **Primary production strategy**: volume_spike 1H "
                "(Phase 1/2 winner, ROBUST walk-forward)")
    lines.append("   - SL=3% TP=4% Trail=6% CD=48h Lev=10x Pos=2%")
    lines.append("   - $231K total PnL, PF=1.29, WR=48%")
    lines.append("")

    # Check if Phase 3 has anything better
    if all_p3_results:
        cross = aggregator.find_cross_pair_configs(all_p3_results, min_pairs=5)
        better = [c for c in cross if c["total_pnl"] > 231_000
                  and c["avg_pf"] > 1.2]
        if better:
            b = better[0]
            lines.append(f"2. **Phase 3 contender**: {b['strategy']} "
                        f"(${b['total_pnl']:+,.0f}, PF={b['avg_pf']:.2f})")
            lines.append("   - Needs further paper trading validation "
                        "before deployment")
        else:
            lines.append("2. **Phase 3 finding**: No new strategy "
                        "consistently beat volume_spike across 5+ pairs")

    lines.append("")
    lines.append("---")
    lines.append("*Generated by Mega Backtest Phase 3*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mega Backtest Phase 3")
    parser.add_argument("--pairs", nargs="*", default=ALL_PAIRS)
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--skip-walkforward", action="store_true")
    parser.add_argument("--skip-15m", action="store_true")
    args = parser.parse_args()

    pairs = args.pairs
    days = args.days
    n_workers = args.workers or mp.cpu_count()
    t0 = time.time()

    proxy_url = load_proxy_url()
    tier1_grid = get_tier1_grid()
    tier2_grid = get_tier2_grid()

    print("=" * 80, flush=True)
    print("MEGA BACKTEST PHASE 3: Multi-Timeframe + New Strategies", flush=True)
    print(f"$50K | {len(pairs)} pairs | {days}d (~{days/365:.1f}y) | "
          f"{n_workers} cores", flush=True)
    print(f"Tier1: {len(tier1_grid)} combos | Tier2: {len(tier2_grid)} combos",
          flush=True)
    print(f"New strategies: {', '.join(NEW_STRATEGY_NAMES)}", flush=True)
    print("=" * 80, flush=True)

    all_results = []
    all_pair_signals = {}  # {tf: {pair: {strat: signals}}}

    # ==================================================================
    # PHASE 1: Load data for each timeframe
    # ==================================================================
    print("\n[1/6] LOADING DATA...", flush=True)
    data_by_tf = {}

    for tf in ["1h", "4h"] + ([] if args.skip_15m else ["15m"]):
        print(f"\n  Loading {tf} data for {len(pairs)} pairs...", flush=True)
        data_by_tf[tf] = load_data_for_timeframe(pairs, tf, days, proxy_url)
        print(f"  {tf}: loaded {len(data_by_tf[tf])} pairs", flush=True)

    if not data_by_tf.get("1h"):
        print("ERROR: No 1H data loaded. Exiting.", flush=True)
        return

    # ==================================================================
    # PHASE 2: Precompute BTC beta for BTC Correlation strategy
    # ==================================================================
    print("\n[2/6] PRECOMPUTING BTC BETA...", flush=True)
    btc_beta_by_tf = {}  # {tf: {pair: btc_beta_df}}

    for tf, data in data_by_tf.items():
        btc_df = data.get("BTCUSDT")
        if btc_df is None:
            print(f"  {tf}: no BTC data, skipping btc_correlation_lag",
                  flush=True)
            btc_beta_by_tf[tf] = {}
            continue

        btc_beta_by_tf[tf] = {}
        for pair, df in data.items():
            if pair == "BTCUSDT":
                continue
            try:
                btc_beta_by_tf[tf][pair] = precompute_btc_beta(btc_df, df)
            except Exception as e:
                print(f"  {tf} {pair}: beta error: {e}", flush=True)

        print(f"  {tf}: computed beta for {len(btc_beta_by_tf[tf])} pairs",
              flush=True)

    # ==================================================================
    # PHASE 3: Detect signals for all strategy x TF combos
    # ==================================================================
    print("\n[3/6] DETECTING SIGNALS...", flush=True)

    for tf, data in data_by_tf.items():
        all_pair_signals[tf] = {}
        total_sigs = 0

        for pair, df in data.items():
            pair_sigs = {}

            # 1. Phase 3 new strategies (on ALL timeframes)
            btc_data = btc_beta_by_tf.get(tf, {}).get(pair)
            p3_sigs = detect_phase3_signals(df, btc_data)
            pair_sigs.update(p3_sigs)

            # 2. Failed 1H strategies (on 15m and 4h only)
            if tf in ("15m", "4h"):
                failed_sigs = detect_existing_signals(df, FAILED_1H_STRATEGIES)
                pair_sigs.update(failed_sigs)

            # 3. Extra strategies on 4h
            if tf == "4h":
                extra_sigs = detect_existing_signals(df, EXTRA_4H_STRATEGIES)
                pair_sigs.update(extra_sigs)

            # 4. Benchmark: volume_spike on 15m and 4h
            if tf in ("15m", "4h"):
                vs_sigs = detect_existing_signals(df, ["volume_spike"])
                pair_sigs.update(vs_sigs)

            all_pair_signals[tf][pair] = pair_sigs
            pair_total = sum(len(s) for s in pair_sigs.values())
            total_sigs += pair_total

        active_strats = set()
        for ps in all_pair_signals[tf].values():
            for s, sigs in ps.items():
                if sigs:
                    active_strats.add(s)

        print(f"  {tf}: {total_sigs:,} total signals, "
              f"{len(active_strats)} active strategies", flush=True)

    # ==================================================================
    # PHASE 4: Simulations (Tier 1 + auto Tier 2)
    # ==================================================================
    print("\n[4/6] RUNNING SIMULATIONS...", flush=True)

    for tf, data in data_by_tf.items():
        tf_cfg = TIMEFRAMES[tf]
        bar_hours = tf_cfg["bar_hours"]
        cd_scale = tf_cfg["cd_scale"]

        # Scale cooldown for timeframe
        scaled_t1 = scale_cooldown_grid(tier1_grid, cd_scale)

        # Build Tier 1 jobs
        jobs = []
        for pair, df in data.items():
            df_dict = _prepare_df_dict(df)
            for strat, sigs in all_pair_signals[tf].get(pair, {}).items():
                if not sigs:
                    continue
                for params in scaled_t1:
                    jobs.append((pair, strat, params, df_dict, sigs, bar_hours))

        if not jobs:
            continue

        # Tag timeframe before running
        tier1_results = run_simulations(jobs, n_workers,
                                        f"Tier1 {tf}")

        # Tag timeframe on results
        for r in tier1_results:
            r["timeframe"] = tf

        # Select top 5 strategies for Tier 2
        if tier1_results:
            t1_rankings = aggregator.rank_strategies(tier1_results)
            top_strats = aggregator.select_top_strategies(t1_rankings, 5)
            print(f"  {tf} Tier1 top 5: {', '.join(top_strats)}", flush=True)

            # Build Tier 2 jobs
            scaled_t2 = scale_cooldown_grid(tier2_grid, cd_scale)
            jobs_t2 = []
            for pair, df in data.items():
                df_dict = _prepare_df_dict(df)
                for strat in top_strats:
                    sigs = all_pair_signals[tf].get(pair, {}).get(strat, [])
                    if not sigs:
                        continue
                    for params in scaled_t2:
                        jobs_t2.append((pair, strat, params, df_dict, sigs,
                                       bar_hours))

            tier2_results = run_simulations(jobs_t2, n_workers,
                                            f"Tier2 {tf}")
            for r in tier2_results:
                r["timeframe"] = tf

            all_results.extend(tier1_results)
            all_results.extend(tier2_results)
        else:
            all_results.extend(tier1_results)

    print(f"\n  Total qualified results: {len(all_results):,}", flush=True)

    # ==================================================================
    # PHASE 5: Walk-Forward Validation
    # ==================================================================
    wf_summaries = []
    wf_raw = []

    if not args.skip_walkforward and all_results:
        print("\n[5/6] WALK-FORWARD VALIDATION...", flush=True)

        # Get top 10 cross-pair configs
        cross = aggregator.find_cross_pair_configs(all_results, min_pairs=3)
        if cross:
            top_wf_configs = cross[:10]
            print(f"  Testing {len(top_wf_configs)} cross-pair configs...",
                  flush=True)

            for cfg in top_wf_configs:
                strat = cfg["strategy"]
                params = cfg["params"]
                # Find which TF this came from
                for tf, data in data_by_tf.items():
                    for pair, df in data.items():
                        sigs = all_pair_signals.get(tf, {}).get(
                            pair, {}).get(strat, [])
                        if len(sigs) < 5:
                            continue
                        bar_hours = TIMEFRAMES[tf]["bar_hours"]

                        def sim_fn(df_, sigs_, params_, pair_):
                            return simulate(df_, sigs_, params_, pair_,
                                          bar_hours=bar_hours)

                        wf = walk_forward_validate(
                            df, sigs, params, pair, strat,
                            sim_fn, method="all"
                        )
                        wf_raw.extend(wf)

            if wf_raw:
                wf_summaries = aggregate_walk_forward(wf_raw)
                robust = sum(1 for w in wf_summaries
                            if w["overall_verdict"] == "ROBUST")
                marginal = sum(1 for w in wf_summaries
                              if w["overall_verdict"] == "MARGINAL")
                overfit = sum(1 for w in wf_summaries
                            if w["overall_verdict"] == "OVERFIT")
                print(f"  Walk-forward: {len(wf_summaries)} configs — "
                      f"ROBUST={robust}, MARGINAL={marginal}, "
                      f"OVERFIT={overfit}", flush=True)
        else:
            print("  No cross-pair configs found for walk-forward",
                  flush=True)
    else:
        print("\n[5/6] WALK-FORWARD: SKIPPED", flush=True)

    # ==================================================================
    # PHASE 6: Reports + Final Verdict
    # ==================================================================
    print("\n[6/6] GENERATING REPORTS...", flush=True)
    total_elapsed = time.time() - t0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 3 summary
    md = generate_phase3_report(all_results, wf_summaries, total_elapsed)
    (OUTPUT_DIR / "phase3_summary.md").write_text(md)

    # Phase 3 all results JSON (top 100 per strat)
    strat_top = defaultdict(list)
    for r in sorted(all_results, key=lambda x: x["total_pnl"], reverse=True):
        s = r["strategy"]
        if len(strat_top[s]) < 100:
            strat_top[s].append(r)

    with open(OUTPUT_DIR / "phase3_all.json", "w") as f:
        json.dump({
            "total_results": len(all_results),
            "elapsed_seconds": round(total_elapsed, 1),
            "top_per_strategy": {k: v for k, v in strat_top.items()},
        }, f, indent=2, default=str)

    # Walk-forward JSON
    if wf_summaries:
        with open(OUTPUT_DIR / "phase3_walkforward.json", "w") as f:
            json.dump({
                "summaries": wf_summaries,
                "raw": wf_raw[:500],
            }, f, indent=2, default=str)

    # FINAL VERDICT
    verdict = generate_final_verdict(all_results, wf_summaries)
    (OUTPUT_DIR / "FINAL_VERDICT.md").write_text(verdict)

    # ==================================================================
    # Console Summary
    # ==================================================================
    print(f"\n{'='*80}", flush=True)
    print("PHASE 3 RESULTS", flush=True)
    print(f"{'='*80}", flush=True)

    rankings = aggregator.rank_strategies(all_results)
    aggregator.print_summary(rankings,
                            aggregator.find_cross_pair_configs(all_results, 3),
                            aggregator.per_pair_best(all_results),
                            "Phase 3", len(all_results), total_elapsed)

    # Compare with Phase 1/2 winner
    print(f"\n{'='*80}", flush=True)
    print("CROSS-PHASE COMPARISON", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Phase 1 winner: volume_spike 1H → $231,069, PF=1.29, WR=48%",
          flush=True)

    cross = aggregator.find_cross_pair_configs(all_results, min_pairs=3)
    if cross:
        best = cross[0]
        p = best["params"]
        print(f"Phase 3 best:   {best['strategy']} → "
              f"${best['total_pnl']:+,.0f}, PF={best['avg_pf']:.2f}, "
              f"WR={best['avg_win_rate']:.1f}%", flush=True)
        print(f"  Params: SL={p['sl_pct']*100:.0f}% TP={p['tp_pct']*100:.0f}% "
              f"Trail={p['trailing_pct']*100:.0f}% Lev={p['leverage']}x "
              f"CD={p['cooldown_bars']}bars", flush=True)
        print(f"  Pairs: {best['n_pairs']}, Trades: {best['total_trades']}",
              flush=True)
    else:
        print("Phase 3 best:   No cross-pair configs found", flush=True)

    if wf_summaries:
        robust = [w for w in wf_summaries
                 if w["overall_verdict"] == "ROBUST"]
        if robust:
            print(f"\n  Best ROBUST walk-forward: {robust[0]['strategy']} "
                  f"(avg degradation {robust[0]['avg_degradation']*100:.1f}%)",
                  flush=True)

    print(f"\n{'='*80}", flush=True)
    print(f"PHASE 3 COMPLETE", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Total results: {len(all_results):,}", flush=True)
    print(f"Total time: {total_elapsed:.1f}s", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)
    print(f"  - phase3_summary.md", flush=True)
    print(f"  - phase3_all.json", flush=True)
    if wf_summaries:
        print(f"  - phase3_walkforward.json", flush=True)
    print(f"  - FINAL_VERDICT.md", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
