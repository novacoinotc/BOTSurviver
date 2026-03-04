#!/usr/bin/env python3
"""
MEGA BACKTEST PHASE 2: Advanced Optimization
=============================================

Builds on Phase 1 results (27 strategies x 19 pairs x 2 years).
Runs combo strategies, walk-forward validation, session filtering,
ensemble voting, and per-pair portfolio optimization.

Usage:
    python bot/backtest/mega/run_phase2.py                 # Full run
    python bot/backtest/mega/run_phase2.py BTCUSDT ETHUSDT # Specific pairs
    python bot/backtest/mega/run_phase2.py --skip-combos   # Skip combo sim
    python bot/backtest/mega/run_phase2.py --workers 4     # Limit cores
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
from backtest.mega.indicators_extended import compute_extended_indicators, resample_1h_to_4h
from backtest.mega.strategies_mega import (
    STRATEGY_DETECTORS, MTF_STRATEGIES, detect_all_signals
)
from backtest.mega.simulator import simulate
from backtest.mega.param_grids import get_tier1_grid, get_tier2_grid
from backtest.mega import aggregator
from backtest.mega.combo_strategies import detect_combo_signals, COMBO_DEFS
from backtest.mega.session_filter import SESSION_CONFIGS, filter_signals_by_session
from backtest.mega.ensemble import build_all_ensemble_levels, ENSEMBLE_LEVELS
from backtest.mega.walk_forward import walk_forward_validate, aggregate_walk_forward
from backtest.mega.per_pair_optimizer import (
    optimize_per_pair, simulate_portfolio, build_fine_grid
)

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


# ---------------------------------------------------------------------------
# Data loading (reuse Phase 1 pattern)
# ---------------------------------------------------------------------------

def load_pair_data(pair: str, days: int, proxy_url: str):
    """Load 1H data + indicators + 4H resampling for one pair."""
    try:
        df = download_klines(pair, "1h", days=days, proxy_url=proxy_url)
        if df is None or len(df) < 100:
            return pair, None, None
        df = compute_all_indicators(df)
        df = compute_extended_indicators(df)
        df_4h = resample_1h_to_4h(df)
        if df_4h is not None and len(df_4h) > 50:
            df_4h = compute_all_indicators(df_4h)
        else:
            df_4h = None
        return pair, df, df_4h
    except Exception as e:
        print(f"  ERROR loading {pair}: {e}", flush=True)
        return pair, None, None


# ---------------------------------------------------------------------------
# Simulation job (for multiprocessing)
# ---------------------------------------------------------------------------

def _sim_job(args):
    """Worker function for parallel simulation."""
    pair, strategy, params, df_dict, signals = args
    if not signals:
        return None
    df = pd.DataFrame(df_dict)
    result = simulate(df, signals, params, pair)
    if result and result["total_trades"] >= 5:
        result["pair"] = pair
        result["strategy"] = strategy
        result["params"] = {k: v for k, v in params.items()}
        return result
    return None


def _prepare_df_dict(df: pd.DataFrame) -> dict:
    """Extract minimal columns for simulation as picklable dict."""
    cols = ["high", "low", "close"]
    if "timestamp" in df.columns:
        cols.append("timestamp")
    return {c: df[c].values for c in cols if c in df.columns}


def _run_parallel(jobs: list, n_workers: int, label: str) -> list[dict]:
    """Run simulation jobs in parallel with progress reporting."""
    if not jobs:
        print(f"  {label}: 0 jobs, skipping", flush=True)
        return []

    print(f"  {label}: {len(jobs):,} simulations across {n_workers} cores", flush=True)
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
                print(f"    {done:,}/{len(jobs):,} ({done/len(jobs)*100:.0f}%) | "
                      f"{speed:,.0f} sims/s | ETA: {eta:.0f}s | "
                      f"qualified: {len(results):,}", flush=True)

    elapsed = time.time() - t0
    print(f"  {label} done: {len(results):,} qualified from "
          f"{len(jobs):,} ({elapsed:.1f}s, "
          f"{len(jobs)/max(elapsed,0.1):,.0f} sims/s)", flush=True)
    return results


# ---------------------------------------------------------------------------
# Phase 2a: Combo strategies
# ---------------------------------------------------------------------------

def run_combos(data_1h, pair_signals, n_workers):
    """Detect combo signals and simulate with Tier1 + Tier2 grids."""
    print(f"\n{'='*70}", flush=True)
    print("PHASE 2a: COMBO STRATEGIES (18 combos)", flush=True)
    print(f"{'='*70}", flush=True)
    t0 = time.time()

    tier1_grid = get_tier1_grid()
    tier2_grid = get_tier2_grid()

    # Detect combo signals for all pairs
    combo_signals = {}  # {pair: {combo_name: [signals]}}
    for pair in data_1h:
        df = data_1h[pair]
        base_sigs = pair_signals[pair]
        combos = detect_combo_signals(base_sigs, df)
        combo_signals[pair] = combos

        active = [f"{c}:{len(v)}" for c, v in combos.items() if v]
        total = sum(len(v) for v in combos.values())
        print(f"  {pair}: {total} combo signals ({len(active)} active combos)", flush=True)

    # Tier 1: all 18 combos
    jobs = []
    for pair in data_1h:
        df_dict = _prepare_df_dict(data_1h[pair])
        for combo_name, sigs in combo_signals[pair].items():
            if not sigs:
                continue
            for params in tier1_grid:
                jobs.append((pair, combo_name, params, df_dict, sigs))

    combo_t1_results = _run_parallel(jobs, n_workers, "Combo Tier1")

    # Select top 5 combos for Tier 2
    combo_rankings = aggregator.rank_strategies(combo_t1_results)
    top_combos = aggregator.select_top_strategies(combo_rankings, 5)
    print(f"\n  Top 5 combos for Tier2: {', '.join(top_combos)}", flush=True)

    # Tier 2: expanded grid on top 5 combos
    jobs_t2 = []
    for pair in data_1h:
        df_dict = _prepare_df_dict(data_1h[pair])
        for combo_name in top_combos:
            sigs = combo_signals[pair].get(combo_name, [])
            if not sigs:
                continue
            for params in tier2_grid:
                jobs_t2.append((pair, combo_name, params, df_dict, sigs))

    combo_t2_results = _run_parallel(jobs_t2, n_workers, "Combo Tier2")

    all_combo = combo_t1_results + combo_t2_results
    elapsed = time.time() - t0

    # Print summary
    rankings = aggregator.rank_strategies(all_combo)
    print(f"\n  --- COMBO RANKING (top 10) ---", flush=True)
    for i, r in enumerate(rankings[:10], 1):
        print(f"  {i:2d}. {r['strategy']:25} | ProfRate={r['profit_rate']:.1f}% | "
              f"Pairs={r['pairs_profitable']}/{r['pairs_total']} | "
              f"BestPnL=${r['best_pnl']:+,.0f} | AvgPF={r['avg_pf']:.2f} | "
              f"WR={r['avg_win_rate']:.1f}%", flush=True)

    return all_combo, combo_signals, elapsed


# ---------------------------------------------------------------------------
# Phase 2b: Walk-forward validation
# ---------------------------------------------------------------------------

def run_walk_forward(data_1h, pair_signals, combo_signals,
                     phase1_results, combo_results):
    """Walk-forward validate top 15 configs (Phase 1 + combos)."""
    print(f"\n{'='*70}", flush=True)
    print("PHASE 2b: WALK-FORWARD VALIDATION", flush=True)
    print(f"{'='*70}", flush=True)
    t0 = time.time()

    # Select top 15 configs to validate
    all_results = phase1_results + combo_results
    cross_configs = aggregator.find_cross_pair_configs(all_results, min_pairs=3)

    # Take top 15 unique strategy+params combinations
    configs_to_test = []
    seen = set()
    for cc in cross_configs[:30]:
        key = (cc["strategy"], json.dumps(cc["params"], sort_keys=True))
        if key not in seen:
            seen.add(key)
            configs_to_test.append({
                "strategy": cc["strategy"],
                "params": cc["params"],
            })
        if len(configs_to_test) >= 15:
            break

    print(f"  Testing {len(configs_to_test)} configs across {len(data_1h)} pairs", flush=True)

    wf_results = []
    for cfg_idx, cfg in enumerate(configs_to_test, 1):
        strat = cfg["strategy"]
        params = cfg["params"]
        print(f"  [{cfg_idx}/{len(configs_to_test)}] {strat} "
              f"(SL={params['sl_pct']*100:.0f}% TP={params['tp_pct']*100:.0f}% "
              f"Lev={params['leverage']}x)...", end=" ", flush=True)

        n_windows = 0
        for pair in data_1h:
            df = data_1h[pair]
            # Get signals — check combos first, then base strategies
            if strat in COMBO_DEFS and combo_signals:
                sigs = combo_signals.get(pair, {}).get(strat, [])
            else:
                sigs = pair_signals.get(pair, {}).get(strat, [])

            if not sigs:
                continue

            results = walk_forward_validate(
                df, sigs, params, pair, strat, simulate, method="all"
            )
            wf_results.extend(results)
            n_windows += len(results)

        print(f"{n_windows} windows", flush=True)

    elapsed = time.time() - t0

    # Aggregate
    wf_summary = aggregate_walk_forward(wf_results)

    n_robust = sum(1 for s in wf_summary if s["overall_verdict"] == "ROBUST")
    n_marginal = sum(1 for s in wf_summary if s["overall_verdict"] == "MARGINAL")
    n_overfit = sum(1 for s in wf_summary if s["overall_verdict"] == "OVERFIT")

    print(f"\n  --- WALK-FORWARD SUMMARY ({elapsed:.1f}s) ---", flush=True)
    print(f"  ROBUST: {n_robust} | MARGINAL: {n_marginal} | OVERFIT: {n_overfit}", flush=True)
    print(f"\n  {'#':>3} {'Strategy':25} {'Verdict':10} {'AvgDeg':>7} "
          f"{'MaxDeg':>7} {'TrainPF':>8} {'TestPF':>8} {'Windows':>4}", flush=True)
    print(f"  {'-'*75}", flush=True)
    for i, s in enumerate(wf_summary[:15], 1):
        v_color = {"ROBUST": "+", "MARGINAL": "~", "OVERFIT": "-"}[s["overall_verdict"]]
        print(f"  {i:3d} {s['strategy']:25} [{v_color}]{s['overall_verdict']:8} "
              f"{s['avg_degradation']:6.1%} {s['max_degradation']:6.1%} "
              f"{s['avg_train_pf']:7.3f} {s['avg_test_pf']:7.3f} "
              f"{s['n_windows']:4d}", flush=True)

    return wf_results, wf_summary, elapsed


# ---------------------------------------------------------------------------
# Phase 2c: Session filter
# ---------------------------------------------------------------------------

def run_session_filter(data_1h, pair_signals, phase1_results):
    """Test session filtering on top 5 strategies."""
    print(f"\n{'='*70}", flush=True)
    print("PHASE 2c: SESSION FILTER ANALYSIS", flush=True)
    print(f"{'='*70}", flush=True)
    t0 = time.time()

    # Get top 5 strategies with their best params
    rankings = aggregator.rank_strategies(phase1_results)
    cross_configs = aggregator.find_cross_pair_configs(phase1_results, min_pairs=3)

    # For each top 5 strategy, find its best cross-pair config
    top5 = aggregator.select_top_strategies(rankings, 5)
    strat_best_params = {}
    for strat in top5:
        for cc in cross_configs:
            if cc["strategy"] == strat:
                strat_best_params[strat] = cc["params"]
                break
        if strat not in strat_best_params:
            # Fallback: use any profitable config
            for r in phase1_results:
                if r["strategy"] == strat and r.get("profit_factor", 0) > 1.0:
                    strat_best_params[strat] = r["params"]
                    break

    print(f"  Testing {len(strat_best_params)} strategies x "
          f"{len(SESSION_CONFIGS)} sessions x {len(data_1h)} pairs", flush=True)

    session_results = []
    for strat, params in strat_best_params.items():
        print(f"  {strat}:", end=" ", flush=True)
        for pair in data_1h:
            df = data_1h[pair]
            sigs = pair_signals.get(pair, {}).get(strat, [])
            if not sigs:
                continue

            for session_name in SESSION_CONFIGS:
                filtered = filter_signals_by_session(sigs, df, session_name)
                if not filtered:
                    continue

                result = simulate(df, filtered, params, pair)
                if result and result["total_trades"] >= 3:
                    result["pair"] = pair
                    result["strategy"] = strat
                    result["session"] = session_name
                    result["params"] = {k: v for k, v in params.items()}
                    session_results.append(result)
        print(f"{sum(1 for r in session_results if r['strategy']==strat)} results", flush=True)

    elapsed = time.time() - t0

    # Analyze: best session per strategy
    print(f"\n  --- SESSION ANALYSIS ({elapsed:.1f}s) ---", flush=True)
    for strat in strat_best_params:
        strat_res = [r for r in session_results if r["strategy"] == strat]
        if not strat_res:
            continue

        print(f"\n  {strat}:", flush=True)
        session_agg = defaultdict(lambda: {"pnl": 0, "trades": 0, "pairs": 0, "pf_sum": 0})
        for r in strat_res:
            sa = session_agg[r["session"]]
            sa["pnl"] += r["total_pnl"]
            sa["trades"] += r["total_trades"]
            sa["pairs"] += 1
            sa["pf_sum"] += r["profit_factor"]

        for sess in SESSION_CONFIGS:
            sa = session_agg[sess]
            if sa["pairs"] == 0:
                continue
            avg_pf = sa["pf_sum"] / sa["pairs"]
            print(f"    {sess:15} | PnL=${sa['pnl']:+10,.0f} | "
                  f"Trades={sa['trades']:4d} | "
                  f"Pairs={sa['pairs']:2d} | AvgPF={avg_pf:.3f}", flush=True)

    return session_results, elapsed


# ---------------------------------------------------------------------------
# Phase 2d: Ensemble voting
# ---------------------------------------------------------------------------

def run_ensemble(data_1h, pair_signals, n_workers):
    """Build ensemble signals and simulate with Tier1 grid."""
    print(f"\n{'='*70}", flush=True)
    print("PHASE 2d: ENSEMBLE VOTING", flush=True)
    print(f"{'='*70}", flush=True)
    t0 = time.time()

    tier1_grid = get_tier1_grid()

    # Build ensemble signals for all pairs
    ensemble_all = {}  # {pair: {ensemble_N: [signals]}}
    for pair in data_1h:
        base_sigs = pair_signals[pair]
        ens = build_all_ensemble_levels(base_sigs)
        ensemble_all[pair] = ens

        counts = {k: len(v) for k, v in ens.items() if v}
        print(f"  {pair}: {counts}", flush=True)

    # Build simulation jobs
    jobs = []
    for pair in data_1h:
        df_dict = _prepare_df_dict(data_1h[pair])
        for ens_name, sigs in ensemble_all[pair].items():
            if not sigs:
                continue
            for params in tier1_grid:
                jobs.append((pair, ens_name, params, df_dict, sigs))

    ensemble_results = _run_parallel(jobs, n_workers, "Ensemble")
    elapsed = time.time() - t0

    # Print summary
    rankings = aggregator.rank_strategies(ensemble_results)
    print(f"\n  --- ENSEMBLE RANKING ---", flush=True)
    for i, r in enumerate(rankings, 1):
        print(f"  {i}. {r['strategy']:15} | ProfRate={r['profit_rate']:.1f}% | "
              f"Pairs={r['pairs_profitable']}/{r['pairs_total']} | "
              f"BestPnL=${r['best_pnl']:+,.0f} | AvgPF={r['avg_pf']:.2f} | "
              f"WR={r['avg_win_rate']:.1f}% | AvgTrades={r['avg_trades']:.0f}", flush=True)

    return ensemble_results, elapsed


# ---------------------------------------------------------------------------
# Phase 2e: Per-pair portfolio optimization
# ---------------------------------------------------------------------------

def run_per_pair_portfolio(data_1h, pair_signals, combo_signals,
                           phase1_results, combo_results):
    """Find best config per pair, fine-tune, build portfolio."""
    print(f"\n{'='*70}", flush=True)
    print("PHASE 2e: PER-PAIR PORTFOLIO OPTIMIZATION", flush=True)
    print(f"{'='*70}", flush=True)
    t0 = time.time()

    # Combine all results to find per-pair best
    all_results = phase1_results + combo_results
    pair_bests = aggregator.per_pair_best(all_results)

    print(f"  Phase 1 per-pair bests found: {len(pair_bests)}", flush=True)

    # Fine-tune each pair
    optimized = {}
    for pair in sorted(data_1h.keys()):
        if pair not in pair_bests:
            print(f"  {pair}: no Phase 1 results, skipping", flush=True)
            continue

        best = pair_bests[pair]
        strat = best["strategy"]
        best_params = best["params"]

        # Get the right signals
        if strat in COMBO_DEFS and combo_signals:
            sigs = combo_signals.get(pair, {}).get(strat, [])
        else:
            sigs = pair_signals.get(pair, {}).get(strat, [])

        if not sigs:
            print(f"  {pair}: no signals for {strat}, using Phase 1 best", flush=True)
            optimized[pair] = best
            continue

        opt = optimize_per_pair(pair, strat, sigs, data_1h[pair],
                                best_params, simulate)

        if opt["best_result"]:
            improvement = opt["best_result"]["total_pnl"] - best["total_pnl"]
            optimized[pair] = opt["best_result"]
            print(f"  {pair:10} {strat:25} | Grid={opt['grid_size']:3d} | "
                  f"Prof={opt['n_profitable']:3d} | "
                  f"PnL: ${best['total_pnl']:+,.0f} -> "
                  f"${opt['best_result']['total_pnl']:+,.0f} "
                  f"({'+' if improvement>=0 else ''}{improvement:,.0f})", flush=True)
        else:
            optimized[pair] = best
            print(f"  {pair:10} {strat:25} | no improvement found", flush=True)

    # Build portfolio
    portfolio = simulate_portfolio(optimized, n_pairs=len(data_1h))

    elapsed = time.time() - t0

    # Also build single-config portfolio for comparison
    best_single_cross = aggregator.find_cross_pair_configs(all_results, min_pairs=5)
    single_config_pnl = 0
    if best_single_cross:
        bc = best_single_cross[0]
        single_config_pnl = bc["total_pnl"]

    print(f"\n  --- PORTFOLIO SUMMARY ({elapsed:.1f}s) ---", flush=True)
    print(f"  Portfolio PnL:      ${portfolio['total_pnl']:+,.2f} "
          f"({portfolio['return_pct']:+.1f}%)", flush=True)
    print(f"  Trades:             {portfolio['total_trades']}", flush=True)
    print(f"  Win Rate:           {portfolio['win_rate']:.1f}%", flush=True)
    print(f"  Profitable Pairs:   {portfolio['n_profitable']}/{portfolio['n_pairs']}", flush=True)
    print(f"  Green Months:       {portfolio['green_months']}/{portfolio['total_months']}", flush=True)
    if single_config_pnl:
        print(f"  Best Single Config: ${single_config_pnl:+,.0f} "
              f"(portfolio {'beats' if portfolio['total_pnl'] > single_config_pnl else 'loses to'} "
              f"single by ${abs(portfolio['total_pnl'] - single_config_pnl):,.0f})", flush=True)

    print(f"\n  Per-pair breakdown:", flush=True)
    for pd_item in portfolio["pair_details"]:
        status = "+" if pd_item["pnl_scaled"] > 0 else "-"
        print(f"    [{status}] {pd_item['pair']:10} {pd_item['strategy']:25} "
              f"PnL=${pd_item['pnl_scaled']:+8,.0f} (full=${pd_item['pnl_full']:+10,.0f}) "
              f"PF={pd_item['pf']:.2f} WR={pd_item['win_rate']:.0f}%", flush=True)

    return optimized, portfolio, elapsed


# ---------------------------------------------------------------------------
# Phase 2f: Final report
# ---------------------------------------------------------------------------

def generate_phase2_report(combo_results, wf_summary, session_results,
                           ensemble_results, portfolio, phase1_results,
                           timings):
    """Generate comprehensive Phase 2 comparison report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# MEGA BACKTEST PHASE 2 — Advanced Optimization Report")
    lines.append("")
    lines.append(f"**Total runtime:** {sum(timings.values()):.1f}s")
    lines.append("")

    # Section 1: Combo strategies
    lines.append("## 1. Combo Strategies (18 combos)")
    lines.append("")
    if combo_results:
        combo_rank = aggregator.rank_strategies(combo_results)
        lines.append("| # | Combo | ProfRate | Pairs | BestPnL | AvgPF | WR |")
        lines.append("|---|-------|---------|-------|---------|-------|-----|")
        for i, r in enumerate(combo_rank[:18], 1):
            lines.append(f"| {i} | {r['strategy']} | {r['profit_rate']:.1f}% | "
                         f"{r['pairs_profitable']}/{r['pairs_total']} | "
                         f"${r['best_pnl']:+,.0f} | {r['avg_pf']:.2f} | "
                         f"{r['avg_win_rate']:.1f}% |")

        # Compare best combo vs Phase 1 winner
        if combo_rank:
            best_combo = combo_rank[0]
            p1_rank = aggregator.rank_strategies(phase1_results)
            if p1_rank:
                lines.append(f"\n**Best combo:** {best_combo['strategy']} "
                             f"(BestPnL=${best_combo['best_pnl']:+,.0f}) vs "
                             f"**Phase 1 winner:** {p1_rank[0]['strategy']} "
                             f"(BestPnL=${p1_rank[0]['best_pnl']:+,.0f})")
    lines.append("")

    # Section 2: Walk-forward
    lines.append("## 2. Walk-Forward Validation")
    lines.append("")
    if wf_summary:
        n_r = sum(1 for s in wf_summary if s["overall_verdict"] == "ROBUST")
        n_m = sum(1 for s in wf_summary if s["overall_verdict"] == "MARGINAL")
        n_o = sum(1 for s in wf_summary if s["overall_verdict"] == "OVERFIT")
        lines.append(f"**Verdicts:** ROBUST={n_r} | MARGINAL={n_m} | OVERFIT={n_o}")
        lines.append("")
        lines.append("| # | Strategy | Verdict | AvgDeg | TrainPF | TestPF | Windows |")
        lines.append("|---|----------|---------|--------|---------|--------|---------|")
        for i, s in enumerate(wf_summary[:15], 1):
            lines.append(f"| {i} | {s['strategy']} | {s['overall_verdict']} | "
                         f"{s['avg_degradation']:.1%} | {s['avg_train_pf']:.3f} | "
                         f"{s['avg_test_pf']:.3f} | {s['n_windows']} |")
    lines.append("")

    # Section 3: Session filter
    lines.append("## 3. Session Filter Analysis")
    lines.append("")
    if session_results:
        sess_agg = defaultdict(lambda: {"pnl": 0, "trades": 0, "n": 0, "pf_sum": 0})
        for r in session_results:
            sa = sess_agg[r["session"]]
            sa["pnl"] += r["total_pnl"]
            sa["trades"] += r["total_trades"]
            sa["n"] += 1
            sa["pf_sum"] += r["profit_factor"]

        lines.append("| Session | TotalPnL | Trades | Configs | AvgPF |")
        lines.append("|---------|----------|--------|---------|-------|")
        for sess in SESSION_CONFIGS:
            sa = sess_agg[sess]
            if sa["n"] == 0:
                continue
            avg_pf = sa["pf_sum"] / sa["n"]
            lines.append(f"| {sess} | ${sa['pnl']:+,.0f} | {sa['trades']} | "
                         f"{sa['n']} | {avg_pf:.3f} |")
    lines.append("")

    # Section 4: Ensemble
    lines.append("## 4. Ensemble Voting")
    lines.append("")
    if ensemble_results:
        ens_rank = aggregator.rank_strategies(ensemble_results)
        lines.append("| Level | ProfRate | Pairs | BestPnL | AvgPF | WR | AvgTrades |")
        lines.append("|-------|---------|-------|---------|-------|-----|-----------|")
        for r in ens_rank:
            lines.append(f"| {r['strategy']} | {r['profit_rate']:.1f}% | "
                         f"{r['pairs_profitable']}/{r['pairs_total']} | "
                         f"${r['best_pnl']:+,.0f} | {r['avg_pf']:.2f} | "
                         f"{r['avg_win_rate']:.1f}% | {r['avg_trades']:.0f} |")
    lines.append("")

    # Section 5: Portfolio
    lines.append("## 5. Per-Pair Portfolio")
    lines.append("")
    if portfolio:
        lines.append(f"- **Total PnL:** ${portfolio['total_pnl']:+,.2f} "
                     f"({portfolio['return_pct']:+.1f}%)")
        lines.append(f"- **Trades:** {portfolio['total_trades']}")
        lines.append(f"- **Win Rate:** {portfolio['win_rate']:.1f}%")
        lines.append(f"- **Profitable Pairs:** {portfolio['n_profitable']}/{portfolio['n_pairs']}")
        lines.append(f"- **Green Months:** {portfolio['green_months']}/{portfolio['total_months']}")
        lines.append("")
        lines.append("| Pair | Strategy | PnL (scaled) | PF | WR | MaxDD |")
        lines.append("|------|----------|-------------|-----|-----|-------|")
        for pd_item in portfolio.get("pair_details", []):
            lines.append(f"| {pd_item['pair']} | {pd_item['strategy']} | "
                         f"${pd_item['pnl_scaled']:+,.0f} | {pd_item['pf']:.2f} | "
                         f"{pd_item['win_rate']:.1f}% | {pd_item['max_dd']:.1f}% |")
    lines.append("")

    # Section 6: Recommendations
    lines.append("## 6. Production Recommendations")
    lines.append("")
    lines.append("Based on Phase 2 analysis:")
    lines.append("")

    # Best robust config
    if wf_summary:
        robust = [s for s in wf_summary if s["overall_verdict"] == "ROBUST"]
        if robust:
            best = robust[0]
            lines.append(f"1. **Most robust strategy:** {best['strategy']} "
                         f"(AvgDeg={best['avg_degradation']:.1%}, "
                         f"TestPF={best['avg_test_pf']:.3f})")
            p = best["params"]
            lines.append(f"   - Params: SL={p['sl_pct']*100:.0f}% "
                         f"TP={p['tp_pct']*100:.0f}% "
                         f"Trail={p['trailing_pct']*100:.0f}% "
                         f"Lev={p['leverage']}x "
                         f"CD={p['cooldown_bars']}h")

    if combo_results:
        combo_rank = aggregator.rank_strategies(combo_results)
        if combo_rank:
            lines.append(f"2. **Best combo:** {combo_rank[0]['strategy']} "
                         f"(BestPnL=${combo_rank[0]['best_pnl']:+,.0f}, "
                         f"ProfRate={combo_rank[0]['profit_rate']:.1f}%)")

    if ensemble_results:
        ens_rank = aggregator.rank_strategies(ensemble_results)
        profitable_ens = [r for r in ens_rank if r["avg_pf"] > 1.0]
        if profitable_ens:
            lines.append(f"3. **Best ensemble:** {profitable_ens[0]['strategy']} "
                         f"(AvgPF={profitable_ens[0]['avg_pf']:.2f}, "
                         f"WR={profitable_ens[0]['avg_win_rate']:.1f}%)")

    report = "\n".join(lines)
    report_path = OUTPUT_DIR / "phase2_summary.md"
    report_path.write_text(report)
    print(f"\n  Report saved to: {report_path}", flush=True)
    return report_path


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mega Backtest Phase 2")
    parser.add_argument("pairs", nargs="*", default=ALL_PAIRS)
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--skip-combos", action="store_true")
    parser.add_argument("--skip-walkforward", action="store_true")
    parser.add_argument("--skip-sessions", action="store_true")
    parser.add_argument("--skip-ensemble", action="store_true")
    parser.add_argument("--skip-portfolio", action="store_true")
    args = parser.parse_args()

    pairs = args.pairs
    days = args.days
    n_workers = args.workers or mp.cpu_count()
    t_total = time.time()

    print("=" * 80, flush=True)
    print("MEGA BACKTEST PHASE 2 — Advanced Optimization", flush=True)
    print(f"$50K | {len(pairs)} pairs | {days} days | {n_workers} cores", flush=True)
    print("=" * 80, flush=True)

    # ==================================================================
    # PHASE 0: Load data + compute indicators
    # ==================================================================
    print("\nPHASE 0: Loading data + indicators...", flush=True)
    t0 = time.time()

    proxy_url = load_proxy_url()
    data_1h = {}
    data_4h = {}

    for pair in pairs:
        print(f"  {pair}...", end=" ", flush=True)
        _, df, df_4h = load_pair_data(pair, days, proxy_url)
        if df is not None:
            data_1h[pair] = df
            if df_4h is not None:
                data_4h[pair] = df_4h
            print(f"{len(df)} 1H bars", flush=True)
        else:
            print("FAILED", flush=True)
        time.sleep(0.5)

    print(f"  Loaded: {len(data_1h)} pairs ({time.time()-t0:.1f}s)", flush=True)
    if not data_1h:
        print("ERROR: No data loaded.", flush=True)
        return

    # ==================================================================
    # PHASE 1: Detect base signals (27 strategies)
    # ==================================================================
    print(f"\nPHASE 1: Detecting base signals...", flush=True)
    t1 = time.time()

    pair_signals = {}
    for pair in data_1h:
        df = data_1h[pair]
        df_4h = data_4h.get(pair)
        strat_sigs = detect_all_signals(df, df_4h)
        pair_signals[pair] = strat_sigs

        total = sum(len(v) for v in strat_sigs.values())
        print(f"  {pair}: {total} signals", flush=True)

    print(f"  Signal detection: {time.time()-t1:.1f}s", flush=True)

    # Run Phase 1 simulation (Tier1 only for baseline)
    print(f"\nPHASE 1 Simulation (baseline)...", flush=True)
    tier1_grid = get_tier1_grid()
    jobs = []
    for pair in data_1h:
        df_dict = _prepare_df_dict(data_1h[pair])
        for strat, sigs in pair_signals[pair].items():
            if not sigs:
                continue
            for params in tier1_grid:
                jobs.append((pair, strat, params, df_dict, sigs))

    phase1_results = _run_parallel(jobs, n_workers, "Phase1 Tier1")

    # ==================================================================
    # PHASE 2a-2e: Advanced modules
    # ==================================================================
    timings = {}
    combo_results = []
    combo_signals = {}
    wf_summary = []
    session_results = []
    ensemble_results = []
    portfolio = {}
    optimized = {}

    # 2a: Combos
    if not args.skip_combos:
        combo_results, combo_signals, t_combo = run_combos(
            data_1h, pair_signals, n_workers)
        timings["combos"] = t_combo
    else:
        print("\n  Skipping combos", flush=True)

    # 2b: Walk-forward
    if not args.skip_walkforward:
        wf_results_raw, wf_summary, t_wf = run_walk_forward(
            data_1h, pair_signals, combo_signals,
            phase1_results, combo_results)
        timings["walkforward"] = t_wf
    else:
        print("\n  Skipping walk-forward", flush=True)

    # 2c: Session filter
    if not args.skip_sessions:
        session_results, t_sess = run_session_filter(
            data_1h, pair_signals, phase1_results)
        timings["sessions"] = t_sess
    else:
        print("\n  Skipping sessions", flush=True)

    # 2d: Ensemble
    if not args.skip_ensemble:
        ensemble_results, t_ens = run_ensemble(
            data_1h, pair_signals, n_workers)
        timings["ensemble"] = t_ens
    else:
        print("\n  Skipping ensemble", flush=True)

    # 2e: Per-pair portfolio
    if not args.skip_portfolio:
        optimized, portfolio, t_port = run_per_pair_portfolio(
            data_1h, pair_signals, combo_signals,
            phase1_results, combo_results)
        timings["portfolio"] = t_port
    else:
        print("\n  Skipping portfolio", flush=True)

    # ==================================================================
    # PHASE 2f: Save all results + final report
    # ==================================================================
    print(f"\n{'='*70}", flush=True)
    print("PHASE 2f: SAVING RESULTS + FINAL REPORT", flush=True)
    print(f"{'='*70}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save individual JSON results
    if combo_results:
        with open(OUTPUT_DIR / "phase2_combos.json", "w") as f:
            json.dump(combo_results[:500], f, indent=2, default=str)

    if wf_summary:
        with open(OUTPUT_DIR / "phase2_walkforward.json", "w") as f:
            json.dump(wf_summary, f, indent=2, default=str)

    if session_results:
        with open(OUTPUT_DIR / "phase2_sessions.json", "w") as f:
            json.dump(session_results[:200], f, indent=2, default=str)

    if ensemble_results:
        with open(OUTPUT_DIR / "phase2_ensemble.json", "w") as f:
            json.dump(ensemble_results[:200], f, indent=2, default=str)

    if portfolio:
        with open(OUTPUT_DIR / "phase2_portfolio.json", "w") as f:
            json.dump(portfolio, f, indent=2, default=str)

    # Generate summary report
    total_elapsed = time.time() - t_total
    timings["total"] = total_elapsed

    report_path = generate_phase2_report(
        combo_results, wf_summary, session_results,
        ensemble_results, portfolio, phase1_results,
        timings
    )

    # Final summary
    print(f"\n{'='*80}", flush=True)
    print("MEGA BACKTEST PHASE 2 COMPLETE", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Total time: {total_elapsed:.1f}s", flush=True)
    for module, t in sorted(timings.items()):
        if module != "total":
            print(f"  {module:15}: {t:.1f}s", flush=True)
    print(f"\nResults saved to: {OUTPUT_DIR}", flush=True)
    print(f"Report: {report_path}", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
