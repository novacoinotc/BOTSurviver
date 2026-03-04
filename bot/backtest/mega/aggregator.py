"""Result aggregation and report generation for mega backtest.

Produces:
1. Strategy ranking (cross-pair performance)
2. Per-pair best strategies
3. Cross-pair configs (profitable on 5+ pairs)
4. Markdown summary report
5. JSON exports
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "results"


def rank_strategies(all_results: list[dict]) -> list[dict]:
    """Rank strategies by aggregate performance across all pairs.

    Returns sorted list of strategy summaries.
    """
    strat_data = defaultdict(list)
    for r in all_results:
        strat_data[r["strategy"]].append(r)

    rankings = []
    for strat, results in strat_data.items():
        profitable = [r for r in results if r["profit_factor"] > 1.0]
        pnls = [r["total_pnl"] for r in results]
        pairs_profitable = len(set(r["pair"] for r in profitable))
        pairs_total = len(set(r["pair"] for r in results))

        rankings.append({
            "strategy": strat,
            "total_configs": len(results),
            "profitable_configs": len(profitable),
            "profit_rate": round(len(profitable) / max(len(results), 1) * 100, 1),
            "pairs_profitable": pairs_profitable,
            "pairs_total": pairs_total,
            "best_pnl": round(max(pnls), 2) if pnls else 0,
            "median_pnl": round(float(np.median(pnls)), 2) if pnls else 0,
            "avg_pnl": round(float(np.mean(pnls)), 2) if pnls else 0,
            "avg_pf": round(float(np.mean([r["profit_factor"] for r in results])), 4),
            "avg_win_rate": round(float(np.mean([r["win_rate"] for r in results])) * 100, 1),
            "avg_trades": round(float(np.mean([r["total_trades"] for r in results])), 0),
        })

    rankings.sort(key=lambda x: (x["pairs_profitable"], x["median_pnl"]), reverse=True)
    return rankings


def find_cross_pair_configs(all_results: list[dict], min_pairs: int = 5) -> list[dict]:
    """Find parameter configs that are profitable on multiple pairs.

    Returns configs sorted by (n_pairs, total_pnl).
    """
    config_map = defaultdict(list)
    for r in all_results:
        if r["profit_factor"] > 1.0:
            key = (r["strategy"], json.dumps(r["params"], sort_keys=True))
            config_map[key].append(r)

    cross_configs = []
    for (strat, ps), entries in config_map.items():
        pairs_ok = set(e["pair"] for e in entries)
        if len(pairs_ok) < min_pairs:
            continue

        total_pnl = sum(e["total_pnl"] for e in entries)
        avg_pf = float(np.mean([e["profit_factor"] for e in entries]))
        avg_wr = float(np.mean([e["win_rate"] for e in entries])) * 100
        worst_dd = min(e["max_drawdown_pct"] for e in entries)

        # Monthly aggregation
        monthly_agg = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
        for e in entries:
            for m in e.get("monthly", []):
                monthly_agg[m["month"]]["pnl"] += m["pnl"]
                monthly_agg[m["month"]]["trades"] += m["trades"]
                monthly_agg[m["month"]]["wins"] += m["wins"]

        green_months = sum(1 for m in monthly_agg.values() if m["pnl"] > 0)

        cross_configs.append({
            "strategy": strat,
            "params": json.loads(ps),
            "n_pairs": len(pairs_ok),
            "pairs": sorted(pairs_ok),
            "total_pnl": round(total_pnl, 2),
            "return_pct": round(total_pnl / 50_000 * 100, 2),
            "avg_pf": round(avg_pf, 4),
            "avg_win_rate": round(avg_wr, 1),
            "worst_dd": round(worst_dd, 4),
            "total_trades": sum(e["total_trades"] for e in entries),
            "green_months": green_months,
            "total_months": len(monthly_agg),
        })

    cross_configs.sort(key=lambda x: (x["n_pairs"], x["total_pnl"]), reverse=True)
    return cross_configs


def per_pair_best(all_results: list[dict]) -> dict[str, dict]:
    """Find the best config for each pair."""
    pair_best = {}
    for r in all_results:
        p = r["pair"]
        if p not in pair_best or r["total_pnl"] > pair_best[p]["total_pnl"]:
            pair_best[p] = r
    return pair_best


def select_top_strategies(rankings: list[dict], top_n: int = 10) -> list[str]:
    """Select top N strategies for Tier 2 based on ranking."""
    return [r["strategy"] for r in rankings[:top_n]]


def generate_summary_md(rankings: list[dict], cross_configs: list[dict],
                        pair_bests: dict, tier: str, elapsed: float) -> str:
    """Generate a markdown summary report."""
    lines = []
    lines.append(f"# Mega Backtest Results — {tier}")
    lines.append(f"")
    lines.append(f"**Runtime:** {elapsed:.1f}s | **Initial equity:** $50,000")
    lines.append(f"")

    # Strategy ranking
    lines.append("## Strategy Ranking")
    lines.append("")
    lines.append("| # | Strategy | Profit Rate | Pairs | Best PnL | Median PnL | Avg PF | Avg WR | Avg Trades |")
    lines.append("|---|----------|------------|-------|----------|-----------|--------|--------|-----------|")
    for i, r in enumerate(rankings[:27], 1):
        lines.append(
            f"| {i} | {r['strategy']} | {r['profit_rate']}% | "
            f"{r['pairs_profitable']}/{r['pairs_total']} | "
            f"${r['best_pnl']:+,.0f} | ${r['median_pnl']:+,.0f} | "
            f"{r['avg_pf']:.2f} | {r['avg_win_rate']:.1f}% | {r['avg_trades']:.0f} |"
        )

    # Cross-pair configs
    if cross_configs:
        lines.append("")
        lines.append("## Cross-Pair Configs (profitable on 5+ pairs)")
        lines.append("")
        lines.append("| # | Strategy | Pairs | Total PnL | Return% | Avg PF | WR | Trades | Green Mo |")
        lines.append("|---|----------|-------|-----------|---------|--------|-----|--------|---------|")
        for i, c in enumerate(cross_configs[:20], 1):
            p = c["params"]
            lines.append(
                f"| {i} | {c['strategy']} | {c['n_pairs']} | "
                f"${c['total_pnl']:+,.0f} | {c['return_pct']:+.1f}% | "
                f"{c['avg_pf']:.2f} | {c['avg_win_rate']:.1f}% | "
                f"{c['total_trades']} | {c['green_months']}/{c['total_months']} |"
            )
            lines.append(
                f"|   | | | SL={p['sl_pct']*100:.0f}% TP={p['tp_pct']*100:.0f}% "
                f"Trail={p['trailing_pct']*100:.0f}% Lev={p['leverage']}x "
                f"CD={p['cooldown_bars']}h Pos={p['position_pct']*100:.0f}% | | | | | |"
            )

    # Per-pair best
    lines.append("")
    lines.append("## Per-Pair Best Config")
    lines.append("")
    lines.append("| Pair | Strategy | PnL | Return% | PF | Trades | DD |")
    lines.append("|------|----------|-----|---------|-----|--------|-----|")
    for pair in sorted(pair_bests.keys()):
        r = pair_bests[pair]
        status = "+" if r["profit_factor"] > 1.0 else "-"
        lines.append(
            f"| [{status}] {pair} | {r['strategy']} | "
            f"${r['total_pnl']:+,.0f} | {r['return_pct']:+.1f}% | "
            f"{r['profit_factor']:.2f} | {r['total_trades']} | "
            f"{r['max_drawdown_pct']*100:.1f}% |"
        )

    return "\n".join(lines)


def save_results(all_results: list[dict], rankings: list[dict],
                 cross_configs: list[dict], pair_bests: dict,
                 tier: str, elapsed: float):
    """Save all results to files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Summary MD
    md = generate_summary_md(rankings, cross_configs, pair_bests, tier, elapsed)
    (OUTPUT_DIR / f"results_mega_{tier}_summary.md").write_text(md)

    # Full results JSON (top 100 per strategy)
    strat_top = defaultdict(list)
    for r in sorted(all_results, key=lambda x: x["total_pnl"], reverse=True):
        s = r["strategy"]
        if len(strat_top[s]) < 100:
            strat_top[s].append(r)

    with open(OUTPUT_DIR / f"results_mega_{tier}_full.json", "w") as f:
        json.dump({"rankings": rankings, "top_per_strategy": dict(strat_top)},
                  f, indent=2, default=str)

    # Cross-pair JSON
    with open(OUTPUT_DIR / f"results_mega_{tier}_cross_pair.json", "w") as f:
        json.dump(cross_configs[:50], f, indent=2, default=str)

    return OUTPUT_DIR


def print_summary(rankings: list[dict], cross_configs: list[dict],
                  pair_bests: dict, tier: str, total_sims: int, elapsed: float):
    """Print concise summary to console."""
    print(f"\n{'='*80}")
    print(f"MEGA BACKTEST RESULTS — {tier.upper()}")
    print(f"{'='*80}")
    print(f"Simulations: {total_sims:,} | Time: {elapsed:.1f}s | "
          f"Speed: {total_sims/max(elapsed,0.1):,.0f} sims/s")

    print(f"\n--- STRATEGY RANKING (top 15) ---")
    print(f"{'#':>3} {'Strategy':25} {'ProfRate':>8} {'Pairs':>6} "
          f"{'BestPnL':>12} {'MedianPnL':>12} {'AvgPF':>6} {'AvgWR':>6}")
    print("-" * 90)
    for i, r in enumerate(rankings[:15], 1):
        print(f"{i:3d} {r['strategy']:25} {r['profit_rate']:7.1f}% "
              f"{r['pairs_profitable']:2d}/{r['pairs_total']:2d} "
              f"${r['best_pnl']:+11,.0f} ${r['median_pnl']:+11,.0f} "
              f"{r['avg_pf']:5.2f} {r['avg_win_rate']:5.1f}%")

    if cross_configs:
        print(f"\n--- CROSS-PAIR CONFIGS (top 10) ---")
        for i, c in enumerate(cross_configs[:10], 1):
            p = c["params"]
            print(f"  #{i:2d} {c['strategy']:25} | {c['n_pairs']} pairs | "
                  f"PnL=${c['total_pnl']:+,.0f} ({c['return_pct']:+.1f}%) | "
                  f"PF={c['avg_pf']:.2f} | WR={c['avg_win_rate']:.1f}%")
            print(f"       SL={p['sl_pct']*100:.0f}% TP={p['tp_pct']*100:.0f}% "
                  f"Trail={p['trailing_pct']*100:.0f}% Lev={p['leverage']}x "
                  f"CD={p['cooldown_bars']}h Pos={p['position_pct']*100:.0f}% "
                  f"Score>={p['min_score']}")

    print(f"\n--- PER-PAIR BEST ---")
    keep = []
    drop = []
    for pair in sorted(pair_bests.keys()):
        r = pair_bests[pair]
        status = "+" if r["profit_factor"] > 1.0 else "-"
        print(f"  [{status}] {pair:10} {r['strategy']:25} "
              f"PnL=${r['total_pnl']:+9,.0f} PF={r['profit_factor']:.2f} "
              f"trades={r['total_trades']:3d}")
        if r["profit_factor"] > 1.0:
            keep.append(pair)
        else:
            drop.append(pair)

    print(f"\n  KEEP ({len(keep)}): {', '.join(keep)}")
    if drop:
        print(f"  DROP ({len(drop)}): {', '.join(drop)}")
