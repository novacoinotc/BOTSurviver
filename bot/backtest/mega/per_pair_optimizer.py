"""Per-pair portfolio optimizer.

Takes the best strategy per pair from Phase 1, runs a fine parameter grid
around the winning config, then simulates a portfolio where each pair
trades independently with equal capital allocation.

Portfolio: $50K / 19 pairs = $2,632 per pair.
Compare portfolio PnL vs best-single-config across all pairs.
"""

import numpy as np
from itertools import product


# ---------------------------------------------------------------------------
# Fine grid builder: narrow range around best params
# ---------------------------------------------------------------------------

def build_fine_grid(best_params: dict) -> list[dict]:
    """Build a fine grid around a winning config (~200 combos).

    Varies each param by ±1 step around the best value.
    """
    sl = best_params["sl_pct"]
    tp = best_params["tp_pct"]
    trail = best_params["trailing_pct"]
    cd = best_params["cooldown_bars"]
    lev = best_params["leverage"]
    pos = best_params["position_pct"]
    ms = best_params["min_score"]

    # Build tight ranges around best values (~200 combos target)
    sl_range = sorted(set([
        max(0.01, sl - 0.01), sl, min(0.10, sl + 0.01)
    ]))
    tp_range = sorted(set([
        max(0.02, tp - 0.02), tp, min(0.20, tp + 0.02)
    ]))
    trail_range = sorted(set([
        max(0.0, trail - 0.02), trail, min(0.10, trail + 0.02)
    ]))
    cd_range = sorted(set([
        max(4, cd - 4), cd, cd + 8
    ]))
    lev_range = sorted(set([
        max(2, lev - 5), lev
    ]))
    pos_range = sorted(set([pos]))
    score_range = sorted(set([int(ms)]))

    grid = []
    for s, t, tr, c, l, p, m in product(
        sl_range, tp_range, trail_range, cd_range,
        lev_range, pos_range, score_range
    ):
        if t <= s:
            continue
        grid.append({
            "sl_pct": round(s, 4),
            "tp_pct": round(t, 4),
            "trailing_pct": round(tr, 4),
            "cooldown_bars": c,
            "leverage": l,
            "position_pct": round(p, 4),
            "min_score": m,
        })
    return grid


# ---------------------------------------------------------------------------
# Per-pair optimization
# ---------------------------------------------------------------------------

def optimize_per_pair(pair: str, strategy: str, signals: list[dict],
                      df, best_params: dict, simulate_fn) -> dict:
    """Run fine grid optimization for a single pair.

    Returns:
        Dict with pair, strategy, best_result, all_results count
    """
    fine_grid = build_fine_grid(best_params)

    best_result = None
    n_tested = 0
    n_profitable = 0

    for params in fine_grid:
        result = simulate_fn(df, signals, params, pair)
        if result and result["total_trades"] >= 3:
            n_tested += 1
            if result["profit_factor"] > 1.0:
                n_profitable += 1
            if best_result is None or result["total_pnl"] > best_result["total_pnl"]:
                best_result = result
                best_result["params"] = {k: v for k, v in params.items()}
                best_result["pair"] = pair
                best_result["strategy"] = strategy

    return {
        "pair": pair,
        "strategy": strategy,
        "grid_size": len(fine_grid),
        "n_tested": n_tested,
        "n_profitable": n_profitable,
        "best_result": best_result,
        "original_params": best_params,
    }


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------

PORTFOLIO_EQUITY = 50_000.0


def simulate_portfolio(pair_results: dict[str, dict],
                       n_pairs: int = 19) -> dict:
    """Simulate portfolio from per-pair best results.

    Each pair gets equal capital allocation: $50K / n_pairs.
    PnL is scaled by the allocation ratio.

    Args:
        pair_results: {pair: best_result_dict} — each has total_pnl based on $50K
        n_pairs: number of pairs in portfolio

    Returns:
        Portfolio summary dict
    """
    per_pair_equity = PORTFOLIO_EQUITY / n_pairs

    total_pnl = 0.0
    total_trades = 0
    total_wins = 0
    pair_details = []
    monthly_agg = {}

    for pair, result in sorted(pair_results.items()):
        if result is None:
            continue

        # Scale PnL from $50K base to per-pair allocation
        scale = per_pair_equity / PORTFOLIO_EQUITY
        scaled_pnl = result["total_pnl"] * scale

        total_pnl += scaled_pnl
        total_trades += result["total_trades"]
        total_wins += result.get("wins", 0)

        pair_details.append({
            "pair": pair,
            "strategy": result.get("strategy", "unknown"),
            "pnl_scaled": round(scaled_pnl, 2),
            "pnl_full": round(result["total_pnl"], 2),
            "pf": result.get("profit_factor", 0),
            "trades": result["total_trades"],
            "win_rate": round(result.get("win_rate", 0) * 100, 1),
            "max_dd": round(result.get("max_drawdown_pct", 0) * 100, 2),
        })

        # Aggregate monthly
        for m in result.get("monthly", []):
            month = m["month"]
            if month not in monthly_agg:
                monthly_agg[month] = {"pnl": 0, "trades": 0, "wins": 0}
            monthly_agg[month]["pnl"] += m["pnl"] * scale
            monthly_agg[month]["trades"] += m["trades"]
            monthly_agg[month]["wins"] += m["wins"]

    green_months = sum(1 for v in monthly_agg.values() if v["pnl"] > 0)
    win_rate = total_wins / max(total_trades, 1) * 100

    return {
        "total_pnl": round(total_pnl, 2),
        "return_pct": round(total_pnl / PORTFOLIO_EQUITY * 100, 2),
        "total_trades": total_trades,
        "win_rate": round(win_rate, 1),
        "n_pairs": len(pair_details),
        "n_profitable": sum(1 for p in pair_details if p["pnl_scaled"] > 0),
        "green_months": green_months,
        "total_months": len(monthly_agg),
        "pair_details": sorted(pair_details, key=lambda x: x["pnl_scaled"], reverse=True),
        "monthly": [{"month": k, **v} for k, v in sorted(monthly_agg.items())],
    }
