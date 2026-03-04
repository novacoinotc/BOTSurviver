"""Parameter grids for Tier 1 and Tier 2 mega backtest.

Tier 1: Core grid (~180 combos per strategy) for all 27 strategies.
Tier 2: Expanded grid (~500 combos) for top 10 strategies from Tier 1.
"""

from itertools import product


def _build_grid(sl_list, tp_list, trail_list, cd_list, lev_list,
                pos_list, score_list) -> list[dict]:
    """Build a list of param dicts from ranges, filtering tp > sl."""
    grid = []
    for sl, tp, tr, cd, lev, pos, ms in product(
        sl_list, tp_list, trail_list, cd_list, lev_list, pos_list, score_list
    ):
        if tp <= sl:
            continue
        grid.append({
            "sl_pct": sl,
            "tp_pct": tp,
            "trailing_pct": tr,
            "cooldown_bars": cd,
            "leverage": lev,
            "position_pct": pos,
            "min_score": ms,
        })
    return grid


# ===========================================================================
# Tier 1 — Core grid (~180 combos per strategy)
# ===========================================================================

TIER1_GRID = _build_grid(
    sl_list=[0.02, 0.03, 0.05],
    tp_list=[0.04, 0.06, 0.10, 0.15],
    trail_list=[0.0, 0.03, 0.06],
    cd_list=[12, 24, 48],
    lev_list=[5, 10],
    pos_list=[0.02],
    score_list=[5],
)

# ===========================================================================
# Tier 2 — Expanded grid (~500 combos for top strategies)
# ===========================================================================

TIER2_GRID = _build_grid(
    sl_list=[0.015, 0.02, 0.03, 0.05],
    tp_list=[0.03, 0.04, 0.06, 0.10, 0.15],
    trail_list=[0.0, 0.03, 0.06],
    cd_list=[8, 12, 24, 48],
    lev_list=[5, 10, 20],
    pos_list=[0.01, 0.02],
    score_list=[4, 5],
)


def get_tier1_grid() -> list[dict]:
    """Get Tier 1 parameter grid."""
    return TIER1_GRID


def get_tier2_grid() -> list[dict]:
    """Get Tier 2 parameter grid."""
    return TIER2_GRID


# Print grid sizes when run directly
if __name__ == "__main__":
    print(f"Tier 1 grid: {len(TIER1_GRID)} combos")
    print(f"Tier 2 grid: {len(TIER2_GRID)} combos")
    print(f"\nTier 1 sample: {TIER1_GRID[0]}")
    print(f"Tier 2 sample: {TIER2_GRID[0]}")

    # Estimate total simulations
    n_strats = 27
    n_pairs = 19
    tier1_total = len(TIER1_GRID) * n_strats * n_pairs
    tier2_total = len(TIER2_GRID) * 10 * n_pairs  # top 10
    print(f"\nEstimated Tier 1 simulations: {tier1_total:,}")
    print(f"Estimated Tier 2 simulations: {tier2_total:,}")
    print(f"Total: {tier1_total + tier2_total:,}")
