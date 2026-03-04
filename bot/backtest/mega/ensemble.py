"""Ensemble voting: enter only when N+ strategies agree on the same bar+direction.

Groups signals from all 27 strategies by (bar, direction). When N or more
strategies fire on the same bar in the same direction, an ensemble signal
is generated with score = sum of individual scores.

Ensemble levels:
    ensemble_2: 2+ strategies agree
    ensemble_3: 3+ strategies agree
    ensemble_4: 4+ strategies agree
    ensemble_5: 5+ strategies agree
"""

from collections import defaultdict


ENSEMBLE_LEVELS = [2, 3, 4, 5]


def build_ensemble_signals(all_signals: dict[str, list[dict]],
                           min_agree: int = 2) -> list[dict]:
    """Build ensemble signals from all strategy signals.

    Args:
        all_signals: {strategy_name: [signal_dicts]} — output of detect_all_signals()
        min_agree: minimum number of strategies that must agree

    Returns:
        List of ensemble signal dicts:
            {"bar": int, "dir": str, "strat": f"ensemble_{min_agree}",
             "score": float, "n_agree": int, "sources": [str]}
    """
    # Group by (bar, direction)
    bar_dir_map = defaultdict(list)
    for strat_name, sigs in all_signals.items():
        for s in sigs:
            key = (s["bar"], s["dir"])
            bar_dir_map[key].append({
                "strat": strat_name,
                "score": s["score"],
            })

    ensemble_signals = []
    for (bar, direction), entries in bar_dir_map.items():
        # Deduplicate: one vote per strategy per bar
        seen_strats = set()
        unique_entries = []
        for e in entries:
            if e["strat"] not in seen_strats:
                seen_strats.add(e["strat"])
                unique_entries.append(e)

        if len(unique_entries) >= min_agree:
            total_score = sum(e["score"] for e in unique_entries)
            sources = sorted(e["strat"] for e in unique_entries)
            ensemble_signals.append({
                "bar": bar,
                "dir": direction,
                "strat": f"ensemble_{min_agree}",
                "score": total_score,
                "n_agree": len(unique_entries),
                "sources": sources,
            })

    # Sort by bar index for chronological processing
    ensemble_signals.sort(key=lambda s: s["bar"])
    return ensemble_signals


def build_all_ensemble_levels(all_signals: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Build ensemble signals for all agreement levels.

    Returns:
        {f"ensemble_{n}": [signal_dicts]} for n in [2, 3, 4, 5]
    """
    return {
        f"ensemble_{n}": build_ensemble_signals(all_signals, min_agree=n)
        for n in ENSEMBLE_LEVELS
    }
