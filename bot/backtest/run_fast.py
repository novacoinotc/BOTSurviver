#!/usr/bin/env python3
"""
Fast backtesting runner — runs ALL backtests in-process (no pickle overhead).
Uses 5m candles for speed while maintaining accuracy.
"""

import sys
import time
import json
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.data_loader import download_klines, load_proxy_url
from backtest.indicators import compute_all_indicators
from backtest.backtester import Backtester
from backtest.strategies import STRATEGY_REGISTRY, get_strategy
from backtest.optimizer import generate_param_grid, _make_strategy_wrapper

# Config
PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
DAYS = 14
INTERVAL = "5m"  # 5m candles = 5x less data, same signals
MIN_TRADES = 5   # Minimum trades to qualify


def run_strategy_pair(df, pair, strategy_name, param_combos):
    """Run all param combos for one strategy on one pair. Returns top 20."""
    results = []
    total = len(param_combos)

    for i, params in enumerate(param_combos):
        # Create fresh wrapper for each param set (resets prev-bar state)
        wrapper = _make_strategy_wrapper(strategy_name)
        bt = Backtester(df, pair, fee_pct=0.0005, initial_equity=5000)
        result = bt.run(wrapper, params)

        if result.total_trades >= MIN_TRADES:
            results.append({
                "strategy": strategy_name,
                "pair": pair,
                "params": {k: v for k, v in params.items() if k != "strategy_name"},
                "total_trades": result.total_trades,
                "wins": result.wins,
                "win_rate": round(result.win_rate, 4),
                "total_pnl": round(result.total_pnl, 2),
                "profit_factor": round(result.profit_factor, 4),
                "max_drawdown_pct": round(result.max_drawdown_pct, 4),
                "sharpe_ratio": round(result.sharpe_ratio, 4),
                "avg_win_pct": round(result.avg_win_pct, 6),
                "avg_loss_pct": round(result.avg_loss_pct, 6),
                "avg_hold_bars": round(result.avg_hold_bars, 1),
            })

    # Sort by profit_factor, then pnl
    results.sort(key=lambda r: (r["profit_factor"], r["total_pnl"]), reverse=True)
    return results[:20]


def main():
    t0_global = time.time()

    print("=" * 70)
    print("FAST BACKTESTING OPTIMIZER")
    print("=" * 70)

    # Step 1: Download data
    print("\nSTEP 1: Downloading data...")
    proxy_url = load_proxy_url()

    data = {}
    for pair in PAIRS:
        print(f"  {pair}...", end=" ", flush=True)
        df = download_klines(pair, interval=INTERVAL, days=DAYS, proxy_url=proxy_url)
        if df.empty:
            print("FAILED")
            continue
        df = compute_all_indicators(df)
        data[pair] = df
        print(f"{len(df)} candles, {len(df.columns)} cols")

    if not data:
        print("ERROR: No data!")
        sys.exit(1)

    # Step 2: Run all strategies
    print(f"\nSTEP 2: Running optimization ({len(STRATEGY_REGISTRY)} strategies x {len(data)} pairs)")
    print("=" * 70)

    all_results = []
    total_backtests = 0

    for s_idx, strategy_name in enumerate(STRATEGY_REGISTRY, 1):
        param_combos = generate_param_grid(strategy_name)
        n_combos = len(param_combos)
        n_pairs = len(data)
        total_for_strategy = n_combos * n_pairs

        print(f"\n[{s_idx}/{len(STRATEGY_REGISTRY)}] {strategy_name}: {n_combos} combos x {n_pairs} pairs = {total_for_strategy} backtests")

        t0_strat = time.time()
        strategy_results = []

        for pair, df in data.items():
            t0_pair = time.time()
            results = run_strategy_pair(df, pair, strategy_name, param_combos)
            elapsed_pair = time.time() - t0_pair
            rate = n_combos / elapsed_pair if elapsed_pair > 0 else 0

            qualified = len(results)
            profitable = sum(1 for r in results if r["profit_factor"] > 1.0)
            best_pf = results[0]["profit_factor"] if results else 0
            best_pnl = results[0]["total_pnl"] if results else 0

            print(f"  {pair}: {elapsed_pair:.1f}s ({rate:.0f} bt/s) | "
                  f"qualified={qualified} profitable={profitable} "
                  f"best_PF={best_pf:.2f} best_PnL=${best_pnl:+.2f}")

            strategy_results.extend(results)
            total_backtests += n_combos

        # Global sort for this strategy
        strategy_results.sort(key=lambda r: (r["profit_factor"], r["total_pnl"]), reverse=True)
        all_results.extend(strategy_results[:20])

        elapsed_strat = time.time() - t0_strat
        if strategy_results:
            best = strategy_results[0]
            print(f"  >>> BEST {strategy_name}: PF={best['profit_factor']:.2f} "
                  f"WR={best['win_rate']*100:.1f}% PnL=${best['total_pnl']:+.2f} "
                  f"trades={best['total_trades']} [{best['pair']}] ({elapsed_strat:.1f}s)")

    # Step 3: Global rankings
    elapsed_total = time.time() - t0_global

    print(f"\n{'=' * 70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total backtests: {total_backtests:,}")
    print(f"Total time: {elapsed_total:.1f}s ({total_backtests/elapsed_total:.0f} bt/s)")

    # Filter and sort
    all_results.sort(key=lambda r: (r["profit_factor"], r["total_pnl"]), reverse=True)

    profitable = [r for r in all_results if r["profit_factor"] > 1.0]
    print(f"Profitable configurations: {len(profitable)}")

    print(f"\n{'=' * 70}")
    print("TOP 30 STRATEGIES (sorted by profit factor)")
    print(f"{'=' * 70}")
    print(f"{'#':>3} | {'Strategy':22} | {'Pair':10} | {'PF':>5} | {'WR':>5} | {'PnL':>9} | {'Trades':>6} | {'DD':>6} | {'Sharpe':>6}")
    print("-" * 95)

    for i, r in enumerate(all_results[:30], 1):
        print(f"{i:3d} | {r['strategy']:22} | {r['pair']:10} | "
              f"{r['profit_factor']:5.2f} | {r['win_rate']*100:4.1f}% | "
              f"${r['total_pnl']:+8.2f} | {r['total_trades']:6d} | "
              f"{r['max_drawdown_pct']*100:5.1f}% | {r['sharpe_ratio']:6.2f}")

    # Save results
    results_file = Path(__file__).parent / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results[:100], f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Print #1 details
    if all_results:
        best = all_results[0]
        print(f"\n{'=' * 70}")
        print("BEST STRATEGY DETAILS")
        print(f"{'=' * 70}")
        print(f"Strategy: {best['strategy']}")
        print(f"Pair: {best['pair']}")
        print(f"Profit Factor: {best['profit_factor']:.2f}")
        print(f"Win Rate: {best['win_rate']*100:.1f}%")
        print(f"Total PnL: ${best['total_pnl']:+.2f}")
        print(f"Trades: {best['total_trades']}")
        print(f"Max Drawdown: {best['max_drawdown_pct']*100:.1f}%")
        print(f"Sharpe: {best['sharpe_ratio']:.2f}")
        print(f"Avg Win: {best['avg_win_pct']*100:.3f}%")
        print(f"Avg Loss: {best['avg_loss_pct']*100:.3f}%")
        print(f"Avg Hold: {best['avg_hold_bars']:.0f} bars")
        print(f"\nParameters:")
        for k, v in sorted(best['params'].items()):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
