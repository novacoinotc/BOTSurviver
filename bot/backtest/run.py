#!/usr/bin/env python3
"""
Main backtesting runner.
Downloads data, computes indicators, runs optimization across all strategies.
Outputs the best strategy+params combinations sorted by profitability.
"""

import sys
import time
import json
from pathlib import Path

# Add bot/ to path so we can run from anywhere
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.data_loader import download_klines, load_proxy_url
from backtest.indicators import compute_all_indicators
from backtest.optimizer import optimize, print_results, save_results

# Top 5 most liquid pairs
PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

# All strategies to test
STRATEGIES = [
    "trend_follow",
    "breakout",
    "mean_reversion",
    "momentum",
    "macd_cross",
    "ema_cross",
    "bollinger_bounce",
    "volume_spike",
    "stoch_reversal",
    "ichimoku_cloud",
    "triple_confirmation",
    "williams_cci_combo",
]

DAYS = 14  # 2 weeks of data
INTERVAL = "1m"
TOP_N = 10  # Keep top N results per strategy


def main():
    print("=" * 70)
    print("BACKTESTING OPTIMIZER — FINDING THE BEST STRATEGY")
    print("=" * 70)
    print(f"Pairs: {PAIRS}")
    print(f"Strategies: {len(STRATEGIES)}")
    print(f"Data: {DAYS} days of {INTERVAL} candles")
    print()

    # --- Step 1: Download data ---
    print("STEP 1: Downloading historical data...")
    proxy_url = load_proxy_url()
    if proxy_url:
        print(f"  Using proxy: {proxy_url[:20]}...")
    else:
        print("  No proxy configured, trying direct access...")

    data = {}  # {pair: df_with_indicators}
    for pair in PAIRS:
        print(f"\n  [{pair}] Downloading {DAYS}d of {INTERVAL} candles...")
        df = download_klines(pair, interval=INTERVAL, days=DAYS, proxy_url=proxy_url)
        if df.empty:
            print(f"  [{pair}] FAILED — no data, skipping")
            continue

        print(f"  [{pair}] Computing indicators on {len(df)} candles...")
        df = compute_all_indicators(df)
        data[pair] = df
        print(f"  [{pair}] Ready: {len(df)} rows, {len(df.columns)} columns")

    if not data:
        print("\nERROR: No data downloaded. Check proxy/network.")
        sys.exit(1)

    print(f"\nData ready for {len(data)} pairs.")

    # --- Step 2: Run optimization ---
    print("\n" + "=" * 70)
    print("STEP 2: Running optimization...")
    print("=" * 70)

    all_results = []  # Flat list of all results across strategies and pairs

    for strategy_name in STRATEGIES:
        print(f"\n{'─' * 50}")
        print(f"Strategy: {strategy_name}")
        print(f"{'─' * 50}")

        for pair, df in data.items():
            try:
                results = optimize(df, pair, strategy_name, top_n=TOP_N)
                for r in results:
                    r["pair"] = pair
                all_results.extend(results)
            except Exception as e:
                print(f"  [{pair}] Error: {e}")

    # --- Step 3: Rank all results ---
    print("\n" + "=" * 70)
    print("STEP 3: GLOBAL RANKINGS")
    print("=" * 70)

    # Filter: must have >= 10 trades and profit_factor > 0
    qualified = [r for r in all_results if r.get("total_trades", 0) >= 10]

    # Sort by profit factor (primary), then by total_pnl (secondary)
    qualified.sort(key=lambda x: (x.get("profit_factor", 0), x.get("total_pnl", 0)), reverse=True)

    print(f"\nTotal configurations tested: {len(all_results)}")
    print(f"Qualified (>= 10 trades): {len(qualified)}")

    profitable = [r for r in qualified if r.get("profit_factor", 0) > 1.0]
    print(f"Profitable (PF > 1.0): {len(profitable)}")

    # Top 30 overall
    print(f"\n{'=' * 70}")
    print("TOP 30 STRATEGIES (sorted by profit factor)")
    print(f"{'=' * 70}")

    for i, r in enumerate(qualified[:30], 1):
        pf = r.get("profit_factor", 0)
        wr = r.get("win_rate", 0) * 100
        pnl = r.get("total_pnl", 0)
        trades = r.get("total_trades", 0)
        dd = r.get("max_drawdown_pct", 0) * 100
        sharpe = r.get("sharpe_ratio", 0)
        pair = r.get("pair", "?")
        strat = r.get("strategy", "?")

        print(f"  #{i:2d} | {strat:22s} | {pair:10s} | "
              f"PF={pf:5.2f} WR={wr:5.1f}% PnL=${pnl:+8.2f} "
              f"trades={trades:3d} DD={dd:5.1f}% Sharpe={sharpe:5.2f}")

    # Save detailed results
    results_file = Path(__file__).parent / "results.json"
    save_results(qualified[:100], str(results_file))
    print(f"\nDetailed results saved to: {results_file}")

    # Print best strategy's full params
    if qualified:
        best = qualified[0]
        print(f"\n{'=' * 70}")
        print("BEST STRATEGY DETAILS")
        print(f"{'=' * 70}")
        print(f"Strategy: {best.get('strategy')}")
        print(f"Pair: {best.get('pair')}")
        print(f"Profit Factor: {best.get('profit_factor', 0):.2f}")
        print(f"Win Rate: {best.get('win_rate', 0)*100:.1f}%")
        print(f"Total PnL: ${best.get('total_pnl', 0):+.2f}")
        print(f"Total Trades: {best.get('total_trades', 0)}")
        print(f"Max Drawdown: {best.get('max_drawdown_pct', 0)*100:.1f}%")
        print(f"Sharpe Ratio: {best.get('sharpe_ratio', 0):.2f}")
        print(f"Avg Win: {best.get('avg_win_pct', 0)*100:.2f}%")
        print(f"Avg Loss: {best.get('avg_loss_pct', 0)*100:.2f}%")
        print(f"\nParameters:")
        params = best.get("params", {})
        for k, v in sorted(params.items()):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
