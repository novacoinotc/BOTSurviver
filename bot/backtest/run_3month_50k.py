#!/usr/bin/env python3
"""
3-month backtest: macd_cross with $50K and leverage 3x-12x.
Accepts pairs as CLI args for parallel execution.
"""
import sys, time, json, numpy as np, pandas as pd
from pathlib import Path
from itertools import product
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.data_loader import download_klines, load_proxy_url
from backtest.indicators import compute_all_indicators
from backtest.run_optimized import detect_all_signals, FEE_PCT

# ============================================================
# CONFIG
# ============================================================
ALL_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
PAIRS = sys.argv[1:] if len(sys.argv) > 1 else ALL_PAIRS
DAYS = 90
INTERVAL = "5m"
INITIAL_EQUITY = 50_000.0
STRATEGY = "macd_cross"

# ============================================================
# PARAMETER GRID — leverage 3x to 12x
# ============================================================
PARAM_GRID = []
for sl, tp, tr, cd, ms, lev, pos in product(
    [0.006, 0.008, 0.010, 0.012, 0.015],       # SL %
    [0.030, 0.040, 0.060, 0.080, 0.100],        # TP %
    [0.015, 0.020, 0.025, 0.030],               # Trailing %
    [5, 10],                                      # Cooldown bars
    [0, 4],                                       # Min score
    [3, 4, 5, 6, 8, 10, 12],                     # Leverage
    [0.005, 0.010],                               # Position size %
):
    if tp <= sl:
        continue
    PARAM_GRID.append({
        "sl_pct": sl, "tp_pct": tp, "trailing_pct": tr,
        "cooldown_bars": cd, "min_score": ms, "leverage": lev,
        "position_pct": pos,
    })

print(f"Param combos: {len(PARAM_GRID)}", flush=True)


# ============================================================
# TRADE SIMULATION ($50K equity)
# ============================================================
def simulate_trades_50k(df, signals, params):
    """Simulate trades with $50K equity, variable leverage."""
    sl_pct = params["sl_pct"]
    tp_pct = params["tp_pct"]
    trailing_pct = params.get("trailing_pct", 0.015)
    cooldown = params.get("cooldown_bars", 15)
    min_score = params.get("min_score", 0)
    leverage = params.get("leverage", 3)
    pos_pct = params.get("position_pct", 0.005)

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n_bars = len(df)

    equity = INITIAL_EQUITY
    peak_equity = equity
    max_dd = 0.0
    trades = []
    last_exit_bar = -cooldown - 1

    for sig in signals:
        bar = sig["bar"]
        if sig["score"] < min_score:
            continue
        if bar <= last_exit_bar + cooldown:
            continue
        if bar >= n_bars - 1:
            continue

        entry_price = closes[bar]
        direction = sig["dir"]
        notional = equity * pos_pct * leverage

        if direction == "LONG":
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)
        else:
            sl_price = entry_price * (1 + sl_pct)
            tp_price = entry_price * (1 - tp_pct)

        best_price = entry_price
        exit_price = None
        exit_reason = None
        exit_bar = None

        for j in range(bar + 1, n_bars):
            h = highs[j]
            l = lows[j]

            if direction == "LONG":
                if h > best_price:
                    best_price = h
                    new_sl = best_price * (1 - trailing_pct)
                    if new_sl > sl_price:
                        sl_price = new_sl
                if l <= sl_price:
                    exit_price = sl_price
                    exit_reason = "sl"
                    exit_bar = j
                    break
                if h >= tp_price:
                    exit_price = tp_price
                    exit_reason = "tp"
                    exit_bar = j
                    break
            else:
                if l < best_price:
                    best_price = l
                    new_sl = best_price * (1 + trailing_pct)
                    if new_sl < sl_price:
                        sl_price = new_sl
                if h >= sl_price:
                    exit_price = sl_price
                    exit_reason = "sl"
                    exit_bar = j
                    break
                if l <= tp_price:
                    exit_price = tp_price
                    exit_reason = "tp"
                    exit_bar = j
                    break

        if exit_price is None:
            exit_price = closes[-1]
            exit_reason = "eod"
            exit_bar = n_bars - 1

        if direction == "LONG":
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        fee_cost = 2 * FEE_PCT * notional
        pnl_usd = notional * pnl_pct - fee_cost

        equity += pnl_usd
        if equity > peak_equity:
            peak_equity = equity
        dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
        if dd < max_dd:
            max_dd = dd

        # Liquidation check
        if equity <= 0:
            trades.append({
                "pnl_usd": pnl_usd, "pnl_pct": pnl_pct,
                "exit_reason": "liquidation", "hold_bars": exit_bar - bar,
                "direction": direction,
            })
            equity = 0
            break

        last_exit_bar = exit_bar
        trades.append({
            "pnl_usd": pnl_usd, "pnl_pct": pnl_pct,
            "exit_reason": exit_reason, "hold_bars": exit_bar - bar,
            "direction": direction,
        })

    if not trades:
        return None

    pnls = [t["pnl_usd"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0

    win_pcts = [t["pnl_pct"] for t in trades if t["pnl_usd"] > 0]
    loss_pcts = [t["pnl_pct"] for t in trades if t["pnl_usd"] <= 0]
    pnl_pcts_arr = [t["pnl_pct"] for t in trades]
    sharpe = 0
    if len(pnl_pcts_arr) > 1 and np.std(pnl_pcts_arr) > 0:
        sharpe = np.mean(pnl_pcts_arr) / np.std(pnl_pcts_arr) * np.sqrt(len(pnl_pcts_arr))

    tp_hits = sum(1 for t in trades if t["exit_reason"] == "tp")
    sl_hits = sum(1 for t in trades if t["exit_reason"] == "sl")
    trail_wins = wins - tp_hits

    return {
        "total_trades": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades),
        "total_pnl": sum(pnls),
        "profit_factor": pf,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sharpe,
        "avg_win_pct": np.mean(win_pcts) if win_pcts else 0,
        "avg_loss_pct": np.mean(loss_pcts) if loss_pcts else 0,
        "avg_hold": np.mean([t["hold_bars"] for t in trades]),
        "tp_hits": tp_hits,
        "sl_hits": sl_hits,
        "trail_wins": trail_wins,
        "final_equity": equity,
        "return_pct": (equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100,
    }


def main():
    t0 = time.time()
    proxy_url = load_proxy_url()

    print("=" * 70, flush=True)
    print(f"3-MONTH BACKTEST: macd_cross | $50K | Leverage 3x-12x", flush=True)
    print("=" * 70, flush=True)
    print(f"Pairs: {PAIRS}", flush=True)
    print(f"Days: {DAYS}", flush=True)
    print(f"Params: {len(PARAM_GRID)} combos", flush=True)
    print(flush=True)

    # Download data
    data = {}
    for pair in PAIRS:
        print(f"  Downloading {pair}...", end=" ", flush=True)
        df = download_klines(pair, interval=INTERVAL, days=DAYS, proxy_url=proxy_url)
        if df.empty:
            print("FAIL", flush=True)
            continue
        df = compute_all_indicators(df)
        data[pair] = df
        print(f"{len(df)} candles", flush=True)

    if not data:
        print("ERROR: No data!", flush=True)
        sys.exit(1)

    # Detect signals
    print(f"\nDetecting macd_cross signals...", flush=True)
    pair_signals = {}
    for pair, df in data.items():
        signals = detect_all_signals(df)
        macd_sigs = [s for s in signals if s["strat"] == STRATEGY]
        print(f"  {pair}: {len(macd_sigs)} macd_cross signals", flush=True)
        pair_signals[pair] = macd_sigs

    # Run simulations
    total_sims = len(PARAM_GRID) * len(data)
    print(f"\nRunning {total_sims:,} simulations...", flush=True)

    all_results = []
    sim_count = 0
    t_sim = time.time()

    for pair, df in data.items():
        sigs = pair_signals[pair]
        if not sigs:
            continue

        t_pair = time.time()
        pair_results = []

        for params in PARAM_GRID:
            result = simulate_trades_50k(df, sigs, params)
            sim_count += 1

            if result and result["total_trades"] >= 10:
                result["pair"] = pair
                result["params"] = {k: v for k, v in params.items()}
                pair_results.append(result)

        elapsed_pair = time.time() - t_pair
        profitable = sum(1 for r in pair_results if r["profit_factor"] > 1.0)
        best_pnl = max((r["total_pnl"] for r in pair_results), default=0)
        best_ret = max((r["return_pct"] for r in pair_results), default=0)
        print(f"  {pair}: {len(pair_results)} qualified, {profitable} profitable, "
              f"best=${best_pnl:+,.2f} ({best_ret:+.1f}%) ({elapsed_pair:.1f}s)", flush=True)
        all_results.extend(pair_results)

    elapsed_sim = time.time() - t_sim
    print(f"\n  {sim_count:,} sims in {elapsed_sim:.1f}s ({sim_count/elapsed_sim:.0f} sims/s)", flush=True)

    # Sort results
    all_results.sort(key=lambda r: (r["profit_factor"], r["total_pnl"]), reverse=True)
    profitable = [r for r in all_results if r["profit_factor"] > 1.0]

    print(f"\n{'='*70}", flush=True)
    print(f"RESULTS (3 MONTHS, $50K INITIAL)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Total qualified: {len(all_results):,}", flush=True)
    print(f"Profitable: {len(profitable):,}", flush=True)

    # Top 50
    print(f"\n{'#':>3} | {'Pair':10} | {'PF':>5} | {'WR':>5} | {'PnL':>12} | {'Ret%':>7} | "
          f"{'Trd':>4} | {'DD':>6} | {'Sharpe':>6} | {'Lev':>3} | {'SL':>5} | {'TP':>5} | "
          f"{'Trail':>5} | {'Pos%':>4}", flush=True)
    print("-" * 135, flush=True)
    for i, r in enumerate(all_results[:50], 1):
        p = r["params"]
        print(f"{i:3d} | {r['pair']:10} | {r['profit_factor']:5.2f} | {r['win_rate']*100:4.1f}% | "
              f"${r['total_pnl']:+11,.2f} | {r['return_pct']:+6.1f}% | {r['total_trades']:4d} | "
              f"{r['max_drawdown_pct']*100:5.1f}% | {r['sharpe_ratio']:6.2f} | {p['leverage']:2d}x | "
              f"{p['sl_pct']*100:4.1f}% | {p['tp_pct']*100:4.1f}% | {p['trailing_pct']*100:4.1f}% | "
              f"{p['position_pct']*100:.1f}%", flush=True)

    # ============================================================
    # CROSS-PAIR CONSISTENCY
    # ============================================================
    strat_cross = defaultdict(list)
    for r in all_results:
        if r["profit_factor"] > 1.0:
            key = json.dumps(r["params"], sort_keys=True)
            strat_cross[key].append(r)

    consistent = []
    for ps, entries in strat_cross.items():
        pairs_ok = set(e["pair"] for e in entries)
        if len(pairs_ok) >= 3:
            consistent.append({
                "params": json.loads(ps),
                "n_pairs": len(pairs_ok),
                "pairs": sorted(pairs_ok),
                "total_pnl": sum(e["total_pnl"] for e in entries),
                "total_return_pct": sum(e["total_pnl"] for e in entries) / INITIAL_EQUITY * 100,
                "avg_pf": np.mean([e["profit_factor"] for e in entries]),
                "avg_wr": np.mean([e["win_rate"] for e in entries]),
                "total_trades": sum(e["total_trades"] for e in entries),
                "worst_dd": min(e["max_drawdown_pct"] for e in entries),
                "per_pair": {e["pair"]: {
                    "pf": e["profit_factor"], "wr": e["win_rate"],
                    "pnl": e["total_pnl"], "ret": e["return_pct"],
                    "trades": e["total_trades"], "dd": e["max_drawdown_pct"],
                    "eq": e["final_equity"],
                } for e in entries},
            })

    consistent.sort(key=lambda x: (x["n_pairs"], x["total_pnl"]), reverse=True)

    print(f"\n{'='*70}", flush=True)
    print("CROSS-PAIR CONFIGS (profitable on 3+ pairs)", flush=True)
    print(f"{'='*70}", flush=True)
    for i, c in enumerate(consistent[:20], 1):
        p = c["params"]
        print(f"  #{i:2d} | {c['n_pairs']} pairs | PF={c['avg_pf']:.2f} WR={c['avg_wr']*100:.1f}% "
              f"PnL=${c['total_pnl']:+,.2f} ({c['total_return_pct']:+.1f}%) "
              f"trades={c['total_trades']} DD={c['worst_dd']*100:.1f}%", flush=True)
        print(f"      Lev={p['leverage']}x SL={p['sl_pct']*100:.1f}% TP={p['tp_pct']*100:.1f}% "
              f"Trail={p['trailing_pct']*100:.1f}% CD={p['cooldown_bars']} "
              f"Pos={p['position_pct']*100:.1f}% MinScore={p['min_score']}", flush=True)

    # ============================================================
    # LEVERAGE ANALYSIS
    # ============================================================
    print(f"\n{'='*70}", flush=True)
    print("LEVERAGE ANALYSIS (best cross-pair config per leverage)", flush=True)
    print(f"{'='*70}", flush=True)
    leverage_best = {}
    for c in consistent:
        lev = c["params"]["leverage"]
        if lev not in leverage_best or c["total_pnl"] > leverage_best[lev]["total_pnl"]:
            leverage_best[lev] = c

    for lev in sorted(leverage_best.keys()):
        c = leverage_best[lev]
        p = c["params"]
        print(f"  {lev:2d}x | {c['n_pairs']} pairs | PF={c['avg_pf']:.2f} "
              f"PnL=${c['total_pnl']:+,.2f} ({c['total_return_pct']:+.1f}%) "
              f"DD={c['worst_dd']*100:.1f}% trades={c['total_trades']} | "
              f"SL={p['sl_pct']*100:.1f}% TP={p['tp_pct']*100:.1f}% Trail={p['trailing_pct']*100:.1f}%",
              flush=True)

    # ============================================================
    # BEST CONFIG DETAIL
    # ============================================================
    if consistent:
        best = consistent[0]
        print(f"\n{'='*70}", flush=True)
        print("BEST CROSS-PAIR CONFIG — PER-PAIR BREAKDOWN", flush=True)
        print(f"{'='*70}", flush=True)
        p = best["params"]
        print(f"Leverage={p['leverage']}x SL={p['sl_pct']*100:.1f}% TP={p['tp_pct']*100:.1f}% "
              f"Trail={p['trailing_pct']*100:.1f}% CD={p['cooldown_bars']} "
              f"Pos={p['position_pct']*100:.1f}% MinScore={p['min_score']}", flush=True)
        print(f"Total PnL: ${best['total_pnl']:+,.2f} ({best['total_return_pct']:+.1f}% return on $50K)",
              flush=True)
        print(flush=True)

        best_key = json.dumps(best["params"], sort_keys=True)
        for r in all_results:
            if json.dumps(r["params"], sort_keys=True) == best_key:
                rr = abs(r["avg_win_pct"] / r["avg_loss_pct"]) if r["avg_loss_pct"] != 0 else 0
                print(f"  {r['pair']:10} | PF={r['profit_factor']:5.2f} WR={r['win_rate']*100:4.1f}% "
                      f"PnL=${r['total_pnl']:+10,.2f} ({r['return_pct']:+5.1f}%) | "
                      f"trades={r['total_trades']:3d} TP={r['tp_hits']} Trail={r['trail_wins']} "
                      f"SL={r['sl_hits']} | R:R={rr:.1f}:1 DD={r['max_drawdown_pct']*100:.1f}% "
                      f"hold={r['avg_hold']*5:.0f}min eq=${r['final_equity']:,.0f}", flush=True)

    # ============================================================
    # RISK/REWARD vs LEVERAGE TABLE
    # ============================================================
    print(f"\n{'='*70}", flush=True)
    print("RISK vs REWARD BY LEVERAGE (5-pair consistent configs only)", flush=True)
    print(f"{'='*70}", flush=True)

    five_pair = [c for c in consistent if c["n_pairs"] == 5]
    if five_pair:
        # Group by leverage
        by_lev = defaultdict(list)
        for c in five_pair:
            by_lev[c["params"]["leverage"]].append(c)

        print(f"  {'Lev':>3} | {'Configs':>7} | {'Best PnL':>12} | {'Best Ret%':>8} | "
              f"{'Avg PnL':>10} | {'Worst DD':>8} | {'Avg Trades':>10}", flush=True)
        print(f"  {'-'*80}", flush=True)

        for lev in sorted(by_lev.keys()):
            configs = by_lev[lev]
            best_pnl = max(c["total_pnl"] for c in configs)
            best_ret = max(c["total_return_pct"] for c in configs)
            avg_pnl = np.mean([c["total_pnl"] for c in configs])
            worst_dd = min(c["worst_dd"] for c in configs)
            avg_trades = np.mean([c["total_trades"] for c in configs])
            print(f"  {lev:3d}x | {len(configs):7d} | ${best_pnl:+11,.2f} | {best_ret:+7.1f}% | "
                  f"${avg_pnl:+9,.2f} | {worst_dd*100:7.1f}% | {avg_trades:10.0f}", flush=True)
    else:
        print("  No configs profitable on all 5 pairs.", flush=True)
        # Show 4-pair configs instead
        four_pair = [c for c in consistent if c["n_pairs"] >= 4]
        if four_pair:
            by_lev = defaultdict(list)
            for c in four_pair:
                by_lev[c["params"]["leverage"]].append(c)
            print(f"  Showing 4+ pair configs instead:", flush=True)
            for lev in sorted(by_lev.keys()):
                configs = by_lev[lev]
                best_pnl = max(c["total_pnl"] for c in configs)
                best_ret = max(c["total_return_pct"] for c in configs)
                worst_dd = min(c["worst_dd"] for c in configs)
                print(f"  {lev:3d}x | {len(configs)} configs | best=${best_pnl:+,.2f} ({best_ret:+.1f}%) "
                      f"DD={worst_dd*100:.1f}%", flush=True)

    # Save results
    output = {
        "config": {
            "initial_equity": INITIAL_EQUITY,
            "days": DAYS,
            "strategy": STRATEGY,
            "pairs": list(PAIRS),
            "total_param_combos": len(PARAM_GRID),
        },
        "summary": {
            "total_qualified": len(all_results),
            "total_profitable": len(profitable),
            "total_sims": sim_count,
            "elapsed_s": round(time.time() - t0, 1),
        },
        "top_results": all_results[:200],
        "cross_pair": consistent[:50],
        "leverage_analysis": {str(k): {
            "params": v["params"],
            "total_pnl": v["total_pnl"],
            "return_pct": v["total_return_pct"],
            "avg_pf": v["avg_pf"],
            "n_pairs": v["n_pairs"],
            "worst_dd": v["worst_dd"],
        } for k, v in leverage_best.items()} if leverage_best else {},
    }

    results_file = Path(__file__).parent / "results_3month_50k.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}", flush=True)
    print(f"Total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
