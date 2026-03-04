#!/usr/bin/env python3
"""2-month backtest: macd_cross + mean_reversion with their winning params."""
import sys, time, json, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.data_loader import download_klines, load_proxy_url
from backtest.indicators import compute_all_indicators
from backtest.run_optimized import detect_all_signals, simulate_trades, INITIAL_EQUITY, FEE_PCT

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
DAYS = 60
INTERVAL = "5m"

# Only the 2 winning strategies
STRATEGIES = ["macd_cross", "mean_reversion"]

# Winning param ranges from 14d backtest
PARAM_GRID = []
from itertools import product
for sl, tp, tr, cd, lev in product(
    [0.008, 0.012, 0.015, 0.020, 0.025],
    [0.015, 0.020, 0.030, 0.040, 0.060],
    [0.015, 0.020, 0.025],
    [5, 10, 20],
    [2, 3],
):
    if tp <= sl: continue
    PARAM_GRID.append({"sl_pct":sl,"tp_pct":tp,"trailing_pct":tr,"cooldown_bars":cd,"leverage":lev,"min_score":0})
    PARAM_GRID.append({"sl_pct":sl,"tp_pct":tp,"trailing_pct":tr,"cooldown_bars":cd,"leverage":lev,"min_score":4})

print(f"Param combos: {len(PARAM_GRID)}")

def run_pair(pair, df):
    signals = detect_all_signals(df)
    results = []
    for strat in STRATEGIES:
        strat_sigs = [s for s in signals if s["strat"] == strat]
        if not strat_sigs: continue
        for p in PARAM_GRID:
            pp = dict(p); pp["strategy_filter"] = strat
            r = simulate_trades(df, strat_sigs, pp)
            if r and r["total_trades"] >= 10:
                r["strategy"] = strat; r["pair"] = pair; r["params"] = {k:v for k,v in p.items()}
                results.append(r)
    return results

def main():
    t0 = time.time()
    proxy_url = load_proxy_url()
    print(f"{'='*70}\n2-MONTH BACKTEST: macd_cross + mean_reversion\n{'='*70}")
    print(f"Pairs: {PAIRS}\nDays: {DAYS}\nParams: {len(PARAM_GRID)} combos\n")

    # Download
    data = {}
    for pair in PAIRS:
        print(f"  {pair}...", end=" ", flush=True)
        df = download_klines(pair, interval=INTERVAL, days=DAYS, proxy_url=proxy_url)
        if df.empty: print("FAIL"); continue
        df = compute_all_indicators(df)
        data[pair] = df
        print(f"{len(df)} candles")

    # Run
    all_results = []
    for pair, df in data.items():
        t1 = time.time()
        results = run_pair(pair, df)
        e = time.time()-t1
        profitable = sum(1 for r in results if r["profit_factor"]>1.0)
        print(f"  {pair}: {len(results)} qualified, {profitable} profitable ({e:.1f}s)")
        all_results.extend(results)

    all_results.sort(key=lambda r: (r["profit_factor"], r["total_pnl"]), reverse=True)
    profitable = [r for r in all_results if r["profit_factor"]>1.0]

    print(f"\n{'='*70}\nRESULTS (2 MONTHS)\n{'='*70}")
    print(f"Total qualified: {len(all_results)}")
    print(f"Profitable: {len(profitable)}")

    print(f"\n{'#':>3} | {'Strat':18} | {'Pair':10} | {'PF':>5} | {'WR':>5} | {'PnL':>10} | {'Trd':>4} | {'TP%':>4} | {'DD':>6} | {'Sharpe':>6} | {'SL':>5} | {'TP':>5} | {'Trail':>5} | {'Lev':>3}")
    print("-"*130)
    for i, r in enumerate(all_results[:40], 1):
        tp_rate = r["tp_hits"]/r["total_trades"]*100 if r["total_trades"]>0 else 0
        p = r["params"]
        print(f"{i:3d} | {r['strategy']:18} | {r['pair']:10} | {r['profit_factor']:5.2f} | {r['win_rate']*100:4.1f}% | ${r['total_pnl']:+9.2f} | {r['total_trades']:4d} | {tp_rate:3.0f}% | {r['max_drawdown_pct']*100:5.1f}% | {r['sharpe_ratio']:6.2f} | {p['sl_pct']*100:4.1f}% | {p['tp_pct']*100:4.1f}% | {p['trailing_pct']*100:4.1f}% | {p['leverage']}x")

    # Cross-pair consistency
    from collections import defaultdict
    strat_cross = defaultdict(list)
    for r in all_results:
        if r["profit_factor"]>1.0:
            key = (r["strategy"], json.dumps(r["params"], sort_keys=True))
            strat_cross[key].append(r)
    consistent = []
    for (strat, ps), entries in strat_cross.items():
        pairs_ok = set(e["pair"] for e in entries)
        if len(pairs_ok) >= 3:
            consistent.append({
                "strategy":strat, "params":json.loads(ps), "n_pairs":len(pairs_ok),
                "pairs":sorted(pairs_ok),
                "total_pnl":sum(e["total_pnl"] for e in entries),
                "avg_pf":np.mean([e["profit_factor"] for e in entries]),
                "avg_wr":np.mean([e["win_rate"] for e in entries]),
                "total_trades":sum(e["total_trades"] for e in entries),
                "max_dd":min(e["max_drawdown_pct"] for e in entries),
            })
    consistent.sort(key=lambda x: (x["n_pairs"], x["total_pnl"]), reverse=True)

    print(f"\n{'='*70}\nCROSS-PAIR (profitable on 3+ pairs)\n{'='*70}")
    for i, c in enumerate(consistent[:15], 1):
        print(f"  #{i} {c['strategy']:18} | {c['n_pairs']} pairs | PF={c['avg_pf']:.2f} WR={c['avg_wr']*100:.1f}% PnL=${c['total_pnl']:+.2f} trades={c['total_trades']} DD={c['max_dd']*100:.1f}% | {','.join(c['pairs'])}")
        if i <= 5:
            p = c["params"]
            print(f"     SL={p['sl_pct']*100:.1f}% TP={p['tp_pct']*100:.1f}% Trail={p['trailing_pct']*100:.1f}% CD={p['cooldown_bars']} Lev={p['leverage']}x MinScore={p['min_score']}")

    # Per-strategy per-pair detail for best config
    if consistent:
        best = consistent[0]
        print(f"\n{'='*70}\nBEST CROSS-PAIR CONFIG — DETAILED BREAKDOWN\n{'='*70}")
        p = best["params"]
        print(f"Strategy: {best['strategy']}")
        print(f"SL={p['sl_pct']*100:.1f}% TP={p['tp_pct']*100:.1f}% Trail={p['trailing_pct']*100:.1f}% CD={p['cooldown_bars']} Lev={p['leverage']}x\n")
        for r in all_results:
            if r["strategy"]==best["strategy"] and json.dumps(r["params"],sort_keys=True)==json.dumps(best["params"],sort_keys=True):
                rr = abs(r["avg_win_pct"]/r["avg_loss_pct"]) if r["avg_loss_pct"]!=0 else 0
                trail_wins = r["wins"]-r["tp_hits"]
                print(f"  {r['pair']:10} | PF={r['profit_factor']:5.2f} WR={r['win_rate']*100:4.1f}% PnL=${r['total_pnl']:+8.2f} trades={r['total_trades']:3d} | TP={r['tp_hits']} Trail={trail_wins} SL={r['sl_hits']} | R:R={rr:.1f}:1 DD={r['max_drawdown_pct']*100:.1f}% hold={r['avg_hold']*5:.0f}min")

    # Save
    with open(Path(__file__).parent/"results_2month.json","w") as f:
        json.dump({"top_results":all_results[:100],"cross_pair":consistent[:30]}, f, indent=2, default=str)

    print(f"\nTotal time: {time.time()-t0:.1f}s")

if __name__=="__main__":
    main()
