#!/usr/bin/env python3
"""
12-month PRODUCTION SIMULATION: macd_cross strategy
- 20 pairs, $50K, leverage 3x-15x
- Real maker/taker fees + slippage
- Monthly compound interest tracking
- Accepts pairs as CLI args for parallel execution
"""
import sys, time, json, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.data_loader import download_klines, load_proxy_url
from backtest.indicators import compute_all_indicators
from backtest.run_optimized import detect_all_signals

# ============================================================
# CONFIG
# ============================================================
ALL_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "MATICUSDT", "SUIUSDT", "ARBUSDT", "OPUSDT", "APTUSDT",
    "NEARUSDT", "LTCUSDT", "ATOMUSDT", "FILUSDT", "INJUSDT",
]
PAIRS = sys.argv[1:] if len(sys.argv) > 1 else ALL_PAIRS
DAYS = 365
INTERVAL = "5m"
INITIAL_EQUITY = 50_000.0
STRATEGY = "macd_cross"

# Real Binance Futures fees
MAKER_FEE = 0.0002   # 0.02%
TAKER_FEE = 0.0005   # 0.05%
SLIPPAGE_MAJOR = 0.0001  # 0.01% for BTC/ETH
SLIPPAGE_ALT = 0.0003    # 0.03% for alts
MAJOR_PAIRS = {"BTCUSDT", "ETHUSDT"}

# ============================================================
# PRODUCTION CONFIGS (winners from 3-month backtest)
# ============================================================
CONFIGS = [
    {"name": "conservative", "sl_pct": 0.010, "tp_pct": 0.030, "trailing_pct": 0.030, "cooldown_bars": 5},
    {"name": "balanced",     "sl_pct": 0.010, "tp_pct": 0.080, "trailing_pct": 0.030, "cooldown_bars": 5},
    {"name": "wide_tp",      "sl_pct": 0.006, "tp_pct": 0.100, "trailing_pct": 0.025, "cooldown_bars": 5},
    {"name": "2mo_winner",   "sl_pct": 0.008, "tp_pct": 0.060, "trailing_pct": 0.025, "cooldown_bars": 5},
]

LEVERAGES = [3, 5, 6, 8, 10, 12, 15]
POS_SIZES = [0.005, 0.010]
MIN_SCORES = [0, 4]

# Build param grid
PARAM_GRID = []
for cfg in CONFIGS:
    for lev in LEVERAGES:
        for pos in POS_SIZES:
            for ms in MIN_SCORES:
                PARAM_GRID.append({
                    "config_name": cfg["name"],
                    "sl_pct": cfg["sl_pct"],
                    "tp_pct": cfg["tp_pct"],
                    "trailing_pct": cfg["trailing_pct"],
                    "cooldown_bars": cfg["cooldown_bars"],
                    "leverage": lev,
                    "position_pct": pos,
                    "min_score": ms,
                })

print(f"Configs: {len(CONFIGS)} x {len(LEVERAGES)} leverages x {len(POS_SIZES)} pos_sizes x {len(MIN_SCORES)} min_scores = {len(PARAM_GRID)} combos", flush=True)


# ============================================================
# FEE CALCULATION
# ============================================================
def calc_fees(pair, notional, exit_reason):
    """Calculate real entry + exit fees including slippage."""
    slippage = SLIPPAGE_MAJOR if pair in MAJOR_PAIRS else SLIPPAGE_ALT
    # Entry: always market order (taker + slippage)
    entry_fee = notional * (TAKER_FEE + slippage)
    # Exit: TP = limit order (maker, no slippage), SL/trail = stop-market (taker + slippage)
    if exit_reason == "tp":
        exit_fee = notional * MAKER_FEE
    else:
        exit_fee = notional * (TAKER_FEE + slippage)
    return entry_fee + exit_fee


# ============================================================
# PRODUCTION SIMULATION WITH MONTHLY TRACKING
# ============================================================
def simulate_production(df, signals, params, pair):
    """Simulate production trading with monthly compound interest tracking."""
    sl_pct = params["sl_pct"]
    tp_pct = params["tp_pct"]
    trailing_pct = params["trailing_pct"]
    cooldown = params["cooldown_bars"]
    min_score = params["min_score"]
    leverage = params["leverage"]
    pos_pct = params["position_pct"]

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    timestamps = df["timestamp"].values
    n_bars = len(df)

    equity = INITIAL_EQUITY
    peak_equity = equity
    max_dd = 0.0
    trades = []
    last_exit_bar = -cooldown - 1

    # Monthly tracking
    monthly = defaultdict(lambda: {
        "start_equity": 0, "end_equity": 0,
        "trades": 0, "wins": 0, "pnl": 0.0,
        "fees_paid": 0.0,
    })

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

        # Skip tiny positions
        if notional < 10:
            continue

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

        # P&L calculation with REAL fees
        if direction == "LONG":
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        fee_cost = calc_fees(pair, notional, exit_reason)
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
                "direction": direction, "fees": fee_cost,
            })
            equity = 0
            break

        last_exit_bar = exit_bar

        # Track monthly (by entry timestamp)
        ts = pd.Timestamp(timestamps[bar])
        month_key = ts.strftime("%Y-%m")
        m = monthly[month_key]
        if m["start_equity"] == 0:
            m["start_equity"] = equity - pnl_usd  # equity before this trade
        m["end_equity"] = equity
        m["trades"] += 1
        m["pnl"] += pnl_usd
        m["fees_paid"] += fee_cost
        if pnl_usd > 0:
            m["wins"] += 1

        trades.append({
            "pnl_usd": pnl_usd, "pnl_pct": pnl_pct,
            "exit_reason": exit_reason, "hold_bars": exit_bar - bar,
            "direction": direction, "fees": fee_cost, "month": month_key,
        })

    if not trades:
        return None

    # Fix monthly end_equity for months with no trades
    sorted_months = sorted(monthly.keys())
    for i, mk in enumerate(sorted_months):
        if i == 0 and monthly[mk]["start_equity"] == 0:
            monthly[mk]["start_equity"] = INITIAL_EQUITY

    # Compute stats
    pnls = [t["pnl_usd"] for t in trades]
    total_fees = sum(t["fees"] for t in trades)
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

    # Monthly returns (compound)
    monthly_returns = []
    for mk in sorted_months:
        m = monthly[mk]
        ret_pct = (m["pnl"] / m["start_equity"] * 100) if m["start_equity"] > 0 else 0
        monthly_returns.append({
            "month": mk,
            "start_equity": round(m["start_equity"], 2),
            "end_equity": round(m["end_equity"], 2),
            "pnl": round(m["pnl"], 2),
            "return_pct": round(ret_pct, 2),
            "trades": m["trades"],
            "wins": m["wins"],
            "win_rate": round(m["wins"] / m["trades"] * 100, 1) if m["trades"] > 0 else 0,
            "fees_paid": round(m["fees_paid"], 2),
        })

    return {
        "total_trades": len(trades),
        "wins": wins,
        "win_rate": round(wins / len(trades), 4),
        "total_pnl": round(sum(pnls), 2),
        "total_fees": round(total_fees, 2),
        "profit_factor": round(pf, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "sharpe_ratio": round(sharpe, 4),
        "avg_win_pct": round(np.mean(win_pcts), 6) if win_pcts else 0,
        "avg_loss_pct": round(np.mean(loss_pcts), 6) if loss_pcts else 0,
        "avg_hold_bars": round(np.mean([t["hold_bars"] for t in trades]), 1),
        "tp_hits": tp_hits,
        "sl_hits": sl_hits,
        "trail_wins": trail_wins,
        "final_equity": round(equity, 2),
        "return_pct": round((equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100, 2),
        "monthly": monthly_returns,
    }


def main():
    t0 = time.time()
    proxy_url = load_proxy_url()

    print("=" * 80, flush=True)
    print("12-MONTH PRODUCTION SIMULATION — macd_cross", flush=True)
    print(f"$50K | 20 pairs | Leverage 3x-15x | Real fees | Compound interest", flush=True)
    print("=" * 80, flush=True)
    print(f"Pairs: {PAIRS}", flush=True)
    print(f"Days: {DAYS} | Interval: {INTERVAL}", flush=True)
    print(f"Param combos: {len(PARAM_GRID)}", flush=True)
    print(f"Fees: Maker={MAKER_FEE*100:.2f}% Taker={TAKER_FEE*100:.2f}% "
          f"Slippage={SLIPPAGE_MAJOR*100:.2f}%/{SLIPPAGE_ALT*100:.2f}%", flush=True)
    print(flush=True)

    # ============================================================
    # STEP 1: Download data
    # ============================================================
    print("STEP 1: Downloading 12 months of data...", flush=True)
    data = {}
    failed_pairs = []
    for pair in PAIRS:
        print(f"  {pair}...", end=" ", flush=True)
        try:
            df = download_klines(pair, interval=INTERVAL, days=DAYS, proxy_url=proxy_url)
            if df.empty or len(df) < 1000:
                print(f"FAIL ({len(df) if not df.empty else 0} candles)", flush=True)
                failed_pairs.append(pair)
                continue
            df = compute_all_indicators(df)
            data[pair] = df
            # Calculate date range
            ts_first = pd.Timestamp(df["timestamp"].iloc[0])
            ts_last = pd.Timestamp(df["timestamp"].iloc[-1])
            months = (ts_last - ts_first).days / 30
            print(f"{len(df)} candles ({months:.1f} months: {ts_first.strftime('%Y-%m-%d')} to {ts_last.strftime('%Y-%m-%d')})", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            failed_pairs.append(pair)

    if not data:
        print("ERROR: No data downloaded!", flush=True)
        sys.exit(1)

    if failed_pairs:
        print(f"\n  Failed pairs: {failed_pairs}", flush=True)
    print(f"\n  Successfully loaded: {len(data)} pairs", flush=True)

    # ============================================================
    # STEP 2: Detect signals
    # ============================================================
    print(f"\nSTEP 2: Detecting macd_cross signals...", flush=True)
    pair_signals = {}
    for pair, df in data.items():
        t1 = time.time()
        signals = detect_all_signals(df)
        macd_sigs = [s for s in signals if s["strat"] == STRATEGY]
        elapsed = time.time() - t1
        print(f"  {pair}: {len(macd_sigs)} signals ({elapsed:.1f}s)", flush=True)
        pair_signals[pair] = macd_sigs

    # ============================================================
    # STEP 3: Run simulations
    # ============================================================
    total_sims = len(PARAM_GRID) * len(data)
    print(f"\nSTEP 3: Running {total_sims:,} production simulations...", flush=True)

    all_results = []
    sim_count = 0
    t_sim = time.time()

    for pair, df in data.items():
        sigs = pair_signals[pair]
        if not sigs:
            print(f"  {pair}: 0 signals, skipping", flush=True)
            continue

        t_pair = time.time()
        pair_results = []

        for params in PARAM_GRID:
            result = simulate_production(df, sigs, params, pair)
            sim_count += 1

            if result and result["total_trades"] >= 5:
                result["pair"] = pair
                result["config_name"] = params["config_name"]
                result["params"] = {
                    "sl_pct": params["sl_pct"],
                    "tp_pct": params["tp_pct"],
                    "trailing_pct": params["trailing_pct"],
                    "cooldown_bars": params["cooldown_bars"],
                    "leverage": params["leverage"],
                    "position_pct": params["position_pct"],
                    "min_score": params["min_score"],
                }
                pair_results.append(result)

        elapsed_pair = time.time() - t_pair
        profitable = sum(1 for r in pair_results if r["profit_factor"] > 1.0)
        best_pnl = max((r["total_pnl"] for r in pair_results), default=0)
        best_ret = max((r["return_pct"] for r in pair_results), default=0)
        total_fees = max((r["total_fees"] for r in pair_results), default=0)
        print(f"  {pair}: {len(pair_results)} qualified, {profitable} profitable, "
              f"best=${best_pnl:+,.2f} ({best_ret:+.1f}%) fees_max=${total_fees:,.0f} ({elapsed_pair:.1f}s)", flush=True)
        all_results.extend(pair_results)

    elapsed_sim = time.time() - t_sim
    print(f"\n  {sim_count:,} sims in {elapsed_sim:.1f}s ({sim_count/elapsed_sim:.0f} sims/s)", flush=True)

    # ============================================================
    # STEP 4: RESULTS
    # ============================================================
    all_results.sort(key=lambda r: (r["profit_factor"], r["total_pnl"]), reverse=True)
    profitable = [r for r in all_results if r["profit_factor"] > 1.0]

    print(f"\n{'='*80}", flush=True)
    print(f"RESULTS — 12-MONTH PRODUCTION SIMULATION ($50K)", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Pairs tested: {len(data)}", flush=True)
    print(f"Total qualified configs: {len(all_results):,}", flush=True)
    print(f"Profitable configs (PF>1): {len(profitable):,} ({len(profitable)/len(all_results)*100:.0f}%)", flush=True)

    # ============================================================
    # TOP 40
    # ============================================================
    print(f"\n{'#':>3} | {'Config':14} | {'Pair':10} | {'PF':>5} | {'WR':>5} | {'PnL':>12} | "
          f"{'Ret%':>7} | {'Trd':>4} | {'DD':>6} | {'Fees':>8} | {'Lev':>3} | {'Pos%':>4}", flush=True)
    print("-" * 120, flush=True)
    for i, r in enumerate(all_results[:40], 1):
        p = r["params"]
        print(f"{i:3d} | {r['config_name']:14} | {r['pair']:10} | {r['profit_factor']:5.2f} | "
              f"{r['win_rate']*100:4.1f}% | ${r['total_pnl']:+11,.2f} | {r['return_pct']:+6.1f}% | "
              f"{r['total_trades']:4d} | {r['max_drawdown_pct']*100:5.1f}% | ${r['total_fees']:7,.0f} | "
              f"{p['leverage']:2d}x | {p['position_pct']*100:.1f}%", flush=True)

    # ============================================================
    # PER-PAIR RANKING (best config per pair)
    # ============================================================
    print(f"\n{'='*80}", flush=True)
    print("PER-PAIR RANKING (best config for each pair, sorted by PnL)", flush=True)
    print(f"{'='*80}", flush=True)

    pair_best = {}
    for r in all_results:
        p = r["pair"]
        if p not in pair_best or r["total_pnl"] > pair_best[p]["total_pnl"]:
            pair_best[p] = r

    sorted_pairs = sorted(pair_best.items(), key=lambda x: x[1]["total_pnl"], reverse=True)
    profitable_pairs = []
    unprofitable_pairs = []

    for pair, r in sorted_pairs:
        p = r["params"]
        status = "KEEP" if r["profit_factor"] > 1.0 else "DROP"
        marker = "+" if status == "KEEP" else "-"
        if status == "KEEP":
            profitable_pairs.append(pair)
        else:
            unprofitable_pairs.append(pair)

        # Monthly avg return
        monthly_rets = [m["return_pct"] for m in r["monthly"]]
        avg_monthly = np.mean(monthly_rets) if monthly_rets else 0
        positive_months = sum(1 for m in monthly_rets if m > 0)

        print(f"  [{marker}] {pair:10} | {r['config_name']:14} | PF={r['profit_factor']:5.2f} "
              f"PnL=${r['total_pnl']:+10,.2f} ({r['return_pct']:+5.1f}%) | "
              f"trades={r['total_trades']:4d} WR={r['win_rate']*100:4.1f}% DD={r['max_drawdown_pct']*100:4.1f}% | "
              f"Lev={p['leverage']}x Pos={p['position_pct']*100:.1f}% | "
              f"avg_mo={avg_monthly:+.2f}% green_mo={positive_months}/{len(monthly_rets)}", flush=True)

    print(f"\n  KEEP ({len(profitable_pairs)}): {', '.join(profitable_pairs)}", flush=True)
    print(f"  DROP ({len(unprofitable_pairs)}): {', '.join(unprofitable_pairs)}", flush=True)

    # ============================================================
    # BEST CONFIG — MONTHLY BREAKDOWN
    # ============================================================
    if all_results:
        # Find best cross-pair config
        # Group by config params (excluding pair)
        config_cross = defaultdict(list)
        for r in all_results:
            if r["profit_factor"] > 1.0:
                key = json.dumps(r["params"], sort_keys=True)
                config_cross[key].append(r)

        # Find config profitable on most pairs
        cross_configs = []
        for params_str, entries in config_cross.items():
            pairs_ok = set(e["pair"] for e in entries)
            total_pnl = sum(e["total_pnl"] for e in entries)
            avg_pf = np.mean([e["profit_factor"] for e in entries])
            total_trades = sum(e["total_trades"] for e in entries)
            total_fees = sum(e["total_fees"] for e in entries)
            worst_dd = min(e["max_drawdown_pct"] for e in entries)

            # Aggregate monthly returns across all pairs
            monthly_agg = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0, "fees": 0})
            for e in entries:
                for m in e["monthly"]:
                    monthly_agg[m["month"]]["pnl"] += m["pnl"]
                    monthly_agg[m["month"]]["trades"] += m["trades"]
                    monthly_agg[m["month"]]["wins"] += m["wins"]
                    monthly_agg[m["month"]]["fees"] += m["fees_paid"]

            cross_configs.append({
                "params": json.loads(params_str),
                "config_name": entries[0]["config_name"],
                "n_pairs": len(pairs_ok),
                "pairs": sorted(pairs_ok),
                "total_pnl": total_pnl,
                "total_return_pct": total_pnl / INITIAL_EQUITY * 100,
                "avg_pf": avg_pf,
                "total_trades": total_trades,
                "total_fees": total_fees,
                "worst_dd": worst_dd,
                "monthly_agg": dict(monthly_agg),
            })

        cross_configs.sort(key=lambda x: (x["n_pairs"], x["total_pnl"]), reverse=True)

        if cross_configs:
            print(f"\n{'='*80}", flush=True)
            print("TOP 10 CROSS-PAIR CONFIGS (profitable on most pairs)", flush=True)
            print(f"{'='*80}", flush=True)
            for i, c in enumerate(cross_configs[:10], 1):
                p = c["params"]
                print(f"  #{i:2d} | {c['config_name']:14} | {c['n_pairs']} pairs | "
                      f"PF={c['avg_pf']:.2f} PnL=${c['total_pnl']:+,.2f} ({c['total_return_pct']:+.1f}%) | "
                      f"trades={c['total_trades']} fees=${c['total_fees']:,.0f} DD={c['worst_dd']*100:.1f}%", flush=True)
                print(f"       Lev={p['leverage']}x SL={p['sl_pct']*100:.1f}% TP={p['tp_pct']*100:.1f}% "
                      f"Trail={p['trailing_pct']*100:.1f}% Pos={p['position_pct']*100:.1f}%", flush=True)

            # Best cross-pair config — monthly compound breakdown
            best = cross_configs[0]
            print(f"\n{'='*80}", flush=True)
            print(f"MONTHLY COMPOUND BREAKDOWN — BEST CONFIG", flush=True)
            print(f"{'='*80}", flush=True)
            p = best["params"]
            print(f"Config: {best['config_name']} | {best['n_pairs']} pairs | "
                  f"Lev={p['leverage']}x SL={p['sl_pct']*100:.1f}% TP={p['tp_pct']*100:.1f}% "
                  f"Trail={p['trailing_pct']*100:.1f}% Pos={p['position_pct']*100:.1f}%", flush=True)
            print(f"Pairs: {', '.join(best['pairs'])}", flush=True)
            print(flush=True)

            # Simulate compound monthly returns
            compound_equity = INITIAL_EQUITY
            print(f"  {'Month':>7} | {'Equity Start':>14} | {'PnL':>10} | {'Fees':>8} | "
                  f"{'Net PnL':>10} | {'Return%':>8} | {'Equity End':>14} | {'Trades':>6} | {'WR':>5}", flush=True)
            print(f"  {'-'*105}", flush=True)

            total_compound_pnl = 0
            sorted_months = sorted(best["monthly_agg"].keys())
            for mk in sorted_months:
                m = best["monthly_agg"][mk]
                monthly_return = m["pnl"] / compound_equity * 100 if compound_equity > 0 else 0
                wr = m["wins"] / m["trades"] * 100 if m["trades"] > 0 else 0
                end_eq = compound_equity + m["pnl"]
                total_compound_pnl += m["pnl"]

                print(f"  {mk:>7} | ${compound_equity:>13,.2f} | ${m['pnl']:>+9,.2f} | "
                      f"${m['fees']:>7,.2f} | ${m['pnl']:>+9,.2f} | {monthly_return:>+7.2f}% | "
                      f"${end_eq:>13,.2f} | {m['trades']:>6} | {wr:4.0f}%", flush=True)
                compound_equity = end_eq

            print(f"  {'-'*105}", flush=True)
            total_return = (compound_equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100
            avg_monthly_return = total_return / len(sorted_months) if sorted_months else 0
            green_months = sum(1 for mk in sorted_months if best["monthly_agg"][mk]["pnl"] > 0)
            print(f"\n  Starting Capital:     ${INITIAL_EQUITY:>13,.2f}", flush=True)
            print(f"  Final Capital:        ${compound_equity:>13,.2f}", flush=True)
            print(f"  Total P&L:            ${total_compound_pnl:>+13,.2f}", flush=True)
            print(f"  Total Return:         {total_return:>+13.2f}%", flush=True)
            print(f"  Avg Monthly Return:   {avg_monthly_return:>+13.2f}%", flush=True)
            print(f"  Green Months:         {green_months}/{len(sorted_months)}", flush=True)
            print(f"  Total Trades:         {best['total_trades']:>13,}", flush=True)
            print(f"  Total Fees Paid:      ${best['total_fees']:>13,.2f}", flush=True)

    # ============================================================
    # LEVERAGE ANALYSIS
    # ============================================================
    if cross_configs:
        print(f"\n{'='*80}", flush=True)
        print("LEVERAGE COMPARISON (best cross-pair config per leverage)", flush=True)
        print(f"{'='*80}", flush=True)

        lev_best = {}
        for c in cross_configs:
            lev = c["params"]["leverage"]
            if lev not in lev_best or c["total_pnl"] > lev_best[lev]["total_pnl"]:
                lev_best[lev] = c

        print(f"  {'Lev':>3} | {'Pairs':>5} | {'PF':>5} | {'Total PnL':>12} | {'Return%':>8} | "
              f"{'Worst DD':>8} | {'Trades':>6} | {'Fees':>8} | {'Config':>14}", flush=True)
        print(f"  {'-'*90}", flush=True)
        for lev in sorted(lev_best.keys()):
            c = lev_best[lev]
            print(f"  {lev:3d}x | {c['n_pairs']:5d} | {c['avg_pf']:5.2f} | ${c['total_pnl']:+11,.2f} | "
                  f"{c['total_return_pct']:+7.1f}% | {c['worst_dd']*100:7.1f}% | "
                  f"{c['total_trades']:6d} | ${c['total_fees']:7,.0f} | {c['config_name']:>14}", flush=True)

    # ============================================================
    # CONFIG COMPARISON
    # ============================================================
    print(f"\n{'='*80}", flush=True)
    print("CONFIG COMPARISON (best leverage per config name)", flush=True)
    print(f"{'='*80}", flush=True)

    for cfg_name in [c["name"] for c in CONFIGS]:
        cfg_results = [c for c in cross_configs if c["config_name"] == cfg_name]
        if not cfg_results:
            print(f"  {cfg_name:14} | No profitable cross-pair configs", flush=True)
            continue
        best_cfg = max(cfg_results, key=lambda x: x["total_pnl"])
        p = best_cfg["params"]
        print(f"  {cfg_name:14} | {best_cfg['n_pairs']} pairs | PF={best_cfg['avg_pf']:.2f} "
              f"PnL=${best_cfg['total_pnl']:+,.2f} ({best_cfg['total_return_pct']:+.1f}%) "
              f"DD={best_cfg['worst_dd']*100:.1f}% Lev={p['leverage']}x", flush=True)

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    output = {
        "config": {
            "initial_equity": INITIAL_EQUITY,
            "days": DAYS,
            "strategy": STRATEGY,
            "pairs_tested": list(data.keys()),
            "pairs_failed": failed_pairs,
            "fees": {"maker": MAKER_FEE, "taker": TAKER_FEE,
                     "slippage_major": SLIPPAGE_MAJOR, "slippage_alt": SLIPPAGE_ALT},
            "param_combos": len(PARAM_GRID),
        },
        "summary": {
            "total_qualified": len(all_results),
            "total_profitable": len(profitable),
            "profitable_pairs": profitable_pairs,
            "unprofitable_pairs": unprofitable_pairs,
            "total_sims": sim_count,
            "elapsed_s": round(time.time() - t0, 1),
        },
        "per_pair": {pair: {
            "config_name": r["config_name"],
            "profit_factor": r["profit_factor"],
            "total_pnl": r["total_pnl"],
            "return_pct": r["return_pct"],
            "total_trades": r["total_trades"],
            "win_rate": r["win_rate"],
            "max_drawdown_pct": r["max_drawdown_pct"],
            "params": r["params"],
            "monthly": r["monthly"],
        } for pair, r in pair_best.items()} if pair_best else {},
        "top_results": all_results[:100],
        "cross_pair": cross_configs[:30] if cross_configs else [],
    }

    results_file = Path(__file__).parent / "results_12month.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}", flush=True)
    print(f"Total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
