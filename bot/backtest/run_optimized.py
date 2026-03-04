#!/usr/bin/env python3
"""
Optimized backtesting: pre-compute signals once, then simulate trades with different SL/TP params.
Signal detection is expensive (row-by-row), trade simulation is cheap (only at signal bars).
"""

import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.data_loader import download_klines, load_proxy_url
from backtest.indicators import compute_all_indicators

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
DAYS = 14
INTERVAL = "5m"
FEE_PCT = 0.0005
INITIAL_EQUITY = 5000.0


# ============================================================
# SIGNAL DETECTION (vectorized where possible, row-by-row where needed)
# ============================================================

def _safe(val):
    """Check if a value is a finite number."""
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return None
    return val


def detect_all_signals(df: pd.DataFrame) -> list[dict]:
    """Pre-compute ALL entry signals across ALL strategies for a DataFrame.
    Returns list of {bar_idx, direction, strategy, score} sorted by bar_idx."""
    signals = []
    n = len(df)

    for i in range(1, n):  # start at 1 to have prev bar
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        rsi_14 = _safe(row.get("rsi_14"))
        adx = _safe(row.get("adx"))
        ema_align = _safe(row.get("ema_alignment"))
        plus_di = _safe(row.get("plus_di"))
        minus_di = _safe(row.get("minus_di"))
        macd_sig = row.get("macd_signal")
        bb_pct = _safe(row.get("bb_pct"))
        stoch_k = _safe(row.get("stoch_rsi_k"))
        volume_ratio = _safe(row.get("volume_ratio")) or 0
        bb_squeeze = row.get("bb_squeeze")
        prev_bb_squeeze = prev.get("bb_squeeze")
        mfi = _safe(row.get("mfi"))
        ema_5m = row.get("ema_trend_5m")
        williams_r = _safe(row.get("williams_r"))
        cci = _safe(row.get("cci"))
        tenkan = _safe(row.get("ichimoku_tenkan"))
        kijun = _safe(row.get("ichimoku_kijun"))
        senkou_a = _safe(row.get("ichimoku_senkou_a"))
        senkou_b = _safe(row.get("ichimoku_senkou_b"))
        price = row["close"]
        prev_rsi = _safe(prev.get("rsi_14"))
        prev_stoch = _safe(prev.get("stoch_rsi_k"))
        prev_ema9 = _safe(prev.get("ema_9"))
        prev_ema21 = _safe(prev.get("ema_21"))
        ema_9 = _safe(row.get("ema_9"))
        ema_21 = _safe(row.get("ema_21"))

        # --- 1. TREND FOLLOW ---
        if adx and ema_align and plus_di and minus_di and rsi_14 and macd_sig:
            # LONG
            if (adx > 25 and ema_align > 0.3 and plus_di > minus_di
                    and 30 <= rsi_14 <= 60 and macd_sig in ("bullish", "bullish_cross")):
                score = 0
                if adx > 25: score += 2
                if adx > 35: score += 1
                if ema_align > 0.5: score += 1
                if macd_sig == "bullish_cross": score += 1
                if ema_5m in ("bullish", "strong_bullish"): score += 2
                if volume_ratio > 1.0: score += 1
                signals.append({"bar": i, "dir": "LONG", "strat": "trend_follow", "score": score,
                                "adx": adx, "ema_5m": ema_5m})
            # SHORT
            elif (adx > 25 and ema_align < -0.3 and minus_di > plus_di
                    and 40 <= rsi_14 <= 70 and macd_sig in ("bearish", "bearish_cross")):
                score = 0
                if adx > 25: score += 2
                if adx > 35: score += 1
                if ema_align < -0.5: score += 1
                if macd_sig == "bearish_cross": score += 1
                if ema_5m in ("bearish", "strong_bearish"): score += 2
                if volume_ratio > 1.0: score += 1
                signals.append({"bar": i, "dir": "SHORT", "strat": "trend_follow", "score": score,
                                "adx": adx, "ema_5m": ema_5m})

        # --- 2. BREAKOUT ---
        if prev_bb_squeeze and not bb_squeeze and volume_ratio > 1.2 and macd_sig:
            score = 4
            if volume_ratio > 2.0: score += 2
            if volume_ratio > 1.5: score += 1
            if adx and adx > 20: score += 1
            if ema_align and abs(ema_align) > 0.5: score += 1

            if macd_sig in ("bullish", "bullish_cross") and (ema_align or 0) >= 0:
                signals.append({"bar": i, "dir": "LONG", "strat": "breakout", "score": score})
            elif macd_sig in ("bearish", "bearish_cross") and (ema_align or 0) <= 0:
                signals.append({"bar": i, "dir": "SHORT", "strat": "breakout", "score": score})

        # --- 3. MEAN REVERSION ---
        if rsi_14 and bb_pct is not None and stoch_k is not None:
            if adx is None or adx < 30:  # only in non-trending
                # LONG oversold
                if rsi_14 < 30 and bb_pct < 0.15 and stoch_k < 20:
                    score = 3
                    if rsi_14 < 22: score += 2
                    if bb_pct < 0.05: score += 1
                    if stoch_k < 10: score += 1
                    if mfi and mfi < 25: score += 1
                    if volume_ratio > 1.5: score += 1
                    signals.append({"bar": i, "dir": "LONG", "strat": "mean_reversion", "score": score})
                # SHORT overbought
                elif rsi_14 > 70 and bb_pct > 0.85 and stoch_k > 80:
                    score = 3
                    if rsi_14 > 78: score += 2
                    if bb_pct > 0.95: score += 1
                    if stoch_k > 90: score += 1
                    if mfi and mfi > 75: score += 1
                    if volume_ratio > 1.5: score += 1
                    signals.append({"bar": i, "dir": "SHORT", "strat": "mean_reversion", "score": score})

        # --- 4. MACD CROSS ---
        if macd_sig and ema_align is not None:
            if macd_sig == "bullish_cross" and ema_align >= 0 and volume_ratio > 0.5:
                signals.append({"bar": i, "dir": "LONG", "strat": "macd_cross", "score": 5})
            elif macd_sig == "bearish_cross" and ema_align <= 0 and volume_ratio > 0.5:
                signals.append({"bar": i, "dir": "SHORT", "strat": "macd_cross", "score": 5})

        # --- 5. EMA CROSS ---
        if ema_9 and ema_21 and prev_ema9 and prev_ema21:
            ema_50 = _safe(row.get("ema_50"))
            if prev_ema9 < prev_ema21 and ema_9 > ema_21:
                if ema_50 and ema_21 > ema_50:
                    signals.append({"bar": i, "dir": "LONG", "strat": "ema_cross", "score": 5})
            elif prev_ema9 > prev_ema21 and ema_9 < ema_21:
                if ema_50 and ema_21 < ema_50:
                    signals.append({"bar": i, "dir": "SHORT", "strat": "ema_cross", "score": 5})

        # --- 6. BOLLINGER BOUNCE ---
        if bb_pct is not None and rsi_14 and (adx is None or adx < 30):
            if bb_pct < 0.15 and rsi_14 < 40:
                score = 4
                if bb_pct < 0.05: score += 2
                if rsi_14 < 30: score += 1
                signals.append({"bar": i, "dir": "LONG", "strat": "bollinger_bounce", "score": score})
            elif bb_pct > 0.85 and rsi_14 > 60:
                score = 4
                if bb_pct > 0.95: score += 2
                if rsi_14 > 70: score += 1
                signals.append({"bar": i, "dir": "SHORT", "strat": "bollinger_bounce", "score": score})

        # --- 7. VOLUME SPIKE ---
        if volume_ratio > 2.0 and adx and adx > 20 and macd_sig and ema_align is not None:
            score = 4
            if volume_ratio > 3.0: score += 2
            if adx > 30: score += 1
            if macd_sig in ("bullish", "bullish_cross") and ema_align > 0:
                signals.append({"bar": i, "dir": "LONG", "strat": "volume_spike", "score": score})
            elif macd_sig in ("bearish", "bearish_cross") and ema_align < 0:
                signals.append({"bar": i, "dir": "SHORT", "strat": "volume_spike", "score": score})

        # --- 8. STOCH REVERSAL ---
        if stoch_k is not None and prev_stoch is not None and bb_pct is not None:
            if prev_stoch < 20 and stoch_k > 20 and bb_pct < 0.3:
                signals.append({"bar": i, "dir": "LONG", "strat": "stoch_reversal", "score": 5})
            elif prev_stoch > 80 and stoch_k < 80 and bb_pct > 0.7:
                signals.append({"bar": i, "dir": "SHORT", "strat": "stoch_reversal", "score": 5})

        # --- 9. ICHIMOKU CLOUD ---
        if tenkan and kijun and senkou_a and senkou_b:
            if price > senkou_a and price > senkou_b and tenkan > kijun:
                signals.append({"bar": i, "dir": "LONG", "strat": "ichimoku_cloud", "score": 5})
            elif price < senkou_a and price < senkou_b and tenkan < kijun:
                signals.append({"bar": i, "dir": "SHORT", "strat": "ichimoku_cloud", "score": 5})

        # --- 10. WILLIAMS + CCI ---
        if williams_r is not None and cci is not None:
            if williams_r < -80 and cci < -100 and volume_ratio > 0.8:
                score = 5
                if williams_r < -90: score += 1
                if cci < -150: score += 1
                signals.append({"bar": i, "dir": "LONG", "strat": "williams_cci", "score": score})
            elif williams_r > -20 and cci > 100 and volume_ratio > 0.8:
                score = 5
                if williams_r > -10: score += 1
                if cci > 150: score += 1
                signals.append({"bar": i, "dir": "SHORT", "strat": "williams_cci", "score": score})

        # --- 11. TRIPLE CONFIRMATION ---
        if rsi_14 and macd_sig and ema_align is not None and stoch_k is not None and bb_pct is not None:
            bull_count = 0
            if rsi_14 < 40: bull_count += 1
            if macd_sig in ("bullish", "bullish_cross"): bull_count += 1
            if ema_align > 0: bull_count += 1
            if stoch_k < 30: bull_count += 1
            if bb_pct < 0.3: bull_count += 1
            if volume_ratio > 1.0: bull_count += 1
            if bull_count >= 4:
                signals.append({"bar": i, "dir": "LONG", "strat": "triple_confirm", "score": bull_count})

            bear_count = 0
            if rsi_14 > 60: bear_count += 1
            if macd_sig in ("bearish", "bearish_cross"): bear_count += 1
            if ema_align < 0: bear_count += 1
            if stoch_k > 70: bear_count += 1
            if bb_pct > 0.7: bear_count += 1
            if volume_ratio > 1.0: bear_count += 1
            if bear_count >= 4:
                signals.append({"bar": i, "dir": "SHORT", "strat": "triple_confirm", "score": bear_count})

        # --- 12. MOMENTUM (RSI cross 50) ---
        if rsi_14 and prev_rsi and adx and macd_sig:
            if prev_rsi < 50 and rsi_14 > 50 and adx > 20 and macd_sig in ("bullish", "bullish_cross"):
                score = 5
                if adx > 30: score += 1
                if volume_ratio > 1.0: score += 1
                signals.append({"bar": i, "dir": "LONG", "strat": "momentum", "score": score})
            elif prev_rsi > 50 and rsi_14 < 50 and adx > 20 and macd_sig in ("bearish", "bearish_cross"):
                score = 5
                if adx > 30: score += 1
                if volume_ratio > 1.0: score += 1
                signals.append({"bar": i, "dir": "SHORT", "strat": "momentum", "score": score})

    return signals


# ============================================================
# FAST TRADE SIMULATION (numpy-accelerated)
# ============================================================

def simulate_trades(df: pd.DataFrame, signals: list[dict], params: dict) -> dict:
    """Simulate trades from pre-computed signals with given SL/TP/trailing params.

    This is the FAST part — only iterates from signal bars, not all bars.
    """
    sl_pct = params["sl_pct"]
    tp_pct = params["tp_pct"]
    trailing_pct = params.get("trailing_pct", 0.015)
    cooldown = params.get("cooldown_bars", 15)
    min_score = params.get("min_score", 0)
    leverage = params.get("leverage", 2)
    pos_pct = params.get("position_pct", 0.005)
    strategy_filter = params.get("strategy_filter")  # None = all

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

        # Filters
        if strategy_filter and sig["strat"] != strategy_filter:
            continue
        if sig["score"] < min_score:
            continue
        if bar <= last_exit_bar + cooldown:
            continue
        if bar >= n_bars - 1:
            continue

        entry_price = closes[bar]
        direction = sig["dir"]
        notional = equity * pos_pct * leverage
        margin = notional / leverage

        # SL/TP prices
        if direction == "LONG":
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)
        else:
            sl_price = entry_price * (1 + sl_pct)
            tp_price = entry_price * (1 - tp_pct)

        # Walk forward from entry bar
        best_price = entry_price
        exit_price = None
        exit_reason = None
        exit_bar = None

        for j in range(bar + 1, n_bars):
            h = highs[j]
            l = lows[j]
            c = closes[j]

            if direction == "LONG":
                # Update trailing
                if h > best_price:
                    best_price = h
                    new_sl = best_price * (1 - trailing_pct)
                    if new_sl > sl_price:
                        sl_price = new_sl

                # Check SL (low goes below SL)
                if l <= sl_price:
                    exit_price = sl_price
                    exit_reason = "sl"
                    exit_bar = j
                    break
                # Check TP
                if h >= tp_price:
                    exit_price = tp_price
                    exit_reason = "tp"
                    exit_bar = j
                    break
            else:  # SHORT
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

        # Force close at end
        if exit_price is None:
            exit_price = closes[-1]
            exit_reason = "eod"
            exit_bar = n_bars - 1

        # Calculate PnL
        if direction == "LONG":
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        fee_cost = 2 * FEE_PCT * notional
        pnl_usd = notional * pnl_pct - fee_cost

        equity += pnl_usd
        if equity > peak_equity:
            peak_equity = equity
        dd = (equity - peak_equity) / peak_equity
        if dd < max_dd:
            max_dd = dd

        last_exit_bar = exit_bar
        trades.append({
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct,
            "exit_reason": exit_reason,
            "hold_bars": exit_bar - bar,
            "direction": direction,
            "strategy": sig["strat"],
        })

    # Compute stats
    if not trades:
        return None

    pnls = [t["pnl_usd"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    losses = len(pnls) - wins
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0

    win_pcts = [t["pnl_pct"] for t in trades if t["pnl_usd"] > 0]
    loss_pcts = [t["pnl_pct"] for t in trades if t["pnl_usd"] <= 0]

    pnl_pcts = [t["pnl_pct"] for t in trades]
    sharpe = 0
    if len(pnl_pcts) > 1 and np.std(pnl_pcts) > 0:
        sharpe = np.mean(pnl_pcts) / np.std(pnl_pcts) * np.sqrt(len(pnl_pcts))

    # Exit reason breakdown
    tp_hits = sum(1 for t in trades if t["exit_reason"] == "tp")
    sl_hits = sum(1 for t in trades if t["exit_reason"] == "sl")

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
        "final_equity": equity,
    }


# ============================================================
# PARAMETER GRIDS
# ============================================================

def generate_trade_params():
    """Generate SL/TP/trailing parameter combinations."""
    from itertools import product

    sl_values = [0.008, 0.012, 0.015, 0.020, 0.025]
    tp_values = [0.015, 0.020, 0.030, 0.040, 0.060]
    trailing_values = [0.010, 0.015, 0.020]
    cooldown_values = [5, 10, 20]
    min_score_values = [0, 4]
    leverage_values = [2, 3]

    combos = []
    for sl, tp, tr, cd, ms, lev in product(
        sl_values, tp_values, trailing_values, cooldown_values, min_score_values, leverage_values
    ):
        # Only keep combos where TP > SL (positive R:R)
        if tp <= sl:
            continue
        combos.append({
            "sl_pct": sl,
            "tp_pct": tp,
            "trailing_pct": tr,
            "cooldown_bars": cd,
            "min_score": ms,
            "leverage": lev,
        })

    return combos


STRATEGY_NAMES = [
    "trend_follow", "breakout", "mean_reversion", "macd_cross",
    "ema_cross", "bollinger_bounce", "volume_spike", "stoch_reversal",
    "ichimoku_cloud", "williams_cci", "triple_confirm", "momentum"
]


def main():
    t0 = time.time()

    print("=" * 70)
    print("OPTIMIZED BACKTESTER — SIGNAL-FIRST APPROACH")
    print("=" * 70)

    # Step 1: Download & compute indicators
    print("\nSTEP 1: Loading data...")
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
        print(f"{len(df)} candles")

    # Step 2: Detect all signals
    print(f"\nSTEP 2: Detecting signals across all pairs...")
    pair_signals = {}
    for pair, df in data.items():
        t1 = time.time()
        signals = detect_all_signals(df)
        elapsed = time.time() - t1
        strat_counts = {}
        for s in signals:
            strat_counts[s["strat"]] = strat_counts.get(s["strat"], 0) + 1
        print(f"  {pair}: {len(signals)} signals in {elapsed:.1f}s")
        for strat, cnt in sorted(strat_counts.items(), key=lambda x: -x[1]):
            print(f"    {strat}: {cnt}")
        pair_signals[pair] = signals

    # Step 3: Generate trade param combos
    trade_params = generate_trade_params()
    print(f"\nSTEP 3: Trade parameter combinations: {len(trade_params)}")

    # Step 4: Run simulations
    total_sims = len(trade_params) * len(STRATEGY_NAMES) * len(data)
    print(f"\nSTEP 4: Running {total_sims:,} simulations...")
    print("=" * 70)

    all_results = []
    sim_count = 0
    t_sim_start = time.time()

    for pair, df in data.items():
        signals = pair_signals[pair]
        if not signals:
            continue

        for strat_name in STRATEGY_NAMES:
            # Filter signals for this strategy
            strat_signals = [s for s in signals if s["strat"] == strat_name]
            if not strat_signals:
                continue

            for params in trade_params:
                p = dict(params)
                p["strategy_filter"] = strat_name
                result = simulate_trades(df, strat_signals, p)
                sim_count += 1

                if result and result["total_trades"] >= 5:
                    result["strategy"] = strat_name
                    result["pair"] = pair
                    result["params"] = {k: v for k, v in params.items()}
                    all_results.append(result)

            # Progress
            elapsed = time.time() - t_sim_start
            rate = sim_count / elapsed if elapsed > 0 else 0
            pct = sim_count / total_sims * 100
            if sim_count % 5000 == 0 or pct > 99:
                print(f"  {sim_count:,}/{total_sims:,} ({pct:.0f}%) - {rate:.0f} sims/s - "
                      f"{len(all_results)} qualified", flush=True)

    elapsed_total = time.time() - t_sim_start
    print(f"\n  Simulation complete: {sim_count:,} runs in {elapsed_total:.1f}s "
          f"({sim_count/elapsed_total:.0f} sims/s)")

    # Step 5: Rank results
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    all_results.sort(key=lambda r: (r["profit_factor"], r["total_pnl"]), reverse=True)

    profitable = [r for r in all_results if r["profit_factor"] > 1.0]
    print(f"Total qualified: {len(all_results)}")
    print(f"Profitable (PF>1): {len(profitable)}")

    # Top 50
    print(f"\n{'#':>3} | {'Strategy':18} | {'Pair':10} | {'PF':>5} | {'WR':>5} | "
          f"{'PnL':>9} | {'Trd':>4} | {'TP%':>4} | {'DD':>6} | {'Sharpe':>6} | "
          f"{'SL%':>5} | {'TP%':>5} | {'Trail':>5}")
    print("-" * 130)

    for i, r in enumerate(all_results[:50], 1):
        tp_rate = r["tp_hits"] / r["total_trades"] * 100 if r["total_trades"] > 0 else 0
        p = r["params"]
        print(f"{i:3d} | {r['strategy']:18} | {r['pair']:10} | "
              f"{r['profit_factor']:5.2f} | {r['win_rate']*100:4.1f}% | "
              f"${r['total_pnl']:+8.2f} | {r['total_trades']:4d} | "
              f"{tp_rate:3.0f}% | {r['max_drawdown_pct']*100:5.1f}% | "
              f"{r['sharpe_ratio']:6.2f} | "
              f"{p['sl_pct']*100:4.1f}% | {p['tp_pct']*100:4.1f}% | "
              f"{p['trailing_pct']*100:4.1f}%")

    # Save
    results_file = Path(__file__).parent / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results[:200], f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    # Best strategy detail
    if all_results:
        best = all_results[0]
        print(f"\n{'=' * 70}")
        print("BEST STRATEGY")
        print(f"{'=' * 70}")
        for k, v in sorted(best.items()):
            if k == "params":
                print(f"  Parameters:")
                for pk, pv in sorted(v.items()):
                    print(f"    {pk}: {pv}")
            else:
                print(f"  {k}: {v}")

    # Cross-pair analysis: find strategies that work on multiple pairs
    print(f"\n{'=' * 70}")
    print("CROSS-PAIR CONSISTENCY (strategies profitable on 3+ pairs)")
    print(f"{'=' * 70}")

    # Group by strategy+params
    from collections import defaultdict
    strategy_cross = defaultdict(list)
    for r in all_results:
        if r["profit_factor"] > 1.0:
            key = (r["strategy"], json.dumps(r["params"], sort_keys=True))
            strategy_cross[key].append(r)

    consistent = []
    for (strat, params_str), entries in strategy_cross.items():
        pairs_profitable = set(e["pair"] for e in entries)
        if len(pairs_profitable) >= 3:
            total_pnl = sum(e["total_pnl"] for e in entries)
            avg_pf = np.mean([e["profit_factor"] for e in entries])
            avg_wr = np.mean([e["win_rate"] for e in entries])
            total_trades = sum(e["total_trades"] for e in entries)
            consistent.append({
                "strategy": strat,
                "params": json.loads(params_str),
                "pairs": list(pairs_profitable),
                "n_pairs": len(pairs_profitable),
                "total_pnl": total_pnl,
                "avg_pf": avg_pf,
                "avg_wr": avg_wr,
                "total_trades": total_trades,
            })

    consistent.sort(key=lambda x: (x["n_pairs"], x["total_pnl"]), reverse=True)

    for i, c in enumerate(consistent[:20], 1):
        print(f"  #{i} {c['strategy']} | {c['n_pairs']} pairs | "
              f"PF={c['avg_pf']:.2f} WR={c['avg_wr']*100:.1f}% "
              f"PnL=${c['total_pnl']:+.2f} trades={c['total_trades']} | "
              f"pairs: {','.join(c['pairs'])}")
        if i <= 5:
            for k, v in sorted(c["params"].items()):
                print(f"      {k}: {v}")

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
