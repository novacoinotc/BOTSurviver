#!/usr/bin/env python3
"""
LOW FREQUENCY BACKTEST: 1h timeframe, multi-confirmation, maker fees.
Goal: ~200 trades/year instead of 1000. Bigger moves, lower fees.
"""
import sys, time, json, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.data_loader import download_klines, load_proxy_url
from backtest.indicators import compute_all_indicators

# ============================================================
# CONFIG
# ============================================================
ALL_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "SUIUSDT", "ARBUSDT", "OPUSDT", "APTUSDT",
    "NEARUSDT", "LTCUSDT", "ATOMUSDT", "FILUSDT", "INJUSDT",
]
PAIRS = sys.argv[1:] if len(sys.argv) > 1 else ALL_PAIRS
DAYS = 365
INTERVAL = "1h"
INITIAL_EQUITY = 50_000.0

# Real fees — MAKER for entries + TP (limit orders)
MAKER_FEE = 0.0002    # 0.02%
TAKER_FEE = 0.0005    # 0.05%
SLIPPAGE_MAJOR = 0.0001
SLIPPAGE_ALT = 0.0002
MAJOR_PAIRS = {"BTCUSDT", "ETHUSDT"}


# ============================================================
# SIGNAL DETECTION — HIGH CONVICTION MULTI-CONFIRMATION
# ============================================================
def _safe(val):
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return None
    return val


def detect_signals_1h(df):
    """Detect high-conviction signals using multi-indicator confirmation.

    3 signal types:
    1. multi_confirm: 4+ indicator groups align (strictest, fewest trades)
    2. trend_follow:  ADX>25 + EMA alignment + MACD + RSI pullback
    3. macd_cross:    Same as before but on 1h (for comparison)
    """
    signals = []
    n = len(df)

    for i in range(1, n):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        price = row["close"]
        rsi_14 = _safe(row.get("rsi_14"))
        rsi_7 = _safe(row.get("rsi_7"))
        adx = _safe(row.get("adx"))
        ema_align = _safe(row.get("ema_alignment"))
        plus_di = _safe(row.get("plus_di"))
        minus_di = _safe(row.get("minus_di"))
        macd_sig = row.get("macd_signal")
        macd_hist = _safe(row.get("macd_hist"))
        prev_macd_hist = _safe(prev.get("macd_hist"))
        bb_pct = _safe(row.get("bb_pct"))
        stoch_k = _safe(row.get("stoch_rsi_k"))
        volume_ratio = _safe(row.get("volume_ratio")) or 0
        mfi = _safe(row.get("mfi"))
        williams_r = _safe(row.get("williams_r"))
        cci = _safe(row.get("cci"))
        tenkan = _safe(row.get("ichimoku_tenkan"))
        kijun = _safe(row.get("ichimoku_kijun"))
        senkou_a = _safe(row.get("ichimoku_senkou_a"))
        senkou_b = _safe(row.get("ichimoku_senkou_b"))
        rsi_div = row.get("rsi_divergence")

        # ============================================================
        # STRATEGY 1: MULTI-CONFIRMATION (strictest)
        # Count how many indicator groups confirm direction
        # ============================================================
        bull_score = 0
        bear_score = 0

        # Group 1: MACD (0-3 pts)
        if macd_sig == "bullish_cross":
            bull_score += 3
        elif macd_sig == "bullish":
            bull_score += 1
        if macd_sig == "bearish_cross":
            bear_score += 3
        elif macd_sig == "bearish":
            bear_score += 1

        # Group 2: EMA alignment (0-2 pts)
        if ema_align is not None:
            if ema_align > 0.5:
                bull_score += 2
            elif ema_align > 0:
                bull_score += 1
            if ema_align < -0.5:
                bear_score += 2
            elif ema_align < 0:
                bear_score += 1

        # Group 3: RSI momentum (0-2 pts)
        if rsi_14 is not None:
            if rsi_14 < 45:
                bull_score += 1
            if rsi_14 < 35:
                bull_score += 1
            if rsi_14 > 55:
                bear_score += 1
            if rsi_14 > 65:
                bear_score += 1

        # Group 4: ADX trend strength (0-2 pts)
        if adx is not None:
            if adx > 25:
                bull_score += 1
                bear_score += 1
            if adx > 35:
                bull_score += 1
                bear_score += 1

        # Group 5: DI direction (0-1 pt)
        if plus_di is not None and minus_di is not None:
            if plus_di > minus_di:
                bull_score += 1
            if minus_di > plus_di:
                bear_score += 1

        # Group 6: Volume confirmation (0-2 pts)
        if volume_ratio > 1.5:
            bull_score += 2
            bear_score += 2
        elif volume_ratio > 1.0:
            bull_score += 1
            bear_score += 1

        # Group 7: Stochastic RSI (0-2 pts)
        if stoch_k is not None:
            if stoch_k < 25:
                bull_score += 2
            elif stoch_k < 40:
                bull_score += 1
            if stoch_k > 75:
                bear_score += 2
            elif stoch_k > 60:
                bear_score += 1

        # Group 8: Bollinger position (0-1 pt)
        if bb_pct is not None:
            if bb_pct < 0.25:
                bull_score += 1
            if bb_pct > 0.75:
                bear_score += 1

        # Group 9: Ichimoku (0-2 pts)
        if all(v is not None for v in [tenkan, kijun, senkou_a, senkou_b]):
            if price > max(senkou_a, senkou_b) and tenkan > kijun:
                bull_score += 2
            elif price > min(senkou_a, senkou_b):
                bull_score += 1
            if price < min(senkou_a, senkou_b) and tenkan < kijun:
                bear_score += 2
            elif price < max(senkou_a, senkou_b):
                bear_score += 1

        # Group 10: Williams %R + CCI (0-2 pts)
        if williams_r is not None:
            if williams_r < -75:
                bull_score += 1
            if williams_r > -25:
                bear_score += 1
        if cci is not None:
            if cci < -100:
                bull_score += 1
            if cci > 100:
                bear_score += 1

        # Group 11: MFI (0-1 pt)
        if mfi is not None:
            if mfi < 30:
                bull_score += 1
            if mfi > 70:
                bear_score += 1

        # Group 12: RSI divergence (0-2 pts)
        if rsi_div == "bullish_div":
            bull_score += 2
        elif rsi_div == "bearish_div":
            bear_score += 2

        # Max possible: ~23 pts. Emit signal with score
        # Only emit LONG or SHORT, not both
        if bull_score >= 6 and bull_score > bear_score + 2:
            signals.append({"bar": i, "dir": "LONG", "strat": "multi_confirm", "score": bull_score})
        elif bear_score >= 6 and bear_score > bull_score + 2:
            signals.append({"bar": i, "dir": "SHORT", "strat": "multi_confirm", "score": bear_score})

        # ============================================================
        # STRATEGY 2: TREND FOLLOW on 1H
        # Classic: ADX > 25 + EMA aligned + MACD confirms + RSI pullback
        # ============================================================
        if adx and ema_align and plus_di and minus_di and rsi_14 and macd_sig:
            tf_score = 0
            # LONG
            if (adx > 25 and ema_align > 0.3 and plus_di > minus_di
                    and 30 <= rsi_14 <= 55 and macd_sig in ("bullish", "bullish_cross")):
                tf_score = 4
                if adx > 35: tf_score += 1
                if ema_align > 0.5: tf_score += 1
                if macd_sig == "bullish_cross": tf_score += 2
                if volume_ratio > 1.2: tf_score += 1
                if stoch_k and stoch_k < 40: tf_score += 1
                signals.append({"bar": i, "dir": "LONG", "strat": "trend_follow_1h", "score": tf_score})
            # SHORT
            elif (adx > 25 and ema_align < -0.3 and minus_di > plus_di
                    and 45 <= rsi_14 <= 70 and macd_sig in ("bearish", "bearish_cross")):
                tf_score = 4
                if adx > 35: tf_score += 1
                if ema_align < -0.5: tf_score += 1
                if macd_sig == "bearish_cross": tf_score += 2
                if volume_ratio > 1.2: tf_score += 1
                if stoch_k and stoch_k > 60: tf_score += 1
                signals.append({"bar": i, "dir": "SHORT", "strat": "trend_follow_1h", "score": tf_score})

        # ============================================================
        # STRATEGY 3: MACD CROSS on 1H (comparison baseline)
        # ============================================================
        if macd_sig and ema_align is not None:
            if macd_sig == "bullish_cross" and ema_align >= 0 and volume_ratio > 0.5:
                signals.append({"bar": i, "dir": "LONG", "strat": "macd_cross_1h", "score": 5})
            elif macd_sig == "bearish_cross" and ema_align <= 0 and volume_ratio > 0.5:
                signals.append({"bar": i, "dir": "SHORT", "strat": "macd_cross_1h", "score": 5})

    return signals


# ============================================================
# TRADE SIMULATION — MAKER FEES + 1H TIMEFRAME
# ============================================================
def simulate_low_freq(df, signals, params, pair):
    """Simulate with maker fees for entries/TP, taker for SL."""
    sl_pct = params["sl_pct"]
    tp_pct = params["tp_pct"]
    trailing_pct = params["trailing_pct"]
    cooldown = params["cooldown_bars"]
    min_score = params["min_score"]
    leverage = params["leverage"]
    pos_pct = params["position_pct"]
    strategy_filter = params.get("strategy_filter")

    is_major = pair in MAJOR_PAIRS
    slippage = SLIPPAGE_MAJOR if is_major else SLIPPAGE_ALT

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
    monthly = defaultdict(lambda: {
        "start_equity": 0, "end_equity": 0,
        "trades": 0, "wins": 0, "pnl": 0.0, "fees": 0.0,
    })

    for sig in signals:
        bar = sig["bar"]
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

        if notional < 10:
            continue

        # Entry fee: MAKER (limit order, no slippage)
        entry_fee = notional * MAKER_FEE

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

        # PnL
        if direction == "LONG":
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        # Exit fee: MAKER for TP (limit), TAKER+slippage for SL/trailing
        if exit_reason == "tp":
            exit_fee = notional * MAKER_FEE
        else:
            exit_fee = notional * (TAKER_FEE + slippage)

        total_fee = entry_fee + exit_fee
        pnl_usd = notional * pnl_pct - total_fee

        equity += pnl_usd
        if equity > peak_equity:
            peak_equity = equity
        dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
        if dd < max_dd:
            max_dd = dd

        if equity <= 0:
            trades.append({"pnl_usd": pnl_usd, "pnl_pct": pnl_pct,
                           "exit_reason": "liquidation", "hold_bars": exit_bar - bar,
                           "direction": direction, "fees": total_fee, "month": ""})
            equity = 0
            break

        last_exit_bar = exit_bar

        # Monthly tracking
        ts = pd.Timestamp(timestamps[bar])
        month_key = ts.strftime("%Y-%m")
        m = monthly[month_key]
        if m["start_equity"] == 0:
            m["start_equity"] = equity - pnl_usd
        m["end_equity"] = equity
        m["trades"] += 1
        m["pnl"] += pnl_usd
        m["fees"] += total_fee
        if pnl_usd > 0:
            m["wins"] += 1

        trades.append({"pnl_usd": pnl_usd, "pnl_pct": pnl_pct,
                       "exit_reason": exit_reason, "hold_bars": exit_bar - bar,
                       "direction": direction, "fees": total_fee, "month": month_key})

    if not trades:
        return None

    pnls = [t["pnl_usd"] for t in trades]
    total_fees = sum(t["fees"] for t in trades)
    wins = sum(1 for p in pnls if p > 0)
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0

    pnl_pcts_arr = [t["pnl_pct"] for t in trades]
    sharpe = 0
    if len(pnl_pcts_arr) > 1 and np.std(pnl_pcts_arr) > 0:
        sharpe = np.mean(pnl_pcts_arr) / np.std(pnl_pcts_arr) * np.sqrt(len(pnl_pcts_arr))

    tp_hits = sum(1 for t in trades if t["exit_reason"] == "tp")
    sl_hits = sum(1 for t in trades if t["exit_reason"] == "sl")

    # Monthly returns
    monthly_returns = []
    for mk in sorted(monthly.keys()):
        m = monthly[mk]
        ret = (m["pnl"] / m["start_equity"] * 100) if m["start_equity"] > 0 else 0
        monthly_returns.append({
            "month": mk, "pnl": round(m["pnl"], 2),
            "return_pct": round(ret, 2), "trades": m["trades"],
            "wins": m["wins"], "fees": round(m["fees"], 2),
        })

    return {
        "total_trades": len(trades),
        "wins": wins, "win_rate": round(wins / len(trades), 4),
        "total_pnl": round(sum(pnls), 2),
        "total_fees": round(total_fees, 2),
        "profit_factor": round(pf, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "sharpe_ratio": round(sharpe, 4),
        "tp_hits": tp_hits, "sl_hits": sl_hits,
        "trail_wins": wins - tp_hits,
        "final_equity": round(equity, 2),
        "return_pct": round((equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100, 2),
        "monthly": monthly_returns,
        "avg_hold_hours": round(np.mean([t["hold_bars"] for t in trades]), 1),
        "fee_pct_of_volume": round(total_fees / sum(abs(p) for p in pnls) * 100, 1) if sum(abs(p) for p in pnls) > 0 else 0,
    }


# ============================================================
# PARAMETER GRID — LOW FREQUENCY, WIDE TARGETS
# ============================================================
STRATEGIES = ["multi_confirm", "trend_follow_1h", "macd_cross_1h"]

PARAM_GRID = []
for sl, tp, tr, cd, ms, lev, pos in product(
    [0.02, 0.03, 0.04, 0.05],         # SL: 2-5% (wider for 1h)
    [0.06, 0.10, 0.15, 0.20],          # TP: 6-20%
    [0.03, 0.05, 0.06],                # Trailing: 3-6%
    [12, 24, 48],                       # Cooldown: 12h, 1d, 2d
    [5, 7, 9],                          # Min score thresholds
    [3, 5, 8, 10],                      # Leverage
    [0.01, 0.02],                       # Position size 1-2%
):
    if tp <= sl:
        continue
    PARAM_GRID.append({
        "sl_pct": sl, "tp_pct": tp, "trailing_pct": tr,
        "cooldown_bars": cd, "min_score": ms, "leverage": lev,
        "position_pct": pos,
    })


def main():
    t0 = time.time()
    proxy_url = load_proxy_url()

    print("=" * 80, flush=True)
    print("LOW FREQUENCY BACKTEST — 1H TIMEFRAME, MULTI-CONFIRMATION, MAKER FEES", flush=True)
    print(f"$50K | {len(PAIRS)} pairs | 12 months | Leverage 3x-10x", flush=True)
    print("=" * 80, flush=True)
    print(f"Strategies: {STRATEGIES}", flush=True)
    print(f"Param combos: {len(PARAM_GRID)}", flush=True)
    print(f"Fees: Maker={MAKER_FEE*100:.2f}% Taker={TAKER_FEE*100:.2f}%", flush=True)
    print(flush=True)

    # Step 1: Download 1h data
    print("STEP 1: Downloading 1H data (12 months)...", flush=True)
    data = {}
    for pair in PAIRS:
        print(f"  {pair}...", end=" ", flush=True)
        try:
            df = download_klines(pair, interval=INTERVAL, days=DAYS, proxy_url=proxy_url)
            if df.empty or len(df) < 200:
                print(f"FAIL ({len(df) if not df.empty else 0})", flush=True)
                continue
            df = compute_all_indicators(df)
            data[pair] = df
            ts_first = pd.Timestamp(df["timestamp"].iloc[0])
            ts_last = pd.Timestamp(df["timestamp"].iloc[-1])
            print(f"{len(df)} candles ({ts_first.strftime('%Y-%m-%d')} to {ts_last.strftime('%Y-%m-%d')})", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
        time.sleep(1)  # Rate limit

    print(f"\n  Loaded: {len(data)} pairs", flush=True)

    # Step 2: Detect signals
    print(f"\nSTEP 2: Detecting signals on 1H...", flush=True)
    pair_signals = {}
    for pair, df in data.items():
        t1 = time.time()
        signals = detect_signals_1h(df)
        elapsed = time.time() - t1
        strat_counts = defaultdict(int)
        for s in signals:
            strat_counts[s["strat"]] += 1
        print(f"  {pair}: {len(signals)} signals ({elapsed:.1f}s)", flush=True)
        for strat, cnt in sorted(strat_counts.items()):
            print(f"    {strat}: {cnt}", flush=True)
        pair_signals[pair] = signals

    # Step 3: Simulate
    total_sims = len(PARAM_GRID) * len(data) * len(STRATEGIES)
    print(f"\nSTEP 3: Running {total_sims:,} simulations...", flush=True)

    all_results = []
    sim_count = 0
    t_sim = time.time()

    for pair, df in data.items():
        sigs = pair_signals[pair]
        if not sigs:
            continue

        t_pair = time.time()
        pair_results = []

        for strat in STRATEGIES:
            strat_sigs = [s for s in sigs if s["strat"] == strat]
            if not strat_sigs:
                continue

            for params in PARAM_GRID:
                p = dict(params)
                p["strategy_filter"] = strat
                result = simulate_low_freq(df, strat_sigs, p, pair)
                sim_count += 1

                if result and result["total_trades"] >= 5:
                    result["pair"] = pair
                    result["strategy"] = strat
                    result["params"] = {k: v for k, v in params.items()}
                    pair_results.append(result)

        elapsed_pair = time.time() - t_pair
        profitable = sum(1 for r in pair_results if r["profit_factor"] > 1.0)
        best_pnl = max((r["total_pnl"] for r in pair_results), default=0)
        print(f"  {pair}: {len(pair_results)} qualified, {profitable} profitable, "
              f"best=${best_pnl:+,.2f} ({elapsed_pair:.1f}s)", flush=True)
        all_results.extend(pair_results)

    elapsed_sim = time.time() - t_sim
    print(f"\n  {sim_count:,} sims in {elapsed_sim:.1f}s ({sim_count/elapsed_sim:.0f} sims/s)", flush=True)

    # Step 4: Results
    all_results.sort(key=lambda r: (r["profit_factor"], r["total_pnl"]), reverse=True)
    profitable = [r for r in all_results if r["profit_factor"] > 1.0]

    print(f"\n{'='*80}", flush=True)
    print(f"RESULTS — LOW FREQUENCY 1H (12 MONTHS, $50K)", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Qualified: {len(all_results):,} | Profitable: {len(profitable):,} ({len(profitable)/max(len(all_results),1)*100:.0f}%)", flush=True)

    # Compare vs 5m baseline
    print(f"\n--- STRATEGY COMPARISON ---", flush=True)
    for strat in STRATEGIES:
        strat_results = [r for r in all_results if r["strategy"] == strat]
        strat_profitable = [r for r in strat_results if r["profit_factor"] > 1.0]
        if strat_results:
            best = max(strat_results, key=lambda r: r["total_pnl"])
            avg_trades = np.mean([r["total_trades"] for r in strat_results])
            avg_fees = np.mean([r["total_fees"] for r in strat_results])
            print(f"  {strat:20} | {len(strat_profitable):4}/{len(strat_results):4} profitable | "
                  f"best=${best['total_pnl']:+,.2f} ({best['return_pct']:+.1f}%) | "
                  f"avg_trades={avg_trades:.0f} avg_fees=${avg_fees:,.0f}", flush=True)

    # Top 40
    print(f"\n{'#':>3} | {'Strategy':20} | {'Pair':10} | {'PF':>5} | {'WR':>5} | {'PnL':>12} | "
          f"{'Ret%':>7} | {'Trd':>4} | {'DD':>6} | {'Fees':>7} | {'Lev':>3} | {'SL':>4} | "
          f"{'TP':>4} | {'Trail':>5} | {'AvgH':>5}", flush=True)
    print("-" * 140, flush=True)
    for i, r in enumerate(all_results[:40], 1):
        p = r["params"]
        print(f"{i:3d} | {r['strategy']:20} | {r['pair']:10} | {r['profit_factor']:5.2f} | "
              f"{r['win_rate']*100:4.1f}% | ${r['total_pnl']:+11,.2f} | {r['return_pct']:+6.1f}% | "
              f"{r['total_trades']:4d} | {r['max_drawdown_pct']*100:5.1f}% | ${r['total_fees']:6,.0f} | "
              f"{p['leverage']:2d}x | {p['sl_pct']*100:.0f}% | {p['tp_pct']*100:.0f}% | "
              f"{p['trailing_pct']*100:.0f}% | {r['avg_hold_hours']:4.0f}h", flush=True)

    # Per-pair ranking
    print(f"\n{'='*80}", flush=True)
    print("PER-PAIR RANKING (best config)", flush=True)
    print(f"{'='*80}", flush=True)

    pair_best = {}
    for r in all_results:
        p = r["pair"]
        if p not in pair_best or r["total_pnl"] > pair_best[p]["total_pnl"]:
            pair_best[p] = r

    sorted_pairs = sorted(pair_best.items(), key=lambda x: x[1]["total_pnl"], reverse=True)
    keep_pairs = []
    drop_pairs = []

    for pair, r in sorted_pairs:
        p = r["params"]
        status = "+" if r["profit_factor"] > 1.0 else "-"
        monthly_rets = [m["return_pct"] for m in r["monthly"]]
        avg_mo = np.mean(monthly_rets) if monthly_rets else 0
        green = sum(1 for m in monthly_rets if m > 0)

        if r["profit_factor"] > 1.0:
            keep_pairs.append(pair)
        else:
            drop_pairs.append(pair)

        print(f"  [{status}] {pair:10} | {r['strategy']:20} | PF={r['profit_factor']:5.2f} "
              f"PnL=${r['total_pnl']:+9,.2f} ({r['return_pct']:+5.1f}%) | "
              f"trades={r['total_trades']:3d} fees=${r['total_fees']:,.0f} | "
              f"Lev={p['leverage']}x SL={p['sl_pct']*100:.0f}% TP={p['tp_pct']*100:.0f}% | "
              f"avg_mo={avg_mo:+.2f}% green={green}/{len(monthly_rets)}", flush=True)

    print(f"\n  KEEP ({len(keep_pairs)}): {', '.join(keep_pairs)}", flush=True)
    print(f"  DROP ({len(drop_pairs)}): {', '.join(drop_pairs)}", flush=True)

    # Cross-pair analysis
    config_cross = defaultdict(list)
    for r in all_results:
        if r["profit_factor"] > 1.0:
            key = (r["strategy"], json.dumps(r["params"], sort_keys=True))
            config_cross[key].append(r)

    cross_configs = []
    for (strat, ps), entries in config_cross.items():
        pairs_ok = set(e["pair"] for e in entries)
        if len(pairs_ok) >= 3:
            total_pnl = sum(e["total_pnl"] for e in entries)
            monthly_agg = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0, "fees": 0})
            for e in entries:
                for m in e["monthly"]:
                    monthly_agg[m["month"]]["pnl"] += m["pnl"]
                    monthly_agg[m["month"]]["trades"] += m["trades"]
                    monthly_agg[m["month"]]["wins"] += m["wins"]
                    monthly_agg[m["month"]]["fees"] += m["fees"]

            cross_configs.append({
                "strategy": strat, "params": json.loads(ps),
                "n_pairs": len(pairs_ok), "pairs": sorted(pairs_ok),
                "total_pnl": total_pnl,
                "return_pct": total_pnl / INITIAL_EQUITY * 100,
                "avg_pf": np.mean([e["profit_factor"] for e in entries]),
                "total_trades": sum(e["total_trades"] for e in entries),
                "total_fees": sum(e["total_fees"] for e in entries),
                "worst_dd": min(e["max_drawdown_pct"] for e in entries),
                "monthly_agg": dict(monthly_agg),
            })

    cross_configs.sort(key=lambda x: (x["n_pairs"], x["total_pnl"]), reverse=True)

    if cross_configs:
        print(f"\n{'='*80}", flush=True)
        print("CROSS-PAIR CONFIGS (profitable on 3+ pairs)", flush=True)
        print(f"{'='*80}", flush=True)
        for i, c in enumerate(cross_configs[:15], 1):
            p = c["params"]
            print(f"  #{i:2d} | {c['strategy']:20} | {c['n_pairs']} pairs | "
                  f"PF={c['avg_pf']:.2f} PnL=${c['total_pnl']:+,.2f} ({c['return_pct']:+.1f}%) | "
                  f"trades={c['total_trades']} fees=${c['total_fees']:,.0f}", flush=True)
            print(f"       Lev={p['leverage']}x SL={p['sl_pct']*100:.0f}% TP={p['tp_pct']*100:.0f}% "
                  f"Trail={p['trailing_pct']*100:.0f}% CD={p['cooldown_bars']}h "
                  f"MinScore={p['min_score']} Pos={p['position_pct']*100:.0f}%", flush=True)

        # Monthly breakdown of best cross-pair
        best = cross_configs[0]
        print(f"\n{'='*80}", flush=True)
        print(f"MONTHLY COMPOUND — BEST CONFIG ({best['strategy']}, {best['n_pairs']} pairs)", flush=True)
        print(f"{'='*80}", flush=True)
        p = best["params"]
        print(f"Lev={p['leverage']}x SL={p['sl_pct']*100:.0f}% TP={p['tp_pct']*100:.0f}% "
              f"Trail={p['trailing_pct']*100:.0f}% CD={p['cooldown_bars']}h MinScore={p['min_score']}", flush=True)
        print(f"Pairs: {', '.join(best['pairs'])}", flush=True)
        print(flush=True)

        compound_eq = INITIAL_EQUITY
        print(f"  {'Month':>7} | {'Start':>12} | {'PnL':>10} | {'Fees':>7} | "
              f"{'Return%':>8} | {'End':>12} | {'Trades':>6} | {'WR':>5}", flush=True)
        print(f"  {'-'*90}", flush=True)

        for mk in sorted(best["monthly_agg"].keys()):
            m = best["monthly_agg"][mk]
            ret = m["pnl"] / compound_eq * 100 if compound_eq > 0 else 0
            wr = m["wins"] / m["trades"] * 100 if m["trades"] > 0 else 0
            end_eq = compound_eq + m["pnl"]
            print(f"  {mk:>7} | ${compound_eq:>11,.2f} | ${m['pnl']:>+9,.2f} | "
                  f"${m['fees']:>6,.0f} | {ret:>+7.2f}% | ${end_eq:>11,.2f} | "
                  f"{m['trades']:>6} | {wr:4.0f}%", flush=True)
            compound_eq = end_eq

        total_ret = (compound_eq - INITIAL_EQUITY) / INITIAL_EQUITY * 100
        avg_mo = total_ret / len(best["monthly_agg"]) if best["monthly_agg"] else 0
        green = sum(1 for m in best["monthly_agg"].values() if m["pnl"] > 0)
        total_months = len(best["monthly_agg"])

        print(f"\n  Final: ${compound_eq:,.2f} | Return: {total_ret:+.2f}% | "
              f"Avg Monthly: {avg_mo:+.2f}% | Green: {green}/{total_months}", flush=True)
        print(f"  Total Fees: ${best['total_fees']:,.2f} | "
              f"Fee/PnL ratio: {best['total_fees']/abs(best['total_pnl'])*100:.0f}% of gross" if best['total_pnl'] != 0 else "", flush=True)

    # Fee comparison vs 5m
    print(f"\n{'='*80}", flush=True)
    print("FEE COMPARISON: 1H MAKER vs 5M TAKER", flush=True)
    print(f"{'='*80}", flush=True)
    if all_results:
        avg_trades_1h = np.mean([r["total_trades"] for r in all_results])
        avg_fees_1h = np.mean([r["total_fees"] for r in all_results])
        print(f"  1H approach: avg {avg_trades_1h:.0f} trades/year, avg ${avg_fees_1h:,.0f} fees", flush=True)
        print(f"  5M approach: avg ~1000 trades/year, avg ~$4,000-$12,000 fees", flush=True)
        print(f"  Fee reduction: ~{(1 - avg_fees_1h/8000)*100:.0f}%", flush=True)

    # Save
    output = {
        "config": {"equity": INITIAL_EQUITY, "days": DAYS, "interval": INTERVAL,
                    "pairs": list(data.keys()), "strategies": STRATEGIES,
                    "fees": {"maker": MAKER_FEE, "taker": TAKER_FEE}},
        "summary": {"qualified": len(all_results), "profitable": len(profitable),
                     "keep_pairs": keep_pairs, "drop_pairs": drop_pairs},
        "top_results": all_results[:200],
        "cross_pair": cross_configs[:30] if cross_configs else [],
        "per_pair": {pair: {"pf": r["profit_factor"], "pnl": r["total_pnl"],
                            "strategy": r["strategy"], "params": r["params"],
                            "monthly": r["monthly"]}
                     for pair, r in pair_best.items()} if pair_best else {},
    }
    results_file = Path(__file__).parent / "results_low_freq.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}", flush=True)
    print(f"Total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
