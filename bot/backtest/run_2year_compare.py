#!/usr/bin/env python3
"""
2-YEAR BACKTEST COMPARISON: Current rules vs Strict EMA alignment rules.

Downloads 1h data ONCE, then runs both rule sets side-by-side.
- CURRENT:  LONG ema_align >= 0,  SHORT ema_align <= 0  (allows neutral 0.0)
- STRICT:   LONG ema_align > 0.3, SHORT ema_align < -0.3 (requires real alignment)

Output: results_2year_current.json + results_2year_strict.json
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
DAYS = 730  # 2 YEARS
INTERVAL = "1h"
INITIAL_EQUITY = 50_000.0

MAKER_FEE = 0.0002
TAKER_FEE = 0.0005
SLIPPAGE_MAJOR = 0.0001
SLIPPAGE_ALT = 0.0002
MAJOR_PAIRS = {"BTCUSDT", "ETHUSDT"}


# ============================================================
# SIGNAL DETECTION — TWO RULE SETS
# ============================================================
def _safe(val):
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return None
    return val


def detect_signals_1h(df, rule_set="current"):
    """Detect signals with either 'current' or 'strict' EMA alignment rules.

    rule_set='current': ema_align >= 0 (LONG), <= 0 (SHORT) — allows neutral
    rule_set='strict':  ema_align > 0.3 (LONG), < -0.3 (SHORT) — requires alignment
    """
    signals = []
    n = len(df)

    for i in range(1, n):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        price = row["close"]
        rsi_14 = _safe(row.get("rsi_14"))
        adx = _safe(row.get("adx"))
        ema_align = _safe(row.get("ema_alignment"))
        plus_di = _safe(row.get("plus_di"))
        minus_di = _safe(row.get("minus_di"))
        macd_sig = row.get("macd_signal")
        macd_hist = _safe(row.get("macd_hist"))
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
        # STRATEGY 1: MULTI-CONFIRMATION (identical for both rule sets)
        # ============================================================
        bull_score = 0
        bear_score = 0

        if macd_sig == "bullish_cross": bull_score += 3
        elif macd_sig == "bullish": bull_score += 1
        if macd_sig == "bearish_cross": bear_score += 3
        elif macd_sig == "bearish": bear_score += 1

        if ema_align is not None:
            if ema_align > 0.5: bull_score += 2
            elif ema_align > 0: bull_score += 1
            if ema_align < -0.5: bear_score += 2
            elif ema_align < 0: bear_score += 1

        if rsi_14 is not None:
            if rsi_14 < 45: bull_score += 1
            if rsi_14 < 35: bull_score += 1
            if rsi_14 > 55: bear_score += 1
            if rsi_14 > 65: bear_score += 1

        if adx is not None:
            if adx > 25: bull_score += 1; bear_score += 1
            if adx > 35: bull_score += 1; bear_score += 1

        if plus_di is not None and minus_di is not None:
            if plus_di > minus_di: bull_score += 1
            if minus_di > plus_di: bear_score += 1

        if volume_ratio > 1.5: bull_score += 2; bear_score += 2
        elif volume_ratio > 1.0: bull_score += 1; bear_score += 1

        if stoch_k is not None:
            if stoch_k < 25: bull_score += 2
            elif stoch_k < 40: bull_score += 1
            if stoch_k > 75: bear_score += 2
            elif stoch_k > 60: bear_score += 1

        if bb_pct is not None:
            if bb_pct < 0.25: bull_score += 1
            if bb_pct > 0.75: bear_score += 1

        if all(v is not None for v in [tenkan, kijun, senkou_a, senkou_b]):
            if price > max(senkou_a, senkou_b) and tenkan > kijun: bull_score += 2
            elif price > min(senkou_a, senkou_b): bull_score += 1
            if price < min(senkou_a, senkou_b) and tenkan < kijun: bear_score += 2
            elif price < max(senkou_a, senkou_b): bear_score += 1

        if williams_r is not None:
            if williams_r < -75: bull_score += 1
            if williams_r > -25: bear_score += 1
        if cci is not None:
            if cci < -100: bull_score += 1
            if cci > 100: bear_score += 1

        if mfi is not None:
            if mfi < 30: bull_score += 1
            if mfi > 70: bear_score += 1

        if rsi_div == "bullish_div": bull_score += 2
        elif rsi_div == "bearish_div": bear_score += 2

        if bull_score >= 6 and bull_score > bear_score + 2:
            signals.append({"bar": i, "dir": "LONG", "strat": "multi_confirm", "score": bull_score})
        elif bear_score >= 6 and bear_score > bull_score + 2:
            signals.append({"bar": i, "dir": "SHORT", "strat": "multi_confirm", "score": bear_score})

        # ============================================================
        # STRATEGY 2: TREND FOLLOW on 1H (same for both, already uses 0.3)
        # ============================================================
        if adx and ema_align and plus_di and minus_di and rsi_14 and macd_sig:
            tf_score = 0
            if (adx > 25 and ema_align > 0.3 and plus_di > minus_di
                    and 30 <= rsi_14 <= 55 and macd_sig in ("bullish", "bullish_cross")):
                tf_score = 4
                if adx > 35: tf_score += 1
                if ema_align > 0.5: tf_score += 1
                if macd_sig == "bullish_cross": tf_score += 2
                if volume_ratio > 1.2: tf_score += 1
                if stoch_k and stoch_k < 40: tf_score += 1
                signals.append({"bar": i, "dir": "LONG", "strat": "trend_follow_1h", "score": tf_score})
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
        # STRATEGY 3: MACD CROSS — THIS IS WHERE THE RULES DIFFER
        # ============================================================
        if macd_sig and ema_align is not None:
            if rule_set == "current":
                # CURRENT: allows neutral EMA (0.0)
                if macd_sig == "bullish_cross" and ema_align >= 0 and volume_ratio > 0.5:
                    signals.append({"bar": i, "dir": "LONG", "strat": "macd_cross_1h", "score": 5})
                elif macd_sig == "bearish_cross" and ema_align <= 0 and volume_ratio > 0.5:
                    signals.append({"bar": i, "dir": "SHORT", "strat": "macd_cross_1h", "score": 5})
            else:
                # STRICT: requires real EMA alignment (> 0.3 / < -0.3)
                if macd_sig == "bullish_cross" and ema_align > 0.3 and volume_ratio > 0.5:
                    signals.append({"bar": i, "dir": "LONG", "strat": "macd_cross_1h", "score": 5})
                elif macd_sig == "bearish_cross" and ema_align < -0.3 and volume_ratio > 0.5:
                    signals.append({"bar": i, "dir": "SHORT", "strat": "macd_cross_1h", "score": 5})

    return signals


# ============================================================
# TRADE SIMULATION (identical to run_low_freq.py)
# ============================================================
def simulate_low_freq(df, signals, params, pair):
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
                    exit_price = sl_price; exit_reason = "sl"; exit_bar = j; break
                if h >= tp_price:
                    exit_price = tp_price; exit_reason = "tp"; exit_bar = j; break
            else:
                if l < best_price:
                    best_price = l
                    new_sl = best_price * (1 + trailing_pct)
                    if new_sl < sl_price:
                        sl_price = new_sl
                if h >= sl_price:
                    exit_price = sl_price; exit_reason = "sl"; exit_bar = j; break
                if l <= tp_price:
                    exit_price = tp_price; exit_reason = "tp"; exit_bar = j; break

        if exit_price is None:
            exit_price = closes[-1]; exit_reason = "eod"; exit_bar = n_bars - 1

        if direction == "LONG":
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        if exit_reason == "tp":
            exit_fee = notional * MAKER_FEE
        else:
            exit_fee = notional * (TAKER_FEE + slippage)

        total_fee = entry_fee + exit_fee
        pnl_usd = notional * pnl_pct - total_fee
        equity += pnl_usd

        if equity > peak_equity: peak_equity = equity
        dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
        if dd < max_dd: max_dd = dd

        if equity <= 0:
            trades.append({"pnl_usd": pnl_usd, "pnl_pct": pnl_pct,
                           "exit_reason": "liquidation", "hold_bars": exit_bar - bar,
                           "direction": direction, "fees": total_fee, "month": ""})
            equity = 0; break

        last_exit_bar = exit_bar
        ts = pd.Timestamp(timestamps[bar])
        month_key = ts.strftime("%Y-%m")
        m = monthly[month_key]
        if m["start_equity"] == 0: m["start_equity"] = equity - pnl_usd
        m["end_equity"] = equity
        m["trades"] += 1; m["pnl"] += pnl_usd; m["fees"] += total_fee
        if pnl_usd > 0: m["wins"] += 1

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
        "total_trades": len(trades), "wins": wins,
        "win_rate": round(wins / len(trades), 4),
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
    }


# ============================================================
# PARAM GRID
# ============================================================
STRATEGIES = ["multi_confirm", "trend_follow_1h", "macd_cross_1h"]

PARAM_GRID = []
for sl, tp, tr, cd, ms, lev, pos in product(
    [0.02, 0.03, 0.04, 0.05],
    [0.06, 0.10, 0.15, 0.20],
    [0.03, 0.05, 0.06],
    [12, 24, 48],
    [5, 7, 9],
    [3, 5, 8, 10],
    [0.01, 0.02],
):
    if tp <= sl:
        continue
    PARAM_GRID.append({
        "sl_pct": sl, "tp_pct": tp, "trailing_pct": tr,
        "cooldown_bars": cd, "min_score": ms, "leverage": lev,
        "position_pct": pos,
    })


def run_backtest(data, pair_signals, rule_name):
    """Run full backtest for one rule set."""
    all_results = []
    for pair, df in data.items():
        sigs = pair_signals[pair]
        if not sigs:
            continue
        for strat in STRATEGIES:
            strat_sigs = [s for s in sigs if s["strat"] == strat]
            if not strat_sigs:
                continue
            for params in PARAM_GRID:
                p = dict(params)
                p["strategy_filter"] = strat
                result = simulate_low_freq(df, strat_sigs, p, pair)
                if result and result["total_trades"] >= 5:
                    result["pair"] = pair
                    result["strategy"] = strat
                    result["params"] = {k: v for k, v in params.items()}
                    all_results.append(result)
    return all_results


def analyze_results(all_results, rule_name):
    """Print and save analysis for one rule set."""
    all_results.sort(key=lambda r: (r["profit_factor"], r["total_pnl"]), reverse=True)
    profitable = [r for r in all_results if r["profit_factor"] > 1.0]

    print(f"\n{'='*80}")
    print(f"  RESULTS — {rule_name.upper()} RULES (2 YEARS, $50K)")
    print(f"{'='*80}")
    print(f"Qualified: {len(all_results):,} | Profitable: {len(profitable):,} ({len(profitable)/max(len(all_results),1)*100:.0f}%)")

    # Strategy comparison
    print(f"\n--- STRATEGY COMPARISON ({rule_name}) ---")
    for strat in STRATEGIES:
        strat_results = [r for r in all_results if r["strategy"] == strat]
        strat_profitable = [r for r in strat_results if r["profit_factor"] > 1.0]
        if strat_results:
            best = max(strat_results, key=lambda r: r["total_pnl"])
            avg_trades = np.mean([r["total_trades"] for r in strat_results])
            print(f"  {strat:20} | {len(strat_profitable):4}/{len(strat_results):4} profitable | "
                  f"best=${best['total_pnl']:+,.2f} ({best['return_pct']:+.1f}%) | avg_trades={avg_trades:.0f}")

    # Top 20
    print(f"\n{'#':>3} | {'Strategy':20} | {'Pair':10} | {'PF':>5} | {'WR':>5} | {'PnL':>12} | "
          f"{'Ret%':>7} | {'Trd':>4} | {'DD':>6} | {'Lev':>3} | {'SL':>4} | {'TP':>4}")
    print("-" * 120)
    for i, r in enumerate(all_results[:20], 1):
        p = r["params"]
        print(f"{i:3d} | {r['strategy']:20} | {r['pair']:10} | {r['profit_factor']:5.2f} | "
              f"{r['win_rate']*100:4.1f}% | ${r['total_pnl']:+11,.2f} | {r['return_pct']:+6.1f}% | "
              f"{r['total_trades']:4d} | {r['max_drawdown_pct']*100:5.1f}% | "
              f"{p['leverage']:2d}x | {p['sl_pct']*100:.0f}% | {p['tp_pct']*100:.0f}%")

    # Per-pair best
    pair_best = {}
    for r in all_results:
        p = r["pair"]
        if p not in pair_best or r["total_pnl"] > pair_best[p]["total_pnl"]:
            pair_best[p] = r

    print(f"\n--- PER-PAIR RANKING ({rule_name}) ---")
    sorted_pairs = sorted(pair_best.items(), key=lambda x: x[1]["total_pnl"], reverse=True)
    keep = [p for p, r in sorted_pairs if r["profit_factor"] > 1.0]
    drop = [p for p, r in sorted_pairs if r["profit_factor"] <= 1.0]

    for pair, r in sorted_pairs:
        p = r["params"]
        status = "+" if r["profit_factor"] > 1.0 else "-"
        print(f"  [{status}] {pair:10} | {r['strategy']:20} | PF={r['profit_factor']:5.2f} "
              f"PnL=${r['total_pnl']:+9,.2f} ({r['return_pct']:+5.1f}%) | "
              f"trades={r['total_trades']:3d} | Lev={p['leverage']}x SL={p['sl_pct']*100:.0f}% TP={p['tp_pct']*100:.0f}%")

    print(f"\n  KEEP ({len(keep)}): {', '.join(keep)}")
    print(f"  DROP ({len(drop)}): {', '.join(drop)}")

    # Cross-pair configs (profitable on 3+ pairs with same params)
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
            cross_configs.append({
                "strategy": strat, "params": json.loads(ps),
                "n_pairs": len(pairs_ok), "pairs": sorted(pairs_ok),
                "total_pnl": total_pnl,
                "return_pct": total_pnl / INITIAL_EQUITY * 100,
                "avg_pf": np.mean([e["profit_factor"] for e in entries]),
                "total_trades": sum(e["total_trades"] for e in entries),
                "total_fees": sum(e["total_fees"] for e in entries),
                "worst_dd": min(e["max_drawdown_pct"] for e in entries),
            })

    cross_configs.sort(key=lambda x: (x["n_pairs"], x["total_pnl"]), reverse=True)

    if cross_configs:
        print(f"\n--- CROSS-PAIR CONFIGS ({rule_name}, profitable on 3+ pairs) ---")
        for i, c in enumerate(cross_configs[:10], 1):
            p = c["params"]
            print(f"  #{i:2d} | {c['strategy']:20} | {c['n_pairs']} pairs | "
                  f"PF={c['avg_pf']:.2f} PnL=${c['total_pnl']:+,.2f} ({c['return_pct']:+.1f}%) | "
                  f"trades={c['total_trades']}")
            print(f"       Lev={p['leverage']}x SL={p['sl_pct']*100:.0f}% TP={p['tp_pct']*100:.0f}% "
                  f"Trail={p['trailing_pct']*100:.0f}% CD={p['cooldown_bars']}h "
                  f"MinScore={p['min_score']} Pos={p['position_pct']*100:.0f}%")

    # Save
    output = {
        "rule_set": rule_name,
        "config": {"equity": INITIAL_EQUITY, "days": DAYS, "interval": INTERVAL,
                    "pairs": list(ALL_PAIRS), "strategies": STRATEGIES},
        "summary": {"qualified": len(all_results), "profitable": len(profitable),
                     "keep_pairs": keep, "drop_pairs": drop},
        "top_results": all_results[:200],
        "cross_pair": cross_configs[:30],
        "per_pair": {pair: {"pf": r["profit_factor"], "pnl": r["total_pnl"],
                            "strategy": r["strategy"], "params": r["params"],
                            "monthly": r["monthly"]}
                     for pair, r in pair_best.items()},
    }
    results_file = Path(__file__).parent / f"results_2year_{rule_name}.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {results_file}")

    return {
        "rule_set": rule_name,
        "qualified": len(all_results),
        "profitable": len(profitable),
        "keep_pairs": len(keep),
        "drop_pairs": len(drop),
        "best_pnl": all_results[0]["total_pnl"] if all_results else 0,
        "best_cross_pairs": cross_configs[0]["n_pairs"] if cross_configs else 0,
        "best_cross_pnl": cross_configs[0]["total_pnl"] if cross_configs else 0,
    }


def main():
    t0 = time.time()
    proxy_url = load_proxy_url()

    print("=" * 80)
    print("2-YEAR BACKTEST COMPARISON: CURRENT vs STRICT EMA ALIGNMENT")
    print(f"$50K | {len(ALL_PAIRS)} pairs | {DAYS} days | 3 strategies")
    print(f"CURRENT: LONG ema>=0, SHORT ema<=0 (allows neutral)")
    print(f"STRICT:  LONG ema>0.3, SHORT ema<-0.3 (requires alignment)")
    print(f"Param combos: {len(PARAM_GRID)}")
    print("=" * 80)

    # ============================================================
    # STEP 1: Download 1H data (shared for both rule sets)
    # ============================================================
    print("\nSTEP 1: Downloading 1H data (2 years)...", flush=True)
    data = {}
    for pair in ALL_PAIRS:
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
        time.sleep(0.5)

    print(f"\n  Loaded: {len(data)} pairs", flush=True)

    # ============================================================
    # STEP 2: Detect signals for BOTH rule sets
    # ============================================================
    print(f"\nSTEP 2: Detecting signals...", flush=True)

    signals_current = {}
    signals_strict = {}

    for pair, df in data.items():
        sigs_c = detect_signals_1h(df, rule_set="current")
        sigs_s = detect_signals_1h(df, rule_set="strict")

        mc_c = sum(1 for s in sigs_c if s["strat"] == "macd_cross_1h")
        mc_s = sum(1 for s in sigs_s if s["strat"] == "macd_cross_1h")
        diff = mc_c - mc_s

        print(f"  {pair}: current={len(sigs_c)} strict={len(sigs_s)} "
              f"(macd_cross: {mc_c} vs {mc_s}, diff={diff})", flush=True)

        signals_current[pair] = sigs_c
        signals_strict[pair] = sigs_s

    # ============================================================
    # STEP 3: Run simulations for BOTH
    # ============================================================
    print(f"\nSTEP 3: Running simulations (CURRENT rules)...", flush=True)
    t1 = time.time()
    results_current = run_backtest(data, signals_current, "current")
    print(f"  Done in {time.time()-t1:.1f}s ({len(results_current)} qualified)", flush=True)

    print(f"\nSTEP 4: Running simulations (STRICT rules)...", flush=True)
    t2 = time.time()
    results_strict = run_backtest(data, signals_strict, "strict")
    print(f"  Done in {time.time()-t2:.1f}s ({len(results_strict)} qualified)", flush=True)

    # ============================================================
    # STEP 5: Analyze and compare
    # ============================================================
    summary_current = analyze_results(results_current, "current")
    summary_strict = analyze_results(results_strict, "strict")

    # ============================================================
    # FINAL COMPARISON
    # ============================================================
    print(f"\n{'='*80}")
    print("FINAL COMPARISON: CURRENT vs STRICT")
    print(f"{'='*80}")
    print(f"{'Metric':<30} | {'CURRENT':>15} | {'STRICT':>15} | {'Winner':>10}")
    print("-" * 80)

    metrics = [
        ("Qualified configs", summary_current["qualified"], summary_strict["qualified"]),
        ("Profitable configs", summary_current["profitable"], summary_strict["profitable"]),
        ("Pairs profitable", summary_current["keep_pairs"], summary_strict["keep_pairs"]),
        ("Best single PnL", summary_current["best_pnl"], summary_strict["best_pnl"]),
        ("Best cross-pair #pairs", summary_current["best_cross_pairs"], summary_strict["best_cross_pairs"]),
        ("Best cross-pair PnL", summary_current["best_cross_pnl"], summary_strict["best_cross_pnl"]),
    ]

    for name, c, s in metrics:
        if isinstance(c, float):
            winner = "CURRENT" if c > s else ("STRICT" if s > c else "TIE")
            print(f"  {name:<28} | ${c:>14,.2f} | ${s:>14,.2f} | {winner:>10}")
        else:
            winner = "CURRENT" if c > s else ("STRICT" if s > c else "TIE")
            print(f"  {name:<28} | {c:>15,} | {s:>15,} | {winner:>10}")

    # MACD cross specific comparison
    print(f"\n--- MACD CROSS ONLY COMPARISON ---")
    for label, results in [("CURRENT", results_current), ("STRICT", results_strict)]:
        mc_results = [r for r in results if r["strategy"] == "macd_cross_1h"]
        mc_profitable = [r for r in mc_results if r["profit_factor"] > 1.0]
        if mc_results:
            best = max(mc_results, key=lambda r: r["total_pnl"])
            unique_pairs = set(r["pair"] for r in mc_profitable)
            print(f"  {label}: {len(mc_profitable)}/{len(mc_results)} profitable | "
                  f"best=${best['total_pnl']:+,.2f} | unique pairs={len(unique_pairs)}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
