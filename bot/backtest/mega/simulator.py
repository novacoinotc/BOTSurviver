"""Fast trade simulator for mega backtest.

Based on simulate_low_freq from run_low_freq.py with identical fee/slippage
logic. Optimized for batch execution across parameter grids.
"""

import numpy as np
from collections import defaultdict

# Fee configuration (matching run_low_freq.py exactly)
MAKER_FEE = 0.0002      # 0.02%
TAKER_FEE = 0.0005      # 0.05%
SLIPPAGE_MAJOR = 0.0001  # BTC/ETH
SLIPPAGE_ALT = 0.0002    # Alts
MAJOR_PAIRS = {"BTCUSDT", "ETHUSDT"}
INITIAL_EQUITY = 50_000.0


def simulate(df, signals: list[dict], params: dict, pair: str,
             bar_hours: float = 1.0) -> dict | None:
    """Simulate trades with maker/taker fees.

    Args:
        df: DataFrame with OHLCV + indicators (needs high, low, close, timestamp)
        signals: list of {"bar", "dir", "strat", "score"} dicts
        params: dict with sl_pct, tp_pct, trailing_pct, cooldown_bars,
                min_score, leverage, position_pct
        pair: symbol string
        bar_hours: hours per bar (0.25 for 15m, 1.0 for 1h, 4.0 for 4h)

    Returns:
        Result dict or None if no trades.
    """
    sl_pct = params["sl_pct"]
    tp_pct = params["tp_pct"]
    trailing_pct = params["trailing_pct"]
    cooldown = params["cooldown_bars"]
    min_score = params["min_score"]
    leverage = params["leverage"]
    pos_pct = params["position_pct"]

    is_major = pair in MAJOR_PAIRS
    slippage = SLIPPAGE_MAJOR if is_major else SLIPPAGE_ALT

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n_bars = len(df)

    # Check for timestamp column for monthly tracking
    has_ts = "timestamp" in df.columns
    if has_ts:
        import pandas as pd
        timestamps = df["timestamp"].values

    equity = INITIAL_EQUITY
    peak_equity = equity
    max_dd = 0.0
    trades = []
    last_exit_bar = -cooldown - 1
    monthly = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0, "fees": 0.0})

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

        if notional < 10:
            continue

        # Entry fee: MAKER
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
                    if trailing_pct > 0:
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
                    if trailing_pct > 0:
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

        # Exit fee
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
                           "exit_reason": "liquidation", "hold_bars": exit_bar - bar})
            equity = 0
            break

        last_exit_bar = exit_bar

        trade_rec = {"pnl_usd": pnl_usd, "pnl_pct": pnl_pct,
                     "exit_reason": exit_reason, "hold_bars": exit_bar - bar}
        trades.append(trade_rec)

        # Monthly tracking
        if has_ts:
            import pandas as pd
            ts = pd.Timestamp(timestamps[bar])
            mk = ts.strftime("%Y-%m")
            monthly[mk]["pnl"] += pnl_usd
            monthly[mk]["trades"] += 1
            monthly[mk]["fees"] += total_fee
            if pnl_usd > 0:
                monthly[mk]["wins"] += 1

    if not trades:
        return None

    pnls = [t["pnl_usd"] for t in trades]
    total_fees = sum(t.get("fees", 0) for t in trades) if "fees" in trades[0] else 0
    # Recompute fees from trades
    wins = sum(1 for p in pnls if p > 0)
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0

    pnl_pcts = [t["pnl_pct"] for t in trades]
    sharpe = 0
    if len(pnl_pcts) > 1 and np.std(pnl_pcts) > 0:
        sharpe = np.mean(pnl_pcts) / np.std(pnl_pcts) * np.sqrt(len(pnl_pcts))

    tp_hits = sum(1 for t in trades if t["exit_reason"] == "tp")
    sl_hits = sum(1 for t in trades if t["exit_reason"] == "sl")

    # Monthly summary
    monthly_list = []
    for mk in sorted(monthly.keys()):
        m = monthly[mk]
        monthly_list.append({
            "month": mk, "pnl": round(m["pnl"], 2),
            "trades": m["trades"], "wins": m["wins"],
        })

    return {
        "total_trades": len(trades),
        "wins": wins,
        "win_rate": round(wins / len(trades), 4),
        "total_pnl": round(sum(pnls), 2),
        "profit_factor": round(pf, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "sharpe_ratio": round(sharpe, 4),
        "tp_hits": tp_hits,
        "sl_hits": sl_hits,
        "final_equity": round(equity, 2),
        "return_pct": round((equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100, 2),
        "avg_hold_hours": round(np.mean([t["hold_bars"] * bar_hours for t in trades]), 1),
        "monthly": monthly_list,
    }


def simulate_batch(df, signals: list[dict], param_list: list[dict], pair: str,
                   bar_hours: float = 1.0) -> list[dict]:
    """Run simulate() for multiple parameter sets on the same signals.

    Returns list of result dicts (only those with 5+ trades).
    """
    results = []
    for params in param_list:
        result = simulate(df, signals, params, pair, bar_hours=bar_hours)
        if result and result["total_trades"] >= 5:
            result["params"] = params
            results.append(result)
    return results
