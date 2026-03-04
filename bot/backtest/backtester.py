"""Core backtesting simulation engine.

Self-contained module that walks through historical OHLCV bars, applies a
user-supplied strategy function, simulates trade execution with fees,
trailing stops, and cooldowns, then returns comprehensive performance metrics.

Dependencies: numpy, pandas (no other bot modules imported).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Immutable summary of a single backtest run."""

    strategy_name: str
    params: dict
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    profit_factor: float       # gross_profit / gross_loss
    max_drawdown_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    avg_hold_bars: float
    sharpe_ratio: float
    trades: list = field(default_factory=list)  # individual trade dicts

    def summary(self) -> str:
        """Human-readable one-liner."""
        return (
            f"{self.strategy_name} | "
            f"trades={self.total_trades} W/L={self.wins}/{self.losses} "
            f"WR={self.win_rate:.1%} PnL=${self.total_pnl:+.2f} "
            f"PF={self.profit_factor:.2f} MDD={self.max_drawdown_pct:.2%} "
            f"Sharpe={self.sharpe_ratio:.2f}"
        )


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """Bar-by-bar backtesting engine for single-pair strategies.

    Parameters
    ----------
    df : pd.DataFrame
        Historical OHLCV data with columns ``open, high, low, close, volume``
        plus any pre-computed indicator columns the strategy needs.  The index
        should be sequential (reset_index) so that integer bar positions are
        meaningful.
    pair : str
        Trading pair symbol (informational, included in trade records).
    fee_pct : float
        One-way fee as a fraction (e.g. 0.0005 = 0.05 %).  Applied on both
        entry and exit.
    initial_equity : float
        Starting account equity in USD.
    """

    INITIAL_EQUITY_DEFAULT = 5000.0

    def __init__(
        self,
        df: pd.DataFrame,
        pair: str,
        fee_pct: float = 0.0005,
        initial_equity: float = 5000.0,
    ):
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        self.df = df.reset_index(drop=True)
        self.pair = pair
        self.fee_pct = fee_pct
        self.initial_equity = initial_equity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        strategy_func: Callable[[pd.Series, dict], Optional[dict]],
        params: dict,
    ) -> BacktestResult:
        """Execute a full backtest.

        Parameters
        ----------
        strategy_func : callable
            ``strategy_func(row, params) -> Optional[dict]``

            Called on every bar when flat.  Must return ``None`` (no trade) or
            a signal dict with keys:

            * ``direction`` – ``"LONG"`` or ``"SHORT"``
            * ``sl_distance_pct`` – stop-loss distance as a fraction of entry
              price (e.g. 0.012 = 1.2 %).
            * ``tp_distance_pct`` – take-profit distance as a fraction.
            * ``trailing_pct`` – (optional) trailing-stop ratchet distance as
              a fraction.  Set to 0 or omit to disable.

        params : dict
            Strategy parameters forwarded to *strategy_func*.  The engine also
            reads the following keys (with defaults):

            * ``position_pct`` (float) – fraction of equity risked per trade
              (default 0.005 = 0.5 %).
            * ``leverage`` (int/float) – leverage multiplier (default 2).
            * ``cooldown_bars`` (int) – bars to skip after closing a trade
              (default 15).

        Returns
        -------
        BacktestResult
        """

        equity = self.initial_equity
        peak_equity = equity

        position_pct = params.get("position_pct", 0.005)
        leverage = params.get("leverage", 2)
        cooldown_bars = params.get("cooldown_bars", 15)

        # Trade state
        in_trade = False
        trade_dir: Optional[str] = None       # "LONG" / "SHORT"
        entry_price: float = 0.0
        stop_loss: float = 0.0
        take_profit: float = 0.0
        trailing_pct: float = 0.0
        best_price: float = 0.0
        notional: float = 0.0
        entry_bar: int = 0

        cooldown_remaining: int = 0

        # Accumulators
        trades: List[Dict] = []
        equity_curve: List[float] = []
        max_drawdown_pct: float = 0.0

        n_bars = len(self.df)

        for i in range(n_bars):
            row = self.df.iloc[i]
            bar_high = float(row["high"])
            bar_low = float(row["low"])
            bar_close = float(row["close"])

            # ── 1. If in a trade, check exits on this bar ────────────
            if in_trade:
                # Update best price for trailing stop
                if trade_dir == "LONG":
                    if bar_high > best_price:
                        best_price = bar_high
                    # Ratchet trailing stop (only in favorable direction)
                    if trailing_pct > 0:
                        new_sl = best_price * (1.0 - trailing_pct)
                        if new_sl > stop_loss:
                            stop_loss = new_sl
                else:  # SHORT
                    if bar_low < best_price:
                        best_price = bar_low
                    if trailing_pct > 0:
                        new_sl = best_price * (1.0 + trailing_pct)
                        if new_sl < stop_loss:
                            stop_loss = new_sl

                # Determine exit
                exit_price: Optional[float] = None
                exit_reason: Optional[str] = None

                if trade_dir == "LONG":
                    if bar_low <= stop_loss:
                        exit_price = stop_loss
                        exit_reason = "sl"
                    elif bar_high >= take_profit:
                        exit_price = take_profit
                        exit_reason = "tp"
                else:  # SHORT
                    if bar_high >= stop_loss:
                        exit_price = stop_loss
                        exit_reason = "sl"
                    elif bar_low <= take_profit:
                        exit_price = take_profit
                        exit_reason = "tp"

                if exit_price is not None:
                    # Compute PnL
                    if trade_dir == "LONG":
                        price_change_pct = (exit_price - entry_price) / entry_price
                    else:
                        price_change_pct = (entry_price - exit_price) / entry_price

                    fee_cost = 2.0 * self.fee_pct * notional
                    raw_pnl = notional * price_change_pct
                    net_pnl = raw_pnl - fee_cost
                    pnl_pct = net_pnl / (notional / leverage) if notional > 0 else 0.0

                    equity += net_pnl
                    hold_bars = i - entry_bar

                    trades.append({
                        "entry_bar": entry_bar,
                        "exit_bar": i,
                        "direction": trade_dir,
                        "entry_price": round(entry_price, 8),
                        "exit_price": round(exit_price, 8),
                        "pnl_pct": round(pnl_pct, 6),
                        "pnl_usd": round(net_pnl, 4),
                        "exit_reason": exit_reason,
                        "hold_bars": hold_bars,
                    })

                    # Reset state
                    in_trade = False
                    trade_dir = None
                    cooldown_remaining = cooldown_bars

            # ── 2. If flat, consider entry ───────────────────────────
            if not in_trade:
                if cooldown_remaining > 0:
                    cooldown_remaining -= 1
                else:
                    signal = strategy_func(row, params)
                    if signal is not None:
                        trade_dir = signal["direction"]
                        sl_dist = signal["sl_distance_pct"]
                        tp_dist = signal["tp_distance_pct"]
                        trailing_pct = signal.get("trailing_pct", 0.0)

                        entry_price = bar_close
                        notional = position_pct * equity * leverage

                        if trade_dir == "LONG":
                            stop_loss = entry_price * (1.0 - sl_dist)
                            take_profit = entry_price * (1.0 + tp_dist)
                        else:  # SHORT
                            stop_loss = entry_price * (1.0 + sl_dist)
                            take_profit = entry_price * (1.0 - tp_dist)

                        best_price = entry_price
                        entry_bar = i
                        in_trade = True

            # ── 3. Track equity curve & drawdown ─────────────────────
            equity_curve.append(equity)
            if equity > peak_equity:
                peak_equity = equity
            if peak_equity > 0:
                dd = (equity - peak_equity) / peak_equity
                if dd < max_drawdown_pct:
                    max_drawdown_pct = dd

        # ── Force-close any open position at the last bar ────────────
        if in_trade:
            last_close = float(self.df.iloc[-1]["close"])
            if trade_dir == "LONG":
                price_change_pct = (last_close - entry_price) / entry_price
            else:
                price_change_pct = (entry_price - last_close) / entry_price

            fee_cost = 2.0 * self.fee_pct * notional
            raw_pnl = notional * price_change_pct
            net_pnl = raw_pnl - fee_cost
            pnl_pct = net_pnl / (notional / leverage) if notional > 0 else 0.0
            equity += net_pnl
            hold_bars = (n_bars - 1) - entry_bar

            trades.append({
                "entry_bar": entry_bar,
                "exit_bar": n_bars - 1,
                "direction": trade_dir,
                "entry_price": round(entry_price, 8),
                "exit_price": round(last_close, 8),
                "pnl_pct": round(pnl_pct, 6),
                "pnl_usd": round(net_pnl, 4),
                "exit_reason": "end_of_data",
                "hold_bars": hold_bars,
            })

            equity_curve[-1] = equity
            if equity > peak_equity:
                peak_equity = equity
            if peak_equity > 0:
                dd = (equity - peak_equity) / peak_equity
                if dd < max_drawdown_pct:
                    max_drawdown_pct = dd

        # ── Aggregate statistics ─────────────────────────────────────
        return self._build_result(
            strategy_name=params.get("strategy_name", "unnamed"),
            params=params,
            trades=trades,
            max_drawdown_pct=max_drawdown_pct,
            final_equity=equity,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_result(
        self,
        strategy_name: str,
        params: dict,
        trades: List[Dict],
        max_drawdown_pct: float,
        final_equity: float,
    ) -> BacktestResult:
        """Compute aggregate metrics from the list of closed trades."""

        total_trades = len(trades)
        if total_trades == 0:
            return BacktestResult(
                strategy_name=strategy_name,
                params=params,
                total_trades=0,
                wins=0,
                losses=0,
                win_rate=0.0,
                total_pnl=0.0,
                profit_factor=0.0,
                max_drawdown_pct=0.0,
                avg_win_pct=0.0,
                avg_loss_pct=0.0,
                avg_hold_bars=0.0,
                sharpe_ratio=0.0,
                trades=[],
            )

        pnls_usd = np.array([t["pnl_usd"] for t in trades])
        pnls_pct = np.array([t["pnl_pct"] for t in trades])
        hold_bars = np.array([t["hold_bars"] for t in trades])

        wins_mask = pnls_usd > 0
        losses_mask = pnls_usd <= 0
        wins = int(wins_mask.sum())
        losses = int(losses_mask.sum())

        gross_profit = float(pnls_usd[wins_mask].sum()) if wins > 0 else 0.0
        gross_loss = float(np.abs(pnls_usd[losses_mask]).sum()) if losses > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0

        win_pcts = pnls_pct[wins_mask]
        loss_pcts = pnls_pct[losses_mask]
        avg_win_pct = float(win_pcts.mean()) if len(win_pcts) > 0 else 0.0
        avg_loss_pct = float(loss_pcts.mean()) if len(loss_pcts) > 0 else 0.0

        # Sharpe ratio: annualise by sqrt(trades_per_day)
        # Estimate trades_per_day from bar count (assume 1-min bars → 1440/day)
        total_bars = len(self.df)
        total_days = max(total_bars / 1440.0, 1.0)
        trades_per_day = total_trades / total_days

        std = float(np.std(pnls_pct, ddof=1)) if total_trades > 1 else 0.0
        mean_ret = float(np.mean(pnls_pct))
        if std > 0:
            sharpe = mean_ret / std * math.sqrt(trades_per_day)
        else:
            sharpe = 0.0

        return BacktestResult(
            strategy_name=strategy_name,
            params=params,
            total_trades=total_trades,
            wins=wins,
            losses=losses,
            win_rate=wins / total_trades if total_trades > 0 else 0.0,
            total_pnl=round(final_equity - self.initial_equity, 4),
            profit_factor=round(profit_factor, 4),
            max_drawdown_pct=round(max_drawdown_pct, 6),
            avg_win_pct=round(avg_win_pct, 6),
            avg_loss_pct=round(avg_loss_pct, 6),
            avg_hold_bars=round(float(hold_bars.mean()), 2),
            sharpe_ratio=round(sharpe, 4),
            trades=trades,
        )
