"""Position tracking and PnL management."""

import logging
from datetime import datetime, timedelta
from typing import Optional

from core.models import Position, Trade
from db.database import Database
from execution.paper_trader import PaperTrader

logger = logging.getLogger(__name__)


class PositionManager:
    """Tracks open positions, calculates realized/unrealized PnL, daily stats."""

    def __init__(self, trader: PaperTrader, db: Database):
        self.trader = trader
        self.db = db
        self._today_start_balance: Optional[float] = None
        self._today_start: Optional[str] = None

    async def initialize(self):
        """Load open positions from DB and set today's starting balance."""
        self._today_start = datetime.utcnow().strftime("%Y-%m-%d")
        self._today_start_balance = self.trader.total_equity

        # Load open trades from DB
        open_trades = await self.db.get_open_trades()
        for t in open_trades:
            from core.models import Direction
            pos = Position(
                id=t["id"],
                pair=t["pair"],
                direction=Direction(t["direction"]),
                entry_price=t["entry_price"],
                quantity=t["quantity"],
                leverage=t["leverage"],
                stop_loss=0,
                take_profit=0,
                margin_used=t["margin_used"],
                entry_fee=t.get("entry_fee", 0),
                opened_at=datetime.fromisoformat(t["opened_at"]),
                entry_reasoning=t.get("entry_reasoning", ""),
            )
            self.trader.positions[t["pair"]] = pos
            logger.info(f"Restored open position: {t['pair']} {t['direction']}")

    def get_open_positions(self) -> list[dict]:
        """Get all open positions as dicts."""
        return [
            {
                "id": p.id,
                "pair": p.pair,
                "direction": p.direction.value,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "quantity": p.quantity,
                "leverage": p.leverage,
                "margin_used": p.margin_used,
                "unrealized_pnl": round(p.unrealized_pnl, 4),
                "stop_loss": p.stop_loss,
                "take_profit": p.take_profit,
                "hold_time_minutes": round(
                    (datetime.utcnow() - p.opened_at).total_seconds() / 60, 1
                ),
                "opened_at": p.opened_at.isoformat(),
            }
            for p in self.trader.positions.values()
        ]

    def get_equity_summary(self) -> dict:
        """Get current equity and PnL summary."""
        total_unrealized = sum(
            p.unrealized_pnl for p in self.trader.positions.values()
        )
        return {
            "balance": round(self.trader.balance, 2),
            "total_equity": round(self.trader.total_equity, 2),
            "margin_used": round(self.trader.total_margin_used, 2),
            "free_margin": round(self.trader.balance, 2),
            "unrealized_pnl": round(total_unrealized, 4),
            "open_positions": len(self.trader.positions),
            "drawdown_pct": round(self.trader.drawdown_pct * 100, 2),
            "initial_balance": self.trader.initial_balance,
            "total_pnl": round(self.trader.total_equity - self.trader.initial_balance, 2),
            "total_pnl_pct": round(
                (self.trader.total_equity - self.trader.initial_balance)
                / self.trader.initial_balance * 100, 2
            ),
        }

    async def compute_daily_stats(self) -> dict:
        """Compute stats for today."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        start = f"{today}T00:00:00"
        end = f"{today}T23:59:59"

        # Get today's closed trades
        trades = await self.db.get_trades(status="closed", limit=1000)
        today_trades = [
            t for t in trades
            if t.get("closed_at") and t["closed_at"][:10] == today
        ]

        pnl_gross = sum(t["pnl"] + t.get("entry_fee", 0) + t.get("exit_fee", 0) for t in today_trades)
        total_fees = sum(t.get("entry_fee", 0) + t.get("exit_fee", 0) for t in today_trades)
        pnl_net = sum(t["pnl"] for t in today_trades)
        winning = [t for t in today_trades if t["pnl"] > 0]
        losing = [t for t in today_trades if t["pnl"] <= 0]

        # API costs for today
        api_costs = await self.db.get_api_costs(since=start)
        total_api_cost = sum(c["cost_usd"] for c in api_costs)

        # Add VPS cost prorated daily ($6/month ÷ 30 = $0.20/day)
        vps_daily_cost = 6.0 / 30.0

        starting = self._today_start_balance or self.trader.initial_balance

        stats = {
            "date": today,
            "starting_balance": round(starting, 2),
            "ending_balance": round(self.trader.total_equity, 2),
            "pnl_gross": round(pnl_gross, 4),
            "pnl_net": round(pnl_net - total_api_cost - vps_daily_cost, 4),
            "total_trades": len(today_trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "total_fees": round(total_fees, 4),
            "total_api_costs": round(total_api_cost, 4),
            "max_drawdown_pct": round(self.trader.drawdown_pct * 100, 2),
            "best_trade_pnl": max((t["pnl"] for t in today_trades), default=0),
            "worst_trade_pnl": min((t["pnl"] for t in today_trades), default=0),
            "avg_hold_time_minutes": round(
                sum(t.get("hold_time_minutes", 0) for t in today_trades) / max(len(today_trades), 1), 2
            ),
        }

        await self.db.upsert_daily_stats(stats)
        return stats

    def check_new_day(self):
        """Reset daily tracking if date changed."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if today != self._today_start:
            self._today_start = today
            self._today_start_balance = self.trader.total_equity
            logger.info(f"New day: {today}, starting balance: {self._today_start_balance:.2f}")
