"""Paper trading simulator with realistic fees, slippage, and funding rates."""

import logging
import uuid
from datetime import datetime
from typing import Optional

from config.settings import settings, MAJOR_PAIRS
from core.models import Direction, Position, Trade, TradeDecision
from db.database import Database

logger = logging.getLogger(__name__)


class PaperTrader:
    """Simulates order execution with real fees and slippage."""

    def __init__(self, db: Database, initial_balance: float = None):
        self.db = db
        self.balance = initial_balance or settings.initial_balance
        self.initial_balance = self.balance
        self.positions: dict[str, Position] = {}  # pair -> Position
        self._peak_balance = self.balance

    async def open_position(self, decision: TradeDecision, current_price: float) -> Optional[Position]:
        """Open a new position based on Claude's decision."""
        pair = decision.pair

        if pair in self.positions:
            logger.warning(f"Already have position for {pair}")
            return None

        leverage = decision.leverage or settings.default_leverage
        pos_pct = decision.position_size_pct or settings.default_position_pct
        margin = self.balance * pos_pct
        notional = margin * leverage

        # Apply slippage
        slippage = settings.slippage_major if pair in MAJOR_PAIRS else settings.slippage_alt
        if decision.direction == Direction.LONG:
            entry_price = current_price * (1 + slippage)
        else:
            entry_price = current_price * (1 - slippage)

        quantity = notional / entry_price

        # Taker fee on entry (market order)
        entry_fee = notional * settings.taker_fee

        # Deduct margin + fee from balance
        total_cost = margin + entry_fee
        if total_cost > self.balance:
            logger.warning(f"Insufficient balance for {pair}: need {total_cost:.2f}, have {self.balance:.2f}")
            return None

        self.balance -= total_cost

        position = Position(
            id=str(uuid.uuid4())[:8],
            pair=pair,
            direction=decision.direction,
            entry_price=entry_price,
            current_price=current_price,
            quantity=quantity,
            leverage=leverage,
            stop_loss=decision.stop_loss or 0,
            take_profit=decision.take_profit or 0,
            margin_used=margin,
            entry_fee=entry_fee,
            entry_reasoning=decision.reasoning,
            entry_indicators=None,
        )

        self.positions[pair] = position

        # Save to DB
        await self.db.insert_trade({
            "id": position.id,
            "pair": pair,
            "direction": position.direction.value,
            "entry_price": entry_price,
            "quantity": quantity,
            "leverage": leverage,
            "margin_used": margin,
            "entry_fee": entry_fee,
            "opened_at": position.opened_at.isoformat(),
            "entry_reasoning": decision.reasoning,
            "entry_indicators": "{}",
            "status": "open",
        })

        logger.info(
            f"OPENED {decision.direction.value} {pair} @ {entry_price:.4f} "
            f"qty={quantity:.6f} lev={leverage}x margin={margin:.2f} fee={entry_fee:.4f}"
        )
        return position

    async def close_position(self, pair: str, current_price: float, reason: str = "") -> Optional[Trade]:
        """Close an existing position."""
        position = self.positions.get(pair)
        if not position:
            logger.warning(f"No position to close for {pair}")
            return None

        # Apply slippage on exit
        slippage = settings.slippage_major if pair in MAJOR_PAIRS else settings.slippage_alt
        if position.direction == Direction.LONG:
            exit_price = current_price * (1 - slippage)
        else:
            exit_price = current_price * (1 + slippage)

        notional = position.quantity * exit_price
        exit_fee = notional * settings.taker_fee

        # Calculate PnL
        if position.direction == Direction.LONG:
            raw_pnl = (exit_price - position.entry_price) * position.quantity
        else:
            raw_pnl = (position.entry_price - exit_price) * position.quantity

        net_pnl = raw_pnl - position.entry_fee - exit_fee
        pnl_pct = net_pnl / position.margin_used if position.margin_used > 0 else 0

        # Return margin + PnL to balance
        self.balance += position.margin_used + net_pnl
        self._peak_balance = max(self._peak_balance, self.balance)

        now = datetime.utcnow()
        hold_minutes = (now - position.opened_at).total_seconds() / 60

        trade = Trade(
            id=position.id,
            pair=pair,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            leverage=position.leverage,
            pnl=round(net_pnl, 4),
            pnl_pct=round(pnl_pct, 4),
            entry_fee=position.entry_fee,
            exit_fee=exit_fee,
            margin_used=position.margin_used,
            hold_time_minutes=round(hold_minutes, 2),
            opened_at=position.opened_at,
            closed_at=now,
            entry_reasoning=position.entry_reasoning,
            exit_reasoning=reason,
        )

        # Update DB
        await self.db.update_trade(position.id, {
            "exit_price": exit_price,
            "pnl": trade.pnl,
            "pnl_pct": trade.pnl_pct,
            "exit_fee": exit_fee,
            "hold_time_minutes": hold_minutes,
            "closed_at": now.isoformat(),
            "exit_reasoning": reason,
            "status": "closed",
        })

        del self.positions[pair]

        emoji = "+" if net_pnl > 0 else ""
        logger.info(
            f"CLOSED {position.direction.value} {pair} @ {exit_price:.4f} "
            f"PnL={emoji}{net_pnl:.4f} ({pnl_pct:.2%}) hold={hold_minutes:.1f}m"
        )
        return trade

    def update_position_price(self, pair: str, current_price: float):
        """Update unrealized PnL for an open position."""
        position = self.positions.get(pair)
        if not position:
            return
        position.current_price = current_price
        if position.direction == Direction.LONG:
            position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
        else:
            position.unrealized_pnl = (position.entry_price - current_price) * position.quantity

    def apply_funding_rate(self, pair: str, funding_rate: float):
        """Apply funding rate cost/credit to an open position (every 8h on Binance Futures).
        Positive rate: longs pay shorts. Negative rate: shorts pay longs."""
        position = self.positions.get(pair)
        if not position:
            return
        notional = position.quantity * position.current_price
        # Longs pay when rate is positive, shorts pay when rate is negative
        if position.direction == Direction.LONG:
            cost = notional * funding_rate  # positive = cost, negative = credit
        else:
            cost = -notional * funding_rate  # inverted for shorts
        self.balance -= cost
        logger.info(f"Funding rate {pair}: {funding_rate:.6f}, cost=${cost:.4f} ({position.direction.value})")

    def check_stop_loss_take_profit(self, pair: str, current_price: float) -> Optional[str]:
        """Check if SL/TP has been hit. Returns 'sl' or 'tp' or None."""
        position = self.positions.get(pair)
        if not position:
            return None

        if position.direction == Direction.LONG:
            if position.stop_loss and current_price <= position.stop_loss:
                return "sl"
            if position.take_profit and current_price >= position.take_profit:
                return "tp"
        else:
            if position.stop_loss and current_price >= position.stop_loss:
                return "sl"
            if position.take_profit and current_price <= position.take_profit:
                return "tp"
        return None

    @property
    def total_equity(self) -> float:
        """Balance + unrealized PnL of all positions."""
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        margin_locked = sum(p.margin_used for p in self.positions.values())
        return self.balance + margin_locked + unrealized

    @property
    def total_margin_used(self) -> float:
        return sum(p.margin_used for p in self.positions.values())

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak."""
        if self._peak_balance == 0:
            return 0
        return (self.total_equity - self._peak_balance) / self._peak_balance

    @property
    def daily_pnl(self) -> float:
        """PnL since the start (approximation - proper daily tracking is in DailyStats)."""
        return self.total_equity - self.initial_balance

    @property
    def daily_pnl_pct(self) -> float:
        if self.initial_balance == 0:
            return 0
        return self.daily_pnl / self.initial_balance
