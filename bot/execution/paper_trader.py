"""Paper trading simulator with realistic Binance Futures execution."""

import logging
import uuid
from datetime import datetime
from typing import Optional

from config.settings import settings, MAJOR_PAIRS
from core.models import Direction, Position, Trade, TradeDecision
from db.database import Database

logger = logging.getLogger(__name__)


class PaperTrader:
    """Simulates Binance Futures execution with real fees, slippage, liquidation, and funding."""

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

        # MAKER fee on entry (limit order — no slippage for maker execution)
        entry_price = current_price  # limit order fills at exact price

        quantity = notional / entry_price

        # Maker fee on entry (limit order)
        entry_fee = notional * settings.maker_fee

        # Deduct margin + fee from balance
        total_cost = margin + entry_fee
        if total_cost > self.balance:
            logger.warning(f"Insufficient balance for {pair}: need {total_cost:.2f}, have {self.balance:.2f}")
            return None

        self.balance -= total_cost

        # Calculate liquidation price (isolated margin)
        # Long: liq = entry * (1 - 1/leverage + maintenance_margin_rate)
        # Short: liq = entry * (1 + 1/leverage - maintenance_margin_rate)
        maint_rate = 0.004  # Binance default maintenance margin rate
        if decision.direction == Direction.LONG:
            liq_price = entry_price * (1 - (1 / leverage) + maint_rate)
        else:
            liq_price = entry_price * (1 + (1 / leverage) - maint_rate)

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
            highest_price=entry_price,
            lowest_price=entry_price,
            liquidation_price=liq_price,
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
            f"qty={quantity:.6f} lev={leverage}x margin={margin:.2f} fee={entry_fee:.4f} "
            f"SL={decision.stop_loss} TP={decision.take_profit} liq={liq_price:.4f}"
        )
        return position

    async def close_position(self, pair: str, current_price: float, reason: str = "") -> Optional[Trade]:
        """Close an existing position.

        Fee logic (matches backtest):
        - Take Profit: MAKER fee (limit order, no slippage)
        - Stop Loss / Trailing / Other: TAKER fee + slippage (market order)
        """
        position = self.positions.get(pair)
        if not position:
            logger.warning(f"No position to close for {pair}")
            return None

        # TP exits use maker fee (limit order, no slippage)
        # SL/trailing/other exits use taker fee + slippage (market order)
        is_tp_exit = "take profit" in reason.lower() or "tp" in reason.lower()

        if is_tp_exit:
            exit_price = current_price  # limit order fills at exact price
            notional = position.quantity * exit_price
            exit_fee = notional * settings.maker_fee
        else:
            # Apply slippage for market exits (SL, trailing, stale, liquidation)
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

        # Net PnL = raw - entry fee - exit fee - funding paid
        net_pnl = raw_pnl - position.entry_fee - exit_fee - position.funding_paid
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
            f"PnL={emoji}{net_pnl:.4f} ({pnl_pct:.2%}) hold={hold_minutes:.1f}m "
            f"funding_paid={position.funding_paid:.4f}"
        )
        return trade

    def update_position_price(self, pair: str, current_price: float):
        """Update unrealized PnL and peak/trough tracking for a position."""
        position = self.positions.get(pair)
        if not position:
            return
        position.current_price = current_price

        # Track peak/trough for trailing stops
        if current_price > position.highest_price:
            position.highest_price = current_price
        if current_price < position.lowest_price:
            position.lowest_price = current_price

        # Update unrealized PnL (includes funding costs)
        if position.direction == Direction.LONG:
            position.unrealized_pnl = (current_price - position.entry_price) * position.quantity - position.funding_paid
        else:
            position.unrealized_pnl = (position.entry_price - current_price) * position.quantity - position.funding_paid

    def apply_funding_rate(self, pair: str, funding_rate: float) -> float:
        """Apply funding rate cost/credit to an open position.
        Returns the cost amount (positive = paid, negative = received)."""
        position = self.positions.get(pair)
        if not position:
            return 0.0
        notional = position.quantity * position.current_price
        # Longs pay when rate is positive, shorts pay when rate is negative
        if position.direction == Direction.LONG:
            cost = notional * funding_rate
        else:
            cost = -notional * funding_rate
        self.balance -= cost
        position.funding_paid += cost
        logger.info(
            f"Funding {pair}: rate={funding_rate:.6f}, cost=${cost:.4f} "
            f"({position.direction.value}), total_funding=${position.funding_paid:.4f}"
        )
        return cost

    def check_stop_loss_take_profit(self, pair: str, current_price: float) -> Optional[str]:
        """Check if SL/TP has been hit. Returns 'sl', 'tp', 'liq', or None."""
        position = self.positions.get(pair)
        if not position:
            return None

        # Check liquidation first (most critical)
        if position.direction == Direction.LONG:
            if position.liquidation_price > 0 and current_price <= position.liquidation_price:
                return "liq"
            if position.stop_loss and current_price <= position.stop_loss:
                return "sl"
            if position.take_profit and current_price >= position.take_profit:
                return "tp"
        else:
            if position.liquidation_price > 0 and current_price >= position.liquidation_price:
                return "liq"
            if position.stop_loss and current_price >= position.stop_loss:
                return "sl"
            if position.take_profit and current_price <= position.take_profit:
                return "tp"

        return None

    def update_trailing_stops(self, pair: str):
        """Update trailing stop based on price movement.

        Matches backtest simulator logic exactly:
        - LONG: new_sl = highest_price * (1 - trailing_pct), apply if new_sl > current SL
        - SHORT: new_sl = lowest_price * (1 + trailing_pct), apply if new_sl < current SL
        No entry_price guard — trail activates as soon as math allows it.
        """
        position = self.positions.get(pair)
        if not position:
            return

        # Prefer percentage-based trailing (matches backtest)
        if position.trailing_stop_pct and position.trailing_stop_pct > 0:
            if position.direction == Direction.LONG:
                new_sl = position.highest_price * (1 - position.trailing_stop_pct)
                if new_sl > position.stop_loss:
                    position.stop_loss = new_sl
            else:
                new_sl = position.lowest_price * (1 + position.trailing_stop_pct)
                if 0 < new_sl < position.stop_loss:
                    position.stop_loss = new_sl
        elif position.trailing_stop_distance and position.trailing_stop_distance > 0:
            # Legacy: fixed distance trailing
            if position.direction == Direction.LONG:
                new_sl = position.highest_price - position.trailing_stop_distance
                if new_sl > position.stop_loss:
                    position.stop_loss = new_sl
            else:
                new_sl = position.lowest_price + position.trailing_stop_distance
                if 0 < new_sl < position.stop_loss:
                    position.stop_loss = new_sl

    def set_trailing_stop(self, pair: str, distance: float):
        """Enable trailing stop for a position with given ATR-based distance."""
        position = self.positions.get(pair)
        if position:
            position.trailing_stop_distance = distance
            logger.info(f"Trailing stop set for {pair}: distance={distance:.4f}")

    def set_trailing_stop_pct(self, pair: str, pct: float):
        """Enable percentage-based trailing stop (matches backtest simulator)."""
        position = self.positions.get(pair)
        if position:
            position.trailing_stop_pct = pct
            logger.info(f"Trailing stop set for {pair}: pct={pct*100:.1f}%")

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
    def margin_ratio(self) -> float:
        """Margin ratio: how much of equity is used as margin. >0.8 = dangerous."""
        equity = self.total_equity
        if equity <= 0:
            return 1.0
        return self.total_margin_used / equity

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak."""
        if self._peak_balance == 0:
            return 0
        return (self.total_equity - self._peak_balance) / self._peak_balance

    @property
    def daily_pnl(self) -> float:
        return self.total_equity - self.initial_balance

    @property
    def daily_pnl_pct(self) -> float:
        if self.initial_balance == 0:
            return 0
        return self.daily_pnl / self.initial_balance
