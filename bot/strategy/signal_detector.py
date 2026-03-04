"""1H timeframe signal detection — volume_spike strategy.

Mega backtest winner (1.1M+ simulations, 3 phases, 2 years, 19 pairs):
- volume_spike: volume_ratio > 2.5 + ADX > 20 + MACD direction + EMA alignment
- $231K profit, PF=1.29, WR=48%, 21/25 green months, 19/19 pairs profitable

Fixed parameters from 2-year mega backtest:
- SL: 3%, TP: 4%, Trailing: 6%, Cooldown: 48h
- Leverage: 10x, Position: 2%
- Entry: maker fee, TP exit: maker fee, SL exit: taker + slippage
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from core.models import ActionType, Direction, MarketSnapshot, TradeDecision

logger = logging.getLogger(__name__)

# ============================================================
# MEGA BACKTEST-WINNING PARAMETERS (DO NOT CHANGE)
# 1.1M simulations, 3 phases, walk-forward ROBUST, 0 overfit
# ============================================================
SL_PCT = 0.03          # 3% stop loss
TP_PCT = 0.04          # 4% take profit (was 6%, changed to 4% — WR 48% vs 28%)
TRAILING_PCT = 0.06    # 6% trailing stop
COOLDOWN_HOURS = 48    # 48 hours between trades per pair
LEVERAGE = 10          # 10x leverage
POSITION_PCT = 0.02    # 2% of equity per trade
MIN_SCORE = 5          # Minimum signal score to enter


@dataclass
class Signal:
    pair: str
    direction: str  # "LONG" or "SHORT"
    pattern_type: str  # volume_spike
    score: float
    reason: str
    indicators_used: list[str] = field(default_factory=list)


class SignalDetector:
    """1H rule-based signal detection. No LLM calls — pure deterministic logic."""

    def __init__(self):
        pass

    def detect(self, snapshot: MarketSnapshot) -> Optional[Signal]:
        """Run volume_spike detector and return signal or None."""
        if snapshot.macd_signal is None:
            return None
        if snapshot.ema_alignment is None:
            return None

        signal = self._detect_volume_spike(snapshot)
        if signal and signal.score >= MIN_SCORE:
            logger.info(
                f"Signal detected: {signal.pattern_type} {signal.direction} on {signal.pair} "
                f"(score={signal.score:.0f}) - {signal.reason}"
            )
            return signal

        return None

    def _detect_volume_spike(self, snapshot: MarketSnapshot) -> Optional[Signal]:
        """Volume spike + directional bias on 1H.

        Mega backtest winner: $231K, PF=1.29, WR=48%, 19/19 pairs.
        Logic identical to backtest (strategies_mega.py detect_volume_spike).

        Conditions:
        - volume_ratio > 2.5 (current volume 2.5x above SMA20)
        - ADX > 20 (some trend present)
        - MACD confirms direction (bullish/bearish)
        - EMA alignment confirms direction (> 0 for LONG, < 0 for SHORT)
        """
        volume_ratio = snapshot.volume_ratio or 0
        adx = snapshot.adx
        macd_signal = snapshot.macd_signal
        ema_alignment = snapshot.ema_alignment

        if adx is None or macd_signal is None or ema_alignment is None:
            return None
        if volume_ratio <= 2.5 or adx <= 20:
            return None

        score = 5
        if volume_ratio > 3.5:
            score += 1
        if volume_ratio > 5.0:
            score += 1

        indicators = [
            f"vol={volume_ratio:.1f}x",
            f"ADX={adx:.0f}",
            f"MACD={macd_signal}",
            f"EMA_align={ema_alignment:+.1f}",
        ]

        # LONG: MACD bullish + EMA alignment positive
        if macd_signal in ("bullish", "bullish_cross") and ema_alignment > 0:
            return Signal(
                pair=snapshot.pair,
                direction="LONG",
                pattern_type="volume_spike",
                score=score,
                reason=f"Vol spike {volume_ratio:.1f}x + MACD bullish + EMA aligned: {', '.join(indicators)}",
                indicators_used=indicators,
            )

        # SHORT: MACD bearish + EMA alignment negative
        if macd_signal in ("bearish", "bearish_cross") and ema_alignment < 0:
            return Signal(
                pair=snapshot.pair,
                direction="SHORT",
                pattern_type="volume_spike",
                score=score,
                reason=f"Vol spike {volume_ratio:.1f}x + MACD bearish + EMA aligned: {', '.join(indicators)}",
                indicators_used=indicators,
            )

        return None

    def calibrate(self, signal: Signal, snapshot: MarketSnapshot, balance: float, regime: str = "unknown") -> TradeDecision:
        """Deterministic calibration with mega backtest-winning parameters. Zero API calls.

        Fixed params: SL=3%, TP=4%, Trail=6%, Lev=10x, Pos=2%
        These parameters produced +$231K over 2 years across 19 pairs.
        """
        price = snapshot.price

        if signal.direction == "LONG":
            sl = price * (1 - SL_PCT)
            tp = price * (1 + TP_PCT)
            action = ActionType.ENTER_LONG
            direction = Direction.LONG
        else:
            sl = price * (1 + SL_PCT)
            tp = price * (1 - TP_PCT)
            action = ActionType.ENTER_SHORT
            direction = Direction.SHORT

        # Confidence from signal score (base 0.65, increases with score)
        confidence = min(0.65 + (signal.score - MIN_SCORE) * 0.05, 0.95)

        decision = TradeDecision(
            action=action,
            pair=signal.pair,
            direction=direction,
            leverage=LEVERAGE,
            position_size_pct=POSITION_PCT,
            entry_price=price,
            stop_loss=round(sl, 6),
            take_profit=round(tp, 6),
            reasoning=f"[{signal.pattern_type}] {signal.reason[:80]}",
            confidence=round(confidence, 2),
        )

        logger.info(
            f"[{signal.pair}] Calibrated {signal.direction}: "
            f"SL={sl:.2f} ({SL_PCT*100:.0f}%) TP={tp:.2f} ({TP_PCT*100:.0f}%) "
            f"lev={LEVERAGE}x size={POSITION_PCT*100:.0f}% R:R=1:{TP_PCT/SL_PCT:.1f}"
        )
        return decision
