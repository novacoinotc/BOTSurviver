"""Claude as the trader: analyzes market data, makes trade decisions."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic

from config.settings import settings
from core.models import (
    ActionType,
    Direction,
    MarketSnapshot,
    TradeDecision,
    ApiCost,
)
from db.database import Database

logger = logging.getLogger(__name__)

# Cost per 1M tokens (approximate)
HAIKU_INPUT_COST = 1.00   # $1/MTok
HAIKU_OUTPUT_COST = 5.00  # $5/MTok
SONNET_INPUT_COST = 3.00  # $3/MTok
SONNET_OUTPUT_COST = 15.00  # $15/MTok

CALIBRATION_SYSTEM = """You calibrate SL/TP/leverage/size for a detected {direction} {pattern_type} signal on {pair}.

Given: price, ATR, pattern type, regime, key indicators.
Set: stop_loss, take_profit, leverage (2-8), position_size_pct (0.003-0.01).

RULES:
- SL >= 2x ATR from entry (NEVER less than 0.6% from entry)
- TP >= SL distance (R:R >= 1:1)
- Regime calibration:
  - ranging: SL 2x ATR, TP 1-1.5x ATR, leverage 3-5
  - trending: SL 2.5x ATR, TP 2-2.5x ATR, leverage 3-6
  - volatile: SL 3x ATR, TP 2x ATR, leverage 2-4, size 0.003-0.005
- SL/TP MUST be absolute prices (not percentages)
- {direction} LONG: stop_loss < price < take_profit
- {direction} SHORT: take_profit < price < stop_loss

Respond with EXACTLY ONE JSON object (no extra text):
{{"stop_loss": 97000.0, "take_profit": 98500.0, "leverage": 3, "position_size_pct": 0.005, "reasoning": "max 30 words", "confidence": 0.75}}
"""

POSITION_MGMT_SYSTEM = """You manage an open {direction} position on {pair}.

Given: entry, current price, SL, TP, indicators, hold time, PnL%.
Decide: EXIT / ADJUST / HOLD.

EXIT if: reversal signals detected — MACD cross against, RSI divergence against, EMA cross against position.
ADJUST if: profit >= 0.5% → move SL to breakeven; profit >= 1% → tighten TP closer.
HOLD if: no reversal signals, let SL/TP work. Small drawdowns (-0.3 to -0.5%) are normal noise.

A -0.3% exit on reversal is better than a -1.5% SL hit. But don't panic-exit on noise.

Respond with EXACTLY ONE JSON object (no extra text):
{{"action": "HOLD", "reasoning": "max 30 words", "confidence": 0.7}}
{{"action": "EXIT", "reasoning": "max 30 words", "confidence": 0.8}}
{{"action": "ADJUST", "stop_loss": 97500.0, "take_profit": 98500.0, "reasoning": "max 30 words", "confidence": 0.7}}
"""

DEEP_ANALYSIS_SYSTEM = """You are a senior quantitative trader doing a deep market review.
Analyze the provided trading data and provide actionable intelligence.

Focus on:
1. **Market Regime**: trending_up, trending_down, ranging, or volatile (with confidence)
2. **Trade Reviews**: For each trade, what went RIGHT and what went WRONG. Be specific.
3. **Pattern Discovery**: Look for winning patterns - specific indicator combinations that work
4. **Rule Proposals**: Concrete, testable rules (e.g., "LONG when RSI<30 AND ADX>25 AND EMA_alignment>0")
5. **Parameter Tuning**: Based on win rate and hold times, suggest parameter changes
6. **Strategy Assessment**: Which strategies are working? Which should be abandoned?

Respond as JSON:
{{
  "market_regime": "trending_up",
  "trade_reviews": [
    {{"trade_id": "abc123", "assessment": "good/bad", "lesson_learned": "...", "tags": ["momentum", "reversal"]}}
  ],
  "proposed_rules": [
    {{"rule": "Avoid LONG when RSI_14 > 75 and MACD bearish_cross", "confidence": 0.7}}
  ],
  "parameter_suggestions": [
    {{"param": "default_leverage", "current": 3, "suggested": 4, "reasoning": "..."}}
  ],
  "strategies_working": ["trend_following on BTC/ETH"],
  "strategies_failing": ["mean_reversion on altcoins in trending market"],
  "overall_assessment": "Brief summary of market conditions and bot performance"
}}
"""


class ClaudeTrader:
    """Uses Claude to make trade decisions and perform deep analysis."""

    def __init__(self, db: Database):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.db = db

    async def calibrate_trade(
        self,
        signal,
        snapshot: MarketSnapshot,
        balance: float,
        params: dict,
        market_regime: str = "unknown",
    ) -> TradeDecision:
        """Claude calibrates SL/TP/leverage/size for a detected signal."""
        price = snapshot.price
        atr = snapshot.atr_14 or (price * 0.005)  # fallback 0.5%
        atr_pct = snapshot.atr_pct or 0.5

        user_prompt = (
            f"Price: {price}, ATR: {atr:.4f} ({atr_pct:.3f}%), Regime: {market_regime}\n"
            f"Pattern: {signal.pattern_type}, Score: {signal.score}, Reason: {signal.reason}\n"
            f"SL hint: {signal.sl_hint_atr_mult}x ATR, TP hint: {signal.tp_hint_atr_mult}x ATR\n"
            f"Balance: ${balance:.2f}\n"
            f"Key indicators: RSI={snapshot.rsi_14}, ADX={snapshot.adx}, "
            f"EMA_align={snapshot.ema_alignment}, MACD={snapshot.macd_signal}, "
            f"BB%={snapshot.bb_pct}, StochK={snapshot.stoch_rsi_k}, "
            f"vol_ratio={snapshot.volume_ratio}, book_imb={snapshot.book_imbalance}"
        )

        system = CALIBRATION_SYSTEM.format(
            direction=signal.direction,
            pattern_type=signal.pattern_type,
            pair=signal.pair,
        )

        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=400,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
            )

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (input_tokens / 1_000_000 * HAIKU_INPUT_COST) + (output_tokens / 1_000_000 * HAIKU_OUTPUT_COST)

            await self.db.insert_api_cost({
                "service": "claude_haiku",
                "tokens_in": input_tokens,
                "tokens_out": output_tokens,
                "cost_usd": round(cost, 6),
                "purpose": "calibration",
                "created_at": datetime.utcnow().isoformat(),
            })

            data = self._parse_json_response(response.content[0].text)

            action = ActionType.ENTER_LONG if signal.direction == "LONG" else ActionType.ENTER_SHORT
            direction = Direction.LONG if signal.direction == "LONG" else Direction.SHORT

            decision = TradeDecision(
                action=action,
                pair=signal.pair,
                direction=direction,
                leverage=data.get("leverage", 3),
                position_size_pct=data.get("position_size_pct", 0.005),
                entry_price=price,
                stop_loss=data.get("stop_loss"),
                take_profit=data.get("take_profit"),
                reasoning=f"[{signal.pattern_type}] {data.get('reasoning', signal.reason)[:80]}",
                confidence=data.get("confidence", 0.7),
            )

            logger.info(
                f"[{signal.pair}] Calibrated {signal.direction}: "
                f"SL={decision.stop_loss} TP={decision.take_profit} "
                f"lev={decision.leverage} size={decision.position_size_pct} "
                f"conf={decision.confidence:.2f}"
            )
            return decision

        except Exception as e:
            logger.warning(f"Calibration API failed ({e}), using deterministic fallback")
            return self._fallback_calibration(signal, snapshot)

    async def manage_position(
        self,
        snapshot: MarketSnapshot,
        position,
        params: dict,
        market_regime: str = "unknown",
    ) -> TradeDecision:
        """Claude manages an open position: EXIT / ADJUST / HOLD."""
        hold_minutes = (datetime.utcnow() - position.opened_at).total_seconds() / 60
        pnl_pct = position.unrealized_pnl / max(position.margin_used, 1) * 100

        user_prompt = (
            f"Pair: {snapshot.pair}, Direction: {position.direction.value}\n"
            f"Entry: {position.entry_price}, Current: {snapshot.price}, PnL: {pnl_pct:+.2f}%\n"
            f"SL: {position.stop_loss}, TP: {position.take_profit}, Hold: {hold_minutes:.0f}min\n"
            f"Regime: {market_regime}\n"
            f"Indicators: RSI={snapshot.rsi_14}, MACD={snapshot.macd_signal}, "
            f"EMA_align={snapshot.ema_alignment}, StochK={snapshot.stoch_rsi_k}, "
            f"ADX={snapshot.adx}, divergence={snapshot.rsi_divergence}, "
            f"vol_ratio={snapshot.volume_ratio}"
        )

        system = POSITION_MGMT_SYSTEM.format(
            direction=position.direction.value,
            pair=snapshot.pair,
        )

        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
            )

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (input_tokens / 1_000_000 * HAIKU_INPUT_COST) + (output_tokens / 1_000_000 * HAIKU_OUTPUT_COST)

            await self.db.insert_api_cost({
                "service": "claude_haiku",
                "tokens_in": input_tokens,
                "tokens_out": output_tokens,
                "cost_usd": round(cost, 6),
                "purpose": "position_mgmt",
                "created_at": datetime.utcnow().isoformat(),
            })

            data = self._parse_json_response(response.content[0].text)
            action_str = data.get("action", "HOLD").upper()

            decision = TradeDecision(
                action=ActionType(action_str),
                pair=snapshot.pair,
                direction=position.direction,
                stop_loss=data.get("stop_loss"),
                take_profit=data.get("take_profit"),
                reasoning=data.get("reasoning", ""),
                confidence=data.get("confidence", 0.5),
            )

            logger.info(
                f"[{snapshot.pair}] Manage: {decision.action.value} "
                f"conf={decision.confidence:.2f} - {decision.reasoning[:60]}"
            )
            return decision

        except Exception as e:
            logger.warning(f"Position mgmt API failed ({e}), defaulting to HOLD")
            return TradeDecision(
                action=ActionType.HOLD,
                pair=snapshot.pair,
                reasoning=f"API error, holding: {e}",
            )

    async def make_decision(
        self,
        snapshot: MarketSnapshot,
        open_positions: list[dict],
        similar_trades: list[dict],
        active_rules: list[dict],
        current_params: dict,
        balance: float,
        pattern_stats: dict = None,
        market_regime: str = "unknown",
        market_context: dict = None,
    ) -> TradeDecision:
        """Legacy method — kept for backward compatibility.
        New flow uses calibrate_trade() and manage_position() separately."""
        # For position management, delegate to manage_position
        has_position = any(p["pair"] == snapshot.pair for p in open_positions)
        if has_position:
            # Find the position object — but we only have dicts here
            # Return HOLD as the engine should use manage_position() directly
            return TradeDecision(
                action=ActionType.HOLD,
                pair=snapshot.pair,
                reasoning="Use manage_position() for position management",
            )
        # For new entries without signal detection, just HOLD
        return TradeDecision(
            action=ActionType.HOLD,
            pair=snapshot.pair,
            reasoning="Use signal_detector + calibrate_trade() for entries",
        )

    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON from Claude response, handling markdown fences and truncation."""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            fixed = text.rstrip()
            if not fixed.endswith("}"):
                last_quote = fixed.rfind('"')
                last_comma = fixed.rfind(",")
                last_brace = fixed.rfind("{")
                cutoff = max(last_comma, last_quote + 1)
                if cutoff > last_brace:
                    fixed = fixed[:cutoff].rstrip(",").rstrip() + "}"
                else:
                    fixed = fixed + '"}'
            return json.loads(fixed)

    def _fallback_calibration(self, signal, snapshot: MarketSnapshot) -> TradeDecision:
        """Deterministic fallback when Claude API fails."""
        price = snapshot.price
        atr = snapshot.atr_14 or (price * 0.005)
        sl_mult = signal.sl_hint_atr_mult
        tp_mult = signal.tp_hint_atr_mult

        if signal.direction == "LONG":
            sl = price - (atr * sl_mult)
            tp = price + (atr * tp_mult)
            action = ActionType.ENTER_LONG
            direction = Direction.LONG
        else:
            sl = price + (atr * sl_mult)
            tp = price - (atr * tp_mult)
            action = ActionType.ENTER_SHORT
            direction = Direction.SHORT

        return TradeDecision(
            action=action,
            pair=signal.pair,
            direction=direction,
            leverage=3,
            position_size_pct=0.005,
            entry_price=price,
            stop_loss=round(sl, 6),
            take_profit=round(tp, 6),
            reasoning=f"[{signal.pattern_type}] fallback: {signal.reason[:60]}",
            confidence=0.65,
        )

    async def deep_analysis(
        self,
        recent_trades: list[dict],
        current_params: dict,
        market_summary: dict,
        memories: list[dict],
    ) -> dict:
        """Use Claude Sonnet for deep analysis: trade reviews, regime detection, optimization."""

        # Load trading knowledge base for deeper analysis
        knowledge = ""
        knowledge_path = Path(__file__).parent / "trading_knowledge.md"
        if knowledge_path.exists():
            knowledge = knowledge_path.read_text()

        user_prompt = f"""## Deep Market Analysis Request

### Trading Knowledge Base
{knowledge}

### Market Summary
{json.dumps(market_summary, indent=2)}

### Current Parameters
{json.dumps(current_params, indent=2)}

### Recent Trades (last period)
{json.dumps(recent_trades[:50], indent=2, default=str)}

### Recent Memory/Lessons
{json.dumps(memories[:15], indent=2, default=str)}

Analyze deeply (apply the trading knowledge above to evaluate each trade):
1. Current market regime (with evidence)
2. Review each trade - what patterns worked/failed?
3. Propose concrete, testable rules with indicator thresholds
4. Which strategies work in current conditions?
5. Parameter adjustments if needed
"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4000,
                system=DEEP_ANALYSIS_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
            )

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (input_tokens / 1_000_000 * SONNET_INPUT_COST) + (output_tokens / 1_000_000 * SONNET_OUTPUT_COST)

            await self.db.insert_api_cost({
                "service": "claude_sonnet",
                "tokens_in": input_tokens,
                "tokens_out": output_tokens,
                "cost_usd": round(cost, 6),
                "purpose": "deep_analysis",
                "created_at": datetime.utcnow().isoformat(),
            })

            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            return json.loads(text)

        except Exception as e:
            logger.error(f"Deep analysis error: {e}")
            return {"error": str(e)}

