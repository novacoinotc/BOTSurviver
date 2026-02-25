"""Claude as the trader: analyzes market data, makes trade decisions."""

import json
import logging
from datetime import datetime
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

TRADE_DECISION_SYSTEM = """You are an elite Binance Futures scalping AI with full autonomy over trading decisions.
Your goal: maximize net PnL while managing risk. You learn from every trade.

INDICATORS YOU RECEIVE:
- RSI (7/14), Stochastic RSI (K/D) - overbought/oversold detection
- EMA (9/21/50) - trend direction and crossovers
- MACD (histogram + signal) - momentum confirmation
- Bollinger Bands (pct, width, squeeze) - volatility + breakout detection
- ADX + DI+/DI- - trend STRENGTH (>25 = trending, <20 = ranging)
- MFI - volume-weighted momentum (divergences = reversal signals)
- ATR (14) + ATR% - volatility for SL/TP sizing
- VWAP - institutional reference price
- Volume Delta - buy vs sell pressure
- Book Imbalance - order book pressure
- Funding Rate - cost of holding positions
- Fear & Greed Index - macro sentiment

STRATEGY FREEDOM:
- You can try ANY scalping strategy: momentum, mean reversion, breakout, trend following
- You can experiment with different leverage (1-10x), SL/TP ratios, hold times
- Learn from your past trades (provided in context) - if a strategy keeps losing, STOP using it
- If a pattern keeps winning, increase confidence and position size
- Use ADX to choose strategy: ADX>25 = trend follow, ADX<20 = mean revert
- Use BB squeeze to detect imminent breakouts
- Use Stochastic RSI for precise entry timing in overbought/oversold zones

HARD RULES (cannot be violated):
- Respond with EXACTLY ONE JSON object (no markdown)
- Confidence 0.0-1.0; only trade above 0.6
- Max {max_positions} open positions, max 1 per pair
- BOTH stop_loss AND take_profit MANDATORY on every entry
- SL should be based on ATR (1-2x ATR from entry)
- Risk per trade: max {risk_pct}% of capital
- In extreme fear (Fear&Greed < 15): reduce position sizes, higher confidence threshold

RESPONSE FORMAT:
{{
  "action": "ENTER_LONG" | "ENTER_SHORT" | "EXIT" | "ADJUST" | "HOLD",
  "pair": "BTCUSDT",
  "direction": "LONG" | "SHORT",
  "leverage": 3,
  "position_size_pct": 0.005,
  "stop_loss": 97000.0,
  "take_profit": 98000.0,
  "reasoning": "Why this decision",
  "confidence": 0.75
}}

For HOLD: {{"action": "HOLD", "pair": "BTCUSDT", "reasoning": "...", "confidence": 0.0}}
For EXIT: {{"action": "EXIT", "pair": "BTCUSDT", "reasoning": "...", "confidence": 0.8}}
For ADJUST: {{"action": "ADJUST", "pair": "BTCUSDT", "stop_loss": 97500, "take_profit": 98500, "reasoning": "...", "confidence": 0.7}}
"""

DEEP_ANALYSIS_SYSTEM = """You are a senior quantitative trader doing a deep market review.
Analyze the provided trading data and provide:

1. **Market Regime**: trending_up, trending_down, ranging, or volatile
2. **Trade Reviews**: For each recent trade, assess what went right/wrong and write a lesson_learned
3. **Rule Proposals**: If you see patterns, propose learned rules
4. **Parameter Suggestions**: Recommend changes to bot parameters if needed

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
  "overall_assessment": "Brief summary of market conditions and bot performance"
}}
"""


class ClaudeTrader:
    """Uses Claude to make trade decisions and perform deep analysis."""

    def __init__(self, db: Database):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.db = db

    async def make_decision(
        self,
        snapshot: MarketSnapshot,
        open_positions: list[dict],
        similar_trades: list[dict],
        active_rules: list[dict],
        current_params: dict,
        balance: float,
        pattern_stats: dict = None,
    ) -> TradeDecision:
        """Ask Claude Haiku to make a trade decision for a single pair."""

        # Build compact prompt
        has_position = any(p["pair"] == snapshot.pair for p in open_positions)

        user_prompt = self._build_trade_prompt(
            snapshot, open_positions, similar_trades, active_rules, current_params, balance, has_position, pattern_stats
        )

        system = TRADE_DECISION_SYSTEM.format(
            max_positions=int(current_params.get("max_open_positions", settings.max_open_positions)),
            risk_pct=current_params.get("max_risk_per_trade_pct", settings.max_risk_per_trade_pct) * 100,
        )

        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=400,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Track cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (input_tokens / 1_000_000 * HAIKU_INPUT_COST) + (output_tokens / 1_000_000 * HAIKU_OUTPUT_COST)

            await self.db.insert_api_cost({
                "service": "claude_haiku",
                "tokens_in": input_tokens,
                "tokens_out": output_tokens,
                "cost_usd": round(cost, 6),
                "purpose": "trade_decision",
                "created_at": datetime.utcnow().isoformat(),
            })

            # Parse response
            text = response.content[0].text.strip()
            # Clean potential markdown wrapping
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            decision_data = json.loads(text)
            decision = TradeDecision(
                action=ActionType(decision_data["action"]),
                pair=decision_data.get("pair", snapshot.pair),
                direction=Direction(decision_data["direction"]) if decision_data.get("direction") else None,
                leverage=decision_data.get("leverage"),
                position_size_pct=decision_data.get("position_size_pct"),
                entry_price=decision_data.get("entry_price"),
                stop_loss=decision_data.get("stop_loss"),
                take_profit=decision_data.get("take_profit"),
                reasoning=decision_data.get("reasoning", ""),
                confidence=decision_data.get("confidence", 0.0),
            )

            logger.info(
                f"[{snapshot.pair}] Claude decision: {decision.action.value} "
                f"conf={decision.confidence:.2f} - {decision.reasoning[:80]}"
            )
            return decision

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response: {e}")
            return TradeDecision(action=ActionType.HOLD, pair=snapshot.pair, reasoning=f"Parse error: {e}")
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return TradeDecision(action=ActionType.HOLD, pair=snapshot.pair, reasoning=f"API error: {e}")

    async def deep_analysis(
        self,
        recent_trades: list[dict],
        current_params: dict,
        market_summary: dict,
        memories: list[dict],
    ) -> dict:
        """Use Claude Sonnet for deep analysis: trade reviews, regime detection, optimization."""

        user_prompt = f"""## Deep Market Analysis Request

### Market Summary
{json.dumps(market_summary, indent=2)}

### Current Parameters
{json.dumps(current_params, indent=2)}

### Recent Trades (last 24h)
{json.dumps(recent_trades[:30], indent=2, default=str)}

### Recent Memory/Lessons
{json.dumps(memories[:10], indent=2, default=str)}

Please analyze and provide:
1. Current market regime
2. Review of each trade (lesson_learned for each)
3. Any patterns you notice → propose as rules
4. Parameter adjustment suggestions if needed
"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6-20250514",
                max_tokens=2000,
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

    def _build_trade_prompt(
        self,
        snapshot: MarketSnapshot,
        open_positions: list[dict],
        similar_trades: list[dict],
        active_rules: list[dict],
        current_params: dict,
        balance: float,
        has_position: bool,
        pattern_stats: dict = None,
    ) -> str:
        """Build compact prompt for trade decision with full context."""
        snap_dict = snapshot.model_dump(exclude_none=True)
        snap_dict["timestamp"] = snap_dict["timestamp"].isoformat()

        parts = [
            f"## Market Snapshot\n{json.dumps(snap_dict, indent=2)}",
            f"\n## Account\nBalance: ${balance:.2f} | Open positions: {len(open_positions)}/{int(current_params.get('max_open_positions', 5))}",
        ]

        if has_position:
            pos = next(p for p in open_positions if p["pair"] == snapshot.pair)
            parts.append(f"\n## Current Position on {snapshot.pair}\n{json.dumps(pos, indent=2)}")

        if open_positions:
            other = [p for p in open_positions if p["pair"] != snapshot.pair]
            if other:
                summary = [{"pair": p["pair"], "dir": p["direction"], "pnl": p["unrealized_pnl"]} for p in other]
                parts.append(f"\n## Other Open Positions\n{json.dumps(summary)}")

        # Win-rate statistics for this pattern
        if pattern_stats and pattern_stats.get("total", 0) > 0:
            parts.append(f"\n## Historical Performance for {snapshot.pair}\n{pattern_stats['summary']}")

        if similar_trades:
            lessons = []
            for t in similar_trades[:5]:
                lessons.append({
                    "pair": t["pair"],
                    "direction": t["direction"],
                    "pnl": t["pnl"],
                    "pnl_pct": t.get("pnl_pct", 0),
                    "lesson": t.get("lesson_learned", ""),
                    "regime": t["market_regime"],
                })
            parts.append(f"\n## Lessons from Similar Trades\n{json.dumps(lessons, indent=2)}")

        if active_rules:
            rules_with_stats = []
            for r in active_rules[:5]:
                applied = r.get("times_applied", 0)
                success = r.get("times_successful", 0)
                rate = f" ({success}/{applied} successful)" if applied > 0 else ""
                rules_with_stats.append(f"- {r['rule']}{rate}")
            parts.append(f"\n## Active Learned Rules\n" + "\n".join(rules_with_stats))

        if has_position:
            parts.append("\n## Task\nDecide: EXIT, ADJUST, or HOLD for this position. Consider funding rate cost if holding long.")
        else:
            parts.append(f"\n## Task\nDecide: ENTER_LONG, ENTER_SHORT, or HOLD for {snapshot.pair}. If entering, set SL based on ATR and TP with min 1.5:1 reward:risk.")

        return "\n".join(parts)
