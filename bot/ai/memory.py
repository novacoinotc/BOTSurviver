"""Adaptive memory system: records trades, searches for similar ones, stores lessons."""

import json
import logging
from datetime import datetime
from typing import Optional

from core.models import Trade, TradeMemory, MarketRegime
from db.database import Database

logger = logging.getLogger(__name__)


class MemorySystem:
    """Records every trade with context, finds similar past trades, and stores lessons."""

    def __init__(self, db: Database):
        self.db = db

    async def record_trade(
        self,
        trade: Trade,
        indicators: dict,
        market_regime: MarketRegime = MarketRegime.UNKNOWN,
        sentiment_score: Optional[int] = None,
    ):
        """Record a closed trade in memory for future learning."""
        memory = {
            "trade_id": trade.id,
            "pair": trade.pair,
            "direction": trade.direction.value,
            "pnl": trade.pnl,
            "pnl_pct": trade.pnl_pct,
            "leverage": trade.leverage,
            "hold_time_minutes": trade.hold_time_minutes,
            "market_regime": market_regime.value,
            "indicators_at_entry": json.dumps(indicators),
            "sentiment_score": sentiment_score,
            "claude_reasoning": trade.entry_reasoning,
            "lesson_learned": "",
            "tags": "[]",
            "created_at": datetime.utcnow().isoformat(),
        }

        await self.db.insert_memory(memory)
        logger.info(f"Recorded trade memory: {trade.pair} {trade.direction.value} PnL={trade.pnl:.4f}")

    async def find_similar(
        self,
        pair: str,
        market_regime: str = "unknown",
        limit: int = 5,
    ) -> list[dict]:
        """Find similar past trades with win-rate statistics for Claude's prompt."""
        memories = await self.db.find_similar_trades(pair, market_regime, limit)
        for m in memories:
            if isinstance(m.get("indicators_at_entry"), str):
                try:
                    m["indicators_at_entry"] = json.loads(m["indicators_at_entry"])
                except json.JSONDecodeError:
                    m["indicators_at_entry"] = {}
            if isinstance(m.get("tags"), str):
                try:
                    m["tags"] = json.loads(m["tags"])
                except json.JSONDecodeError:
                    m["tags"] = []
        return memories

    async def get_pattern_stats(self, pair: str, direction: str, market_regime: str) -> dict:
        """Get win-rate statistics for a specific pattern (pair + direction + regime).
        Returns stats like: 'BTCUSDT LONG in trending_up: 7/10 trades won (70%)'
        """
        all_memories = await self.db.get_recent_memories(limit=500)
        matching = [
            m for m in all_memories
            if (m.get("pair") == pair or pair == "")
            and (m.get("direction") == direction or direction == "")
            and (m.get("market_regime") == market_regime or market_regime == "")
        ]
        total = len(matching)
        if total == 0:
            return {"total": 0, "wins": 0, "win_rate": 0, "avg_pnl_pct": 0, "summary": "No historical data"}

        wins = sum(1 for m in matching if m.get("pnl", 0) > 0)
        avg_pnl = sum(m.get("pnl_pct", 0) for m in matching) / total
        avg_hold = sum(m.get("hold_time_minutes", 0) for m in matching) / total
        best = max(m.get("pnl_pct", 0) for m in matching)
        worst = min(m.get("pnl_pct", 0) for m in matching)

        summary = (
            f"{pair or 'ALL'} {direction or 'ALL'} in {market_regime or 'ALL'}: "
            f"{wins}/{total} won ({wins/total*100:.0f}%), "
            f"avg PnL {avg_pnl:.2f}%, best {best:.2f}%, worst {worst:.2f}%, "
            f"avg hold {avg_hold:.0f}min"
        )

        return {
            "total": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": round(wins / total * 100, 1),
            "avg_pnl_pct": round(avg_pnl, 4),
            "avg_hold_minutes": round(avg_hold, 1),
            "best_pnl_pct": round(best, 4),
            "worst_pnl_pct": round(worst, 4),
            "summary": summary,
        }

    async def update_lessons(self, trade_reviews: list[dict]):
        """Update lessons learned from Claude's deep analysis."""
        for review in trade_reviews:
            trade_id = review.get("trade_id")
            lesson = review.get("lesson_learned", "")
            tags = review.get("tags", [])

            if not trade_id or not lesson:
                continue

            # Find the memory entry for this trade
            memories = await self.db.find_similar_trades(
                pair="", market_regime="", limit=100
            )
            for mem in memories:
                if mem.get("trade_id") == trade_id:
                    await self.db.update_memory_lesson(mem["id"], lesson, tags)
                    logger.info(f"Updated lesson for trade {trade_id}: {lesson[:60]}")
                    break

    async def add_rule(self, rule: str, source_trades: list[str], confidence: float = 0.5):
        """Add a new learned rule."""
        now = datetime.utcnow().isoformat()
        await self.db.insert_rule({
            "rule": rule,
            "source_trades": json.dumps(source_trades),
            "confidence": confidence,
            "times_applied": 0,
            "times_successful": 0,
            "active": 1,
            "created_at": now,
            "updated_at": now,
        })
        logger.info(f"New learned rule: {rule[:80]}")

    async def get_active_rules(self) -> list[dict]:
        """Get all active learned rules."""
        return await self.db.get_active_rules()

    async def get_recent_memories(self, limit: int = 20) -> list[dict]:
        """Get recent trade memories with lessons."""
        memories = await self.db.get_recent_memories(limit)
        for m in memories:
            if isinstance(m.get("indicators_at_entry"), str):
                try:
                    m["indicators_at_entry"] = json.loads(m["indicators_at_entry"])
                except json.JSONDecodeError:
                    m["indicators_at_entry"] = {}
            if isinstance(m.get("tags"), str):
                try:
                    m["tags"] = json.loads(m["tags"])
                except json.JSONDecodeError:
                    m["tags"] = []
        return memories

    async def get_stats(self) -> dict:
        """Get memory system stats."""
        memories = await self.db.get_recent_memories(limit=1000)
        rules = await self.db.get_active_rules()

        total = len(memories)
        with_lessons = sum(1 for m in memories if m.get("lesson_learned"))
        winning = sum(1 for m in memories if m.get("pnl", 0) > 0)

        return {
            "total_memories": total,
            "memories_with_lessons": with_lessons,
            "winning_trades": winning,
            "losing_trades": total - winning,
            "active_rules": len(rules),
            "top_rules": [
                {"rule": r["rule"][:100], "confidence": r["confidence"]}
                for r in rules[:5]
            ],
        }
