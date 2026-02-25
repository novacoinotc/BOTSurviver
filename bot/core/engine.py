"""Main engine: orchestrates the trading loop, connects all components."""

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional

from config.settings import settings
from config.pairs import get_top_pairs
from core.events import EventBus, EventType, Event
from core.models import ActionType, MarketRegime
from data.candles import CandleStore
from data.orderbook import OrderBookStore
from data.stream_manager import StreamManager
from db.database import Database
from strategy.market_analyzer import MarketAnalyzer
from ai.claude_trader import ClaudeTrader
from ai.memory import MemorySystem
from ai.optimizer import Optimizer
from ai.sentiment import SentimentAnalyzer
from execution.paper_trader import PaperTrader
from execution.position_manager import PositionManager
from risk.risk_manager import RiskManager
from risk.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class TradingEngine:
    """Main engine that connects all components and runs the trading loop."""

    def __init__(self):
        # Core
        self.db = Database()
        self.event_bus = EventBus()

        # Data
        self.candle_store = CandleStore()
        self.orderbook_store = OrderBookStore()
        self.stream_manager: Optional[StreamManager] = None

        # Strategy
        self.market_analyzer = MarketAnalyzer(self.candle_store, self.orderbook_store)

        # AI
        self.claude_trader = ClaudeTrader(self.db)
        self.memory = MemorySystem(self.db)
        self.optimizer = Optimizer(self.db)
        self.sentiment = SentimentAnalyzer(self.db)

        # Execution
        self.paper_trader = PaperTrader(self.db)
        self.position_manager = PositionManager(self.paper_trader, self.db)

        # Risk
        self.risk_manager = RiskManager()
        self.circuit_breaker = CircuitBreaker()

        # State
        self.pairs: list[str] = []
        self._running = False
        self._started_at: Optional[datetime] = None
        self._analysis_count = 0
        self._last_deep_analysis: Optional[datetime] = None
        self._last_optimization: Optional[datetime] = None
        self._current_regime = MarketRegime.UNKNOWN

    async def start(self):
        """Initialize all components and start the trading loop."""
        logger.info("=" * 60)
        logger.info("TRADING ENGINE STARTING")
        logger.info("=" * 60)

        # Connect DB
        await self.db.connect()

        # Get top pairs
        self.pairs = await get_top_pairs(count=20)
        logger.info(f"Trading pairs: {self.pairs}")

        # Initialize optimizer params
        await self.optimizer.initialize()
        params = await self.db.get_current_params()
        self.risk_manager.update_params(params)

        # Initialize position manager
        await self.position_manager.initialize()

        # Initialize circuit breaker
        self.circuit_breaker.initialize(self.paper_trader.total_equity)

        # Fetch initial sentiment
        await self.sentiment.fetch_fear_greed()
        await self.sentiment.fetch_news()
        self.market_analyzer.set_sentiment(self.sentiment.current_sentiment)
        self.market_analyzer.set_fear_greed(self.sentiment.fear_greed or 50)

        # Start WebSocket streams
        self.stream_manager = StreamManager(
            pairs=self.pairs,
            on_kline=self._on_kline,
            on_book_ticker=self._on_book_ticker,
            on_agg_trade=self._on_agg_trade,
        )
        await self.stream_manager.start()

        self._running = True
        self._started_at = datetime.utcnow()

        logger.info(
            f"Engine started. Balance: ${self.paper_trader.balance:.2f}, "
            f"Pairs: {len(self.pairs)}, Mode: {'PAPER' if settings.paper_trading else 'LIVE'}"
        )

        # Run main loops concurrently
        await asyncio.gather(
            self._analysis_loop(),
            self._sentiment_loop(),
            self._deep_analysis_loop(),
            self._optimization_loop(),
            self._daily_stats_loop(),
            self._health_check_loop(),
        )

    async def stop(self):
        """Gracefully shut down the engine."""
        logger.info("Stopping trading engine...")
        self._running = False
        if self.stream_manager:
            await self.stream_manager.stop()
        try:
            await self.position_manager.compute_daily_stats()
        except Exception as e:
            logger.warning(f"Could not compute final stats: {e}")
        try:
            await self.db.close()
        except Exception:
            pass
        logger.info("Engine stopped.")

    # --- WebSocket Handlers ---

    async def _on_kline(self, data: dict):
        """Handle incoming kline data."""
        self.candle_store.update_from_kline(data)

        # Check SL/TP on every price update
        pair = data["s"]
        kline = data["k"]
        price = float(kline["c"])

        self.paper_trader.update_position_price(pair, price)

        trigger = self.paper_trader.check_stop_loss_take_profit(pair, price)
        if trigger:
            reason = f"Stop loss hit" if trigger == "sl" else "Take profit hit"
            trade = await self.paper_trader.close_position(pair, price, reason)
            if trade:
                indicators = {}
                await self.memory.record_trade(trade, indicators, self._current_regime)

    async def _on_book_ticker(self, data: dict):
        """Handle incoming order book ticker."""
        self.orderbook_store.update_from_book_ticker(data)

    async def _on_agg_trade(self, data: dict):
        """Handle aggregate trade data (used for volume tracking)."""
        pass  # Volume is already tracked in candles

    # --- Main Analysis Loop ---

    async def _analysis_loop(self):
        """Main loop: analyze each pair every ~60 seconds."""
        # Wait for data to accumulate
        await asyncio.sleep(30)

        while self._running:
            try:
                self.position_manager.check_new_day()
                self.circuit_breaker.check_new_day(self.paper_trader.total_equity)

                params = await self.db.get_current_params()
                self.risk_manager.update_params(params)

                active_pairs = self.candle_store.pairs_with_data
                if not active_pairs:
                    counts = self.candle_store.get_candle_counts()
                    max_count = max(counts.values()) if counts else 0
                    total_pairs_receiving = len(counts)
                    logger.info(
                        f"Accumulating data: {total_pairs_receiving} pairs receiving, "
                        f"max {max_count}/14 candles (need ~{max(0, 14-max_count)} more minutes)"
                    )
                    await asyncio.sleep(10)
                    continue

                for pair in active_pairs:
                    if not self._running:
                        break

                    await self._analyze_pair(pair, params)
                    await asyncio.sleep(1)  # small delay between pairs

                self._analysis_count += 1
                await asyncio.sleep(settings.analysis_interval_seconds)

            except Exception as e:
                logger.error(f"Analysis loop error: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _analyze_pair(self, pair: str, params: dict):
        """Analyze a single pair and execute Claude's decision."""
        # Get market snapshot
        snapshot = self.market_analyzer.get_snapshot(pair)
        if not snapshot:
            return

        # Get context for Claude
        regime = self._current_regime.value
        similar_trades = await self.memory.find_similar(pair, regime)
        active_rules = await self.memory.get_active_rules()
        open_positions = self.position_manager.get_open_positions()
        has_position = pair in self.paper_trader.positions

        # Check circuit breaker
        cb_active, cb_reason = self.circuit_breaker.check(self.paper_trader.total_equity)

        # Get win-rate statistics for this pair + regime
        pattern_stats = await self.memory.get_pattern_stats(
            pair=pair, direction="", market_regime=regime
        )

        # Ask Claude for decision
        decision = await self.claude_trader.make_decision(
            snapshot=snapshot,
            open_positions=open_positions,
            similar_trades=similar_trades,
            active_rules=active_rules,
            current_params=params,
            balance=self.paper_trader.balance,
            pattern_stats=pattern_stats,
        )

        # Validate with risk manager
        is_valid, rejection = self.risk_manager.validate(
            decision=decision,
            balance=self.paper_trader.balance,
            open_positions=len(self.paper_trader.positions),
            has_position_for_pair=has_position,
            circuit_breaker_active=cb_active,
        )

        if not is_valid:
            if decision.action != ActionType.HOLD:
                logger.info(f"[{pair}] Decision rejected: {rejection}")
            return

        # Execute decision
        price = self.candle_store.get_latest_price(pair) or snapshot.price

        if decision.action in (ActionType.ENTER_LONG, ActionType.ENTER_SHORT):
            position = await self.paper_trader.open_position(decision, price)
            if position:
                # Store indicators with the trade
                indicators = snapshot.model_dump(exclude_none=True, exclude={"timestamp"})
                await self.db.update_trade(position.id, {
                    "entry_indicators": str(indicators),
                    "market_regime": regime,
                    "sentiment_score": snapshot.fear_greed,
                })

        elif decision.action == ActionType.EXIT:
            trade = await self.paper_trader.close_position(pair, price, decision.reasoning)
            if trade:
                indicators = snapshot.model_dump(exclude_none=True, exclude={"timestamp"})
                await self.memory.record_trade(trade, indicators, self._current_regime)

        elif decision.action == ActionType.ADJUST:
            position = self.paper_trader.positions.get(pair)
            if position:
                if decision.stop_loss:
                    position.stop_loss = decision.stop_loss
                if decision.take_profit:
                    position.take_profit = decision.take_profit
                logger.info(f"[{pair}] Adjusted: SL={position.stop_loss} TP={position.take_profit}")

    # --- Sentiment Loop ---

    async def _sentiment_loop(self):
        """Fetch sentiment every 15 minutes."""
        while self._running:
            try:
                if self.sentiment.should_fetch():
                    news = await self.sentiment.fetch_news()
                    fg = await self.sentiment.fetch_fear_greed()
                    self.market_analyzer.set_sentiment(news)
                    if fg:
                        self.market_analyzer.set_fear_greed(fg)

                    if self.sentiment.has_breaking_news:
                        logger.warning(f"BREAKING NEWS: {self.sentiment.breaking_headlines}")

            except Exception as e:
                logger.error(f"Sentiment loop error: {e}")

            await asyncio.sleep(settings.sentiment_poll_minutes * 60)

    # --- Deep Analysis Loop ---

    async def _deep_analysis_loop(self):
        """Run deep analysis with Claude Sonnet every 4 hours."""
        await asyncio.sleep(60)  # wait for initial data

        while self._running:
            try:
                recent_trades = await self.db.get_trades(status="closed", limit=30)
                if recent_trades:
                    memories = await self.memory.get_recent_memories(limit=10)
                    params = await self.db.get_current_params()
                    market_summary = self.market_analyzer.get_market_summary(self.pairs)

                    result = await self.claude_trader.deep_analysis(
                        recent_trades=recent_trades,
                        current_params=params,
                        market_summary=market_summary,
                        memories=memories,
                    )

                    if "error" not in result:
                        # Update market regime
                        regime = result.get("market_regime", "unknown")
                        self._current_regime = MarketRegime(regime) if regime in MarketRegime.__members__.values() else MarketRegime.UNKNOWN

                        # Update lessons
                        reviews = result.get("trade_reviews", [])
                        await self.memory.update_lessons(reviews)

                        # Add new rules
                        for rule in result.get("proposed_rules", []):
                            await self.memory.add_rule(
                                rule=rule["rule"],
                                source_trades=[],
                                confidence=rule.get("confidence", 0.5),
                            )

                        logger.info(f"Deep analysis complete. Regime: {regime}, Reviews: {len(reviews)}")

                self._last_deep_analysis = datetime.utcnow()

            except Exception as e:
                logger.error(f"Deep analysis error: {e}", exc_info=True)

            await asyncio.sleep(settings.deep_analysis_interval_hours * 3600)

    # --- Optimization Loop ---

    async def _optimization_loop(self):
        """Run optimizer every 6 hours."""
        await asyncio.sleep(120)  # wait for some trades

        while self._running:
            try:
                if await self.optimizer.should_run():
                    recent_trades = await self.db.get_trades(status="closed", limit=50)
                    if len(recent_trades) >= 5:
                        daily_stats = await self.position_manager.compute_daily_stats()
                        changes = await self.optimizer.run(daily_stats, recent_trades)
                        if changes:
                            params = await self.db.get_current_params()
                            self.risk_manager.update_params(params)

                self._last_optimization = datetime.utcnow()

            except Exception as e:
                logger.error(f"Optimization loop error: {e}", exc_info=True)

            await asyncio.sleep(settings.optimization_interval_hours * 3600)

    # --- Daily Stats Loop ---

    async def _daily_stats_loop(self):
        """Compute and save daily stats every hour."""
        while self._running:
            try:
                await self.position_manager.compute_daily_stats()
            except Exception as e:
                logger.error(f"Daily stats error: {e}")
            await asyncio.sleep(3600)

    # --- Health Check ---

    async def _health_check_loop(self):
        """Monitor system health every 30 seconds."""
        while self._running:
            try:
                if self.stream_manager and self.stream_manager.seconds_since_last_message > 60:
                    logger.warning("No WebSocket data for >60s, streams may be disconnected")

            except Exception as e:
                logger.error(f"Health check error: {e}")

            await asyncio.sleep(30)

    # --- Status ---

    def get_status(self) -> dict:
        """Get comprehensive engine status for the API."""
        cb_active, cb_reason = self.circuit_breaker.check(self.paper_trader.total_equity)

        return {
            "running": self._running,
            "mode": "paper" if settings.paper_trading else "live",
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "uptime_minutes": round(
                (datetime.utcnow() - self._started_at).total_seconds() / 60, 1
            ) if self._started_at else 0,
            "pairs": self.pairs,
            "active_pairs": self.candle_store.pairs_with_data,
            "analysis_cycles": self._analysis_count,
            "market_regime": self._current_regime.value,
            "circuit_breaker": {
                "active": cb_active,
                "reason": cb_reason,
                **self.circuit_breaker.status,
            },
            "ws_connected": self.stream_manager.is_connected if self.stream_manager else False,
            "last_deep_analysis": self._last_deep_analysis.isoformat() if self._last_deep_analysis else None,
            "last_optimization": self._last_optimization.isoformat() if self._last_optimization else None,
            **self.position_manager.get_equity_summary(),
        }
