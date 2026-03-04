"""Generate compact market snapshot for Claude from all data sources."""

import logging
from datetime import datetime
from typing import Optional

from core.models import MarketSnapshot, MarketRegime
from data.candles import CandleStore
from data.orderbook import OrderBookStore
from strategy.indicators import calculate_all, calculate_5m

logger = logging.getLogger(__name__)


class MarketAnalyzer:
    """Combines candles, orderbook, indicators, sentiment into a snapshot for Claude."""

    def __init__(self, candle_store: CandleStore, orderbook_store: OrderBookStore):
        self.candles = candle_store
        self.orderbook = orderbook_store
        self._funding_rates: dict[str, float] = {}
        self._open_interest: dict[str, float] = {}
        self._oi_change: dict[str, float] = {}
        self._long_short_ratios: dict[str, float] = {}
        self._sentiment: dict = {}
        self._fear_greed: Optional[int] = None
        self._breaking_news: Optional[str] = None

    def set_funding_rate(self, pair: str, rate: float):
        self._funding_rates[pair] = rate

    def set_funding_rates(self, rates: dict[str, float]):
        self._funding_rates.update(rates)

    def set_open_interest(self, pair: str, oi: float, change_pct: Optional[float] = None):
        self._open_interest[pair] = oi
        if change_pct is not None:
            self._oi_change[pair] = change_pct

    def set_long_short_ratio(self, pair: str, ratio: float):
        self._long_short_ratios[pair] = ratio

    def set_sentiment(self, sentiment: dict):
        self._sentiment = sentiment

    def set_fear_greed(self, value: int):
        self._fear_greed = value

    def set_breaking_news(self, news: Optional[str]):
        self._breaking_news = news

    def get_snapshot(self, pair: str) -> Optional[MarketSnapshot]:
        """Generate a full market snapshot for a single pair.

        Primary indicators come from 1H candles (trading timeframe).
        Real-time price comes from 1m data. 5m provides secondary context.
        """
        # Need 1H data for signal detection (50+ candles for EMA50)
        if not self.candles.has_enough_data(pair, min_candles=50, timeframe="1h"):
            # Fallback: check if we have 1m data at least (for price tracking)
            if not self.candles.has_enough_data(pair, min_candles=14):
                return None
            # Have 1m but not enough 1h — create minimal snapshot for position tracking
            price = self.candles.get_latest_price(pair)
            if not price:
                return None
            return MarketSnapshot(
                pair=pair,
                price=price,
                fear_greed=self._fear_greed,
                timestamp=datetime.utcnow(),
            )

        price = self.candles.get_latest_price(pair)
        if not price:
            return None

        # PRIMARY: Get indicator data from 1H candles (trading timeframe)
        df_1h = self.candles.get_dataframe(pair, "1h")
        indicators = calculate_all(df_1h)

        # SECONDARY: Get 5m multi-timeframe indicators
        df_5m = self.candles.get_dataframe(pair, "5m")
        indicators_5m = calculate_5m(df_5m)

        # Price changes from 1m (real-time)
        change_1m = self.candles.get_price_change(pair, minutes=1)
        change_5m = self.candles.get_price_change(pair, minutes=5)
        change_1h = self.candles.get_price_change(pair, minutes=60)

        # Orderbook
        book_imbalance = self.orderbook.get_imbalance(pair)
        spread_pct = self.orderbook.get_spread(pair)

        # Volume delta
        volume_delta = self.candles.get_volume_delta(pair, minutes=5)

        snapshot = MarketSnapshot(
            pair=pair,
            price=price,
            change_1m=round(change_1m * 100, 4) if change_1m else 0.0,
            change_5m=round(change_5m * 100, 4) if change_5m else 0.0,
            change_1h=round(change_1h * 100, 4) if change_1h else 0.0,
            # All main indicators from 1H timeframe
            rsi_7=indicators.get("rsi_7"),
            rsi_14=indicators.get("rsi_14"),
            ema_9=indicators.get("ema_9"),
            ema_21=indicators.get("ema_21"),
            ema_50=indicators.get("ema_50"),
            bb_upper=indicators.get("bb_upper"),
            bb_lower=indicators.get("bb_lower"),
            bb_pct=indicators.get("bb_pct"),
            macd_hist=indicators.get("macd_hist"),
            macd_signal=indicators.get("macd_signal"),
            vwap=indicators.get("vwap"),
            price_vs_vwap=indicators.get("price_vs_vwap"),
            atr_14=indicators.get("atr_14"),
            atr_pct=indicators.get("atr_pct"),
            volume_delta_5m=indicators.get("volume_delta_5m") or volume_delta,
            book_imbalance=round(book_imbalance, 2) if book_imbalance else None,
            funding_rate=self._funding_rates.get(pair),
            stoch_rsi_k=indicators.get("stoch_rsi_k"),
            stoch_rsi_d=indicators.get("stoch_rsi_d"),
            adx=indicators.get("adx"),
            plus_di=indicators.get("plus_di"),
            minus_di=indicators.get("minus_di"),
            mfi=indicators.get("mfi"),
            bb_width=indicators.get("bb_width"),
            bb_squeeze=indicators.get("bb_squeeze"),
            # Advanced features
            spread_pct=round(spread_pct * 100, 6) if spread_pct else None,
            ema_alignment=indicators.get("ema_alignment"),
            rsi_divergence=indicators.get("rsi_divergence"),
            consecutive_direction=indicators.get("consecutive_direction"),
            price_position_range=indicators.get("price_position_range"),
            volume_buy_ratio=indicators.get("volume_buy_ratio"),
            volume_ratio=indicators.get("volume_ratio"),
            price_ema_ratio=indicators.get("price_ema_ratio"),
            dist_to_support_pct=indicators.get("dist_to_support_pct"),
            dist_to_resistance_pct=indicators.get("dist_to_resistance_pct"),
            momentum_score=indicators.get("momentum_score"),
            trend_strength_score=indicators.get("trend_strength_score"),
            # Multi-timeframe 5m
            rsi_14_5m=indicators_5m.get("rsi_14_5m"),
            ema_trend_5m=indicators_5m.get("ema_trend_5m"),
            adx_5m=indicators_5m.get("adx_5m"),
            macd_signal_5m=indicators_5m.get("macd_signal_5m"),
            # Futures data
            open_interest=self._open_interest.get(pair),
            open_interest_change_pct=self._oi_change.get(pair),
            long_short_ratio=self._long_short_ratios.get(pair),
            # News / sentiment
            breaking_news=self._breaking_news,
            sentiment=self._sentiment if self._sentiment else None,
            fear_greed=self._fear_greed,
            timestamp=datetime.utcnow(),
        )

        return snapshot

    def detect_regime_fast(self, pair: str) -> MarketRegime:
        """Fast regime detection from 1H indicators (no Claude call needed)."""
        df = self.candles.get_dataframe(pair, "1h")
        if df.empty or len(df) < 50:
            # Fallback to 1m if not enough 1h data
            df = self.candles.get_dataframe(pair, "1m")
            if df.empty or len(df) < 21:
                return MarketRegime.UNKNOWN

        indicators = calculate_all(df)
        adx = indicators.get("adx", 0) or 0
        ema_align = indicators.get("ema_alignment", 0) or 0
        bb_width = indicators.get("bb_width", 0) or 0
        rsi = indicators.get("rsi_14", 50) or 50

        # ADX > 25 = trending
        if adx > 25:
            if ema_align > 0:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN

        # BB width > 0.04 = volatile
        if bb_width > 0.04:
            return MarketRegime.VOLATILE

        # Otherwise ranging
        return MarketRegime.RANGING

    def get_market_regime_consensus(self, pairs: list[str]) -> MarketRegime:
        """Get overall market regime from consensus of top pairs."""
        regimes = []
        for pair in pairs[:10]:  # check top 10 pairs
            regime = self.detect_regime_fast(pair)
            if regime != MarketRegime.UNKNOWN:
                regimes.append(regime)

        if not regimes:
            return MarketRegime.UNKNOWN

        # Count votes
        from collections import Counter
        counts = Counter(regimes)
        return counts.most_common(1)[0][0]

    def get_all_snapshots(self, pairs: list[str]) -> dict[str, MarketSnapshot]:
        """Generate snapshots for all active pairs."""
        snapshots = {}
        for pair in pairs:
            snap = self.get_snapshot(pair)
            if snap:
                snapshots[pair] = snap
        return snapshots

    def get_market_summary(self, pairs: list[str]) -> dict:
        """Get a high-level market summary across all pairs."""
        bullish = 0
        bearish = 0
        total = 0

        for pair in pairs:
            df = self.candles.get_dataframe(pair, "1m")
            if df.empty:
                continue
            total += 1
            change = self.candles.get_price_change(pair, minutes=5)
            if change and change > 0:
                bullish += 1
            elif change and change < 0:
                bearish += 1

        regime = self.get_market_regime_consensus(pairs)

        return {
            "total_pairs": total,
            "bullish_pairs": bullish,
            "bearish_pairs": bearish,
            "neutral_pairs": total - bullish - bearish,
            "market_regime": regime.value,
            "fear_greed": self._fear_greed,
            "timestamp": datetime.utcnow().isoformat(),
        }
