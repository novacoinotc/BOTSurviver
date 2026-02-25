"""Generate compact market snapshot for Claude from all data sources."""

import logging
from datetime import datetime
from typing import Optional

from core.models import MarketSnapshot
from data.candles import CandleStore
from data.orderbook import OrderBookStore
from strategy.indicators import calculate_all

logger = logging.getLogger(__name__)


class MarketAnalyzer:
    """Combines candles, orderbook, indicators, sentiment into a snapshot for Claude."""

    def __init__(self, candle_store: CandleStore, orderbook_store: OrderBookStore):
        self.candles = candle_store
        self.orderbook = orderbook_store
        self._funding_rates: dict[str, float] = {}
        self._sentiment: dict = {}
        self._fear_greed: Optional[int] = None

    def set_funding_rate(self, pair: str, rate: float):
        self._funding_rates[pair] = rate

    def set_sentiment(self, sentiment: dict):
        self._sentiment = sentiment

    def set_fear_greed(self, value: int):
        self._fear_greed = value

    def get_snapshot(self, pair: str) -> Optional[MarketSnapshot]:
        """Generate a full market snapshot for a single pair."""
        if not self.candles.has_enough_data(pair, min_candles=14):
            return None

        price = self.candles.get_latest_price(pair)
        if not price:
            return None

        # Get indicator data from 1m candles
        df_1m = self.candles.get_dataframe(pair, "1m")
        indicators = calculate_all(df_1m)

        # Price changes
        change_1m = self.candles.get_price_change(pair, minutes=1)
        change_5m = self.candles.get_price_change(pair, minutes=5)
        change_1h = self.candles.get_price_change(pair, minutes=60)

        # Orderbook
        book_imbalance = self.orderbook.get_imbalance(pair)

        # Volume delta
        volume_delta = self.candles.get_volume_delta(pair, minutes=5)

        snapshot = MarketSnapshot(
            pair=pair,
            price=price,
            change_1m=round(change_1m * 100, 4) if change_1m else 0.0,
            change_5m=round(change_5m * 100, 4) if change_5m else 0.0,
            change_1h=round(change_1h * 100, 4) if change_1h else 0.0,
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
            sentiment=self._sentiment if self._sentiment else None,
            fear_greed=self._fear_greed,
            timestamp=datetime.utcnow(),
        )

        return snapshot

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

        return {
            "total_pairs": total,
            "bullish_pairs": bullish,
            "bearish_pairs": bearish,
            "neutral_pairs": total - bullish - bearish,
            "fear_greed": self._fear_greed,
            "timestamp": datetime.utcnow().isoformat(),
        }
