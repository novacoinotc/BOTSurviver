"""In-memory OHLCV candle storage (500 candles x 20 pairs x 2 timeframes)."""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAX_CANDLES = 500


@dataclass
class Candle:
    timestamp: int  # ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int
    taker_buy_volume: float
    taker_buy_quote_volume: float
    is_closed: bool


class CandleStore:
    """Stores OHLCV data in memory for all pairs and timeframes."""

    def __init__(self, max_candles: int = MAX_CANDLES):
        self.max_candles = max_candles
        # {pair: {timeframe: deque[Candle]}}
        self._candles: dict[str, dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=max_candles))
        )
        self._current: dict[str, dict[str, Optional[Candle]]] = defaultdict(
            lambda: defaultdict(lambda: None)
        )

    def update_from_kline(self, data: dict):
        """Update candle from Binance kline WebSocket message."""
        kline = data["k"]
        pair = data["s"]  # e.g., BTCUSDT
        tf = kline["i"]   # e.g., 1m, 5m

        candle = Candle(
            timestamp=kline["t"],
            open=float(kline["o"]),
            high=float(kline["h"]),
            low=float(kline["l"]),
            close=float(kline["c"]),
            volume=float(kline["v"]),
            quote_volume=float(kline["q"]),
            trades=kline["n"],
            taker_buy_volume=float(kline["V"]),
            taker_buy_quote_volume=float(kline["Q"]),
            is_closed=kline["x"],
        )

        if candle.is_closed:
            self._candles[pair][tf].append(candle)
            self._current[pair][tf] = None
        else:
            self._current[pair][tf] = candle

    def get_candles(self, pair: str, timeframe: str = "1m", count: Optional[int] = None) -> list[Candle]:
        """Get closed candles for a pair + timeframe."""
        candles = list(self._candles[pair][timeframe])
        if count:
            candles = candles[-count:]
        return candles

    def get_current_candle(self, pair: str, timeframe: str = "1m") -> Optional[Candle]:
        """Get the current (not yet closed) candle."""
        return self._current[pair][timeframe]

    def get_dataframe(self, pair: str, timeframe: str = "1m", count: Optional[int] = None) -> pd.DataFrame:
        """Get candles as a pandas DataFrame for indicator calculation."""
        candles = self.get_candles(pair, timeframe, count)
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame([
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "quote_volume": c.quote_volume,
                "trades": c.trades,
                "taker_buy_volume": c.taker_buy_volume,
                "taker_buy_quote_volume": c.taker_buy_quote_volume,
            }
            for c in candles
        ])
        return df

    def get_latest_price(self, pair: str) -> Optional[float]:
        """Get the latest price for a pair."""
        current = self._current[pair].get("1m")
        if current:
            return current.close
        candles_1m = self._candles[pair].get("1m")
        if candles_1m:
            return candles_1m[-1].close
        return None

    def get_price_change(self, pair: str, minutes: int = 1) -> Optional[float]:
        """Get price change as a percentage over the last N minutes."""
        candles = self.get_candles(pair, "1m", count=minutes + 1)
        if len(candles) < 2:
            return None
        old_price = candles[0].close
        new_price = candles[-1].close
        if old_price == 0:
            return None
        return (new_price - old_price) / old_price

    def get_volume_delta(self, pair: str, minutes: int = 5) -> Optional[float]:
        """Calculate volume delta (buy volume - sell volume) over N minutes."""
        candles = self.get_candles(pair, "1m", count=minutes)
        if not candles:
            return None
        buy_vol = sum(c.taker_buy_quote_volume for c in candles)
        total_vol = sum(c.quote_volume for c in candles)
        sell_vol = total_vol - buy_vol
        return buy_vol - sell_vol

    def has_enough_data(self, pair: str, min_candles: int = 14, timeframe: str = "1m") -> bool:
        """Check if we have enough data to calculate indicators."""
        return len(self._candles[pair][timeframe]) >= min_candles

    @property
    def pairs_with_data(self) -> list[str]:
        return [p for p in self._candles if self.has_enough_data(p)]

    @property
    def pairs_with_1h_data(self) -> list[str]:
        """Pairs with enough 1H candles for indicator calculation (need 50+)."""
        return [p for p in self._candles if self.has_enough_data(p, min_candles=50, timeframe="1h")]

    def get_candle_counts(self) -> dict[str, int]:
        """Get count of closed 1m candles per pair (for debugging)."""
        return {p: len(self._candles[p]["1m"]) for p in self._candles}
