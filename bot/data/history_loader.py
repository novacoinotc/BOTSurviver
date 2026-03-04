"""Load historical candles from Binance REST API at startup (via proxy if needed)."""

import asyncio
import logging
from typing import Optional

import httpx

from config.settings import settings
from data.candles import CandleStore, Candle

logger = logging.getLogger(__name__)

BINANCE_FAPI = "https://fapi.binance.com"
BINANCE_FAPI_ALT = "https://fapi1.binance.com"


async def load_historical_candles(
    candle_store: CandleStore,
    pairs: list[str],
    timeframes: list[str] = None,
    limit: int = 499,
) -> int:
    """Fetch historical klines from Binance REST API and populate the candle store.

    Args:
        candle_store: The CandleStore to populate
        pairs: List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"])
        timeframes: Timeframes to fetch (default: ["1m", "5m"])
        limit: Number of candles to fetch per pair/timeframe (max 1500)

    Returns:
        Total number of candles loaded
    """
    if timeframes is None:
        timeframes = ["1m", "5m", "1h"]

    proxy = settings.proxy_url if settings.proxy_url else None
    if not proxy:
        logger.warning("No proxy configured, cannot load historical candles (Binance REST geo-blocked)")
        return 0

    base_url = BINANCE_FAPI
    total_loaded = 0

    async with httpx.AsyncClient(timeout=30, follow_redirects=False, proxy=proxy) as client:
        # Test connectivity first
        try:
            test = await client.get(f"{base_url}/fapi/v1/klines", params={
                "symbol": "BTCUSDT", "interval": "1m", "limit": 1
            })
            if test.status_code in (451, 302, 403):
                base_url = BINANCE_FAPI_ALT
                test2 = await client.get(f"{base_url}/fapi/v1/klines", params={
                    "symbol": "BTCUSDT", "interval": "1m", "limit": 1
                })
                if test2.status_code in (451, 302, 403):
                    logger.warning("Cannot load historical candles: Binance REST API blocked even via proxy")
                    return 0
            logger.info(f"Historical candle loader connected to {base_url} via proxy")
        except Exception as e:
            logger.warning(f"Historical candle loader failed to connect: {e}")
            return 0

        # Fetch candles for each pair and timeframe
        for pair in pairs:
            for tf in timeframes:
                try:
                    count = await _fetch_klines(client, candle_store, base_url, pair, tf, limit)
                    total_loaded += count
                    # Small delay to avoid rate limits
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.warning(f"Failed to load {pair} {tf} candles: {e}")

        logger.info(f"Historical candle loading complete: {total_loaded} candles for {len(pairs)} pairs")

    return total_loaded


async def _fetch_klines(
    client: httpx.AsyncClient,
    candle_store: CandleStore,
    base_url: str,
    pair: str,
    timeframe: str,
    limit: int,
) -> int:
    """Fetch klines for a single pair/timeframe and store them."""
    resp = await client.get(f"{base_url}/fapi/v1/klines", params={
        "symbol": pair,
        "interval": timeframe,
        "limit": limit,
    })
    resp.raise_for_status()
    data = resp.json()

    count = 0
    for k in data:
        # Binance kline format: [open_time, open, high, low, close, volume, close_time,
        #                         quote_volume, trades, taker_buy_volume, taker_buy_quote_volume, ignore]
        # Skip the last candle (it's the current unclosed one)
        is_closed = True  # Historical candles are all closed except possibly the last
        if k == data[-1]:
            is_closed = False  # Last candle might still be open

        candle = Candle(
            timestamp=k[0],
            open=float(k[1]),
            high=float(k[2]),
            low=float(k[3]),
            close=float(k[4]),
            volume=float(k[5]),
            quote_volume=float(k[7]),
            trades=k[8],
            taker_buy_volume=float(k[9]),
            taker_buy_quote_volume=float(k[10]),
            is_closed=is_closed,
        )

        if is_closed:
            candle_store._candles[pair][timeframe].append(candle)
            count += 1
        else:
            candle_store._current[pair][timeframe] = candle

    return count
