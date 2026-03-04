"""WebSocket multi-stream manager for Binance Futures data."""

import asyncio
import json
import logging
import time
from typing import Callable, Optional

import aiohttp

logger = logging.getLogger(__name__)

BINANCE_WS_BASE = "wss://fstream.binance.com"
MAX_STREAMS_PER_CONNECTION = 200  # Binance limit


class StreamManager:
    """Manages WebSocket connections to Binance Futures for multiple pairs."""

    def __init__(self, pairs: list[str], on_kline: Callable, on_book_ticker: Callable, on_agg_trade: Callable, on_mark_price: Callable = None):
        self.pairs = [p.lower() for p in pairs]
        self.on_kline = on_kline
        self.on_book_ticker = on_book_ticker
        self.on_agg_trade = on_agg_trade
        self.on_mark_price = on_mark_price
        self._sessions: list[aiohttp.ClientSession] = []
        self._ws_connections: list[aiohttp.ClientWebSocketResponse] = []
        self._running = False
        self._reconnect_delay = 1
        self._last_message_time = 0.0

    def _build_streams(self) -> list[str]:
        """Build stream names for all pairs."""
        streams = []
        for pair in self.pairs:
            streams.append(f"{pair}@kline_1m")
            streams.append(f"{pair}@kline_5m")
            streams.append(f"{pair}@kline_1h")
            streams.append(f"{pair}@bookTicker")
            streams.append(f"{pair}@aggTrade")
            streams.append(f"{pair}@markPrice@1s")  # funding rate + mark price
        return streams

    async def start(self):
        """Start all WebSocket streams."""
        self._running = True
        streams = self._build_streams()

        # Split into batches if needed
        for i in range(0, len(streams), MAX_STREAMS_PER_CONNECTION):
            batch = streams[i:i + MAX_STREAMS_PER_CONNECTION]
            asyncio.create_task(self._connect_streams(batch))

        logger.info(f"Started streams for {len(self.pairs)} pairs ({len(streams)} streams)")

    async def _connect_streams(self, streams: list[str]):
        """Connect to a batch of streams with auto-reconnect."""
        stream_path = "/".join(streams)
        url = f"{BINANCE_WS_BASE}/stream?streams={stream_path}"

        while self._running:
            try:
                session = aiohttp.ClientSession()
                self._sessions.append(session)
                ws = await session.ws_connect(url, heartbeat=20, receive_timeout=30)
                self._ws_connections.append(ws)
                self._reconnect_delay = 1
                logger.info(f"WebSocket connected ({len(streams)} streams)")

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        self._last_message_time = time.time()
                        await self._handle_message(json.loads(msg.data))
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {ws.exception()}")
                        break
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                        break

            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
            finally:
                if ws and not ws.closed:
                    await ws.close()
                await session.close()

            if self._running:
                logger.info(f"Reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 60)

    async def _handle_message(self, data: dict):
        """Route message to the appropriate handler."""
        if "stream" not in data or "data" not in data:
            return

        stream = data["stream"]
        payload = data["data"]

        try:
            if "@kline_" in stream:
                await self.on_kline(payload)
            elif "@bookTicker" in stream:
                await self.on_book_ticker(payload)
            elif "@aggTrade" in stream:
                await self.on_agg_trade(payload)
            elif "@markPrice" in stream:
                if self.on_mark_price:
                    await self.on_mark_price(payload)
        except Exception as e:
            logger.error(f"Error handling {stream}: {e}")

    async def stop(self):
        """Close all WebSocket connections."""
        self._running = False
        for ws in self._ws_connections:
            if not ws.closed:
                await ws.close()
        for session in self._sessions:
            await session.close()
        self._ws_connections.clear()
        self._sessions.clear()
        logger.info("All WebSocket streams closed")

    @property
    def is_connected(self) -> bool:
        return any(not ws.closed for ws in self._ws_connections)

    @property
    def seconds_since_last_message(self) -> float:
        if self._last_message_time == 0:
            return float("inf")
        return time.time() - self._last_message_time
