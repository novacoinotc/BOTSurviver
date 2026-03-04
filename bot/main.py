"""Entry point for the scalping bot."""

import asyncio
import logging
import signal
import sys

import uvicorn

from config.settings import settings
from core.engine import TradingEngine
from api.server import app, set_engine

# Configure logging — only StreamHandler; nohup redirects stdout to bot.log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


async def main():
    engine = TradingEngine()
    set_engine(engine)

    # Handle shutdown signals
    loop = asyncio.get_event_loop()

    def shutdown(sig):
        logger.info(f"Received signal {sig}, shutting down...")
        asyncio.create_task(engine.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: shutdown(s))

    # Start API server in background
    config = uvicorn.Config(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    logger.info(f"Starting API server on {settings.api_host}:{settings.api_port}")

    # Run both engine and API server
    await asyncio.gather(
        engine.start(),
        server.serve(),
    )


if __name__ == "__main__":
    try:
        import uvloop
        uvloop.install()
        logger.info("Using uvloop")
    except ImportError:
        logger.info("uvloop not available, using default event loop")

    asyncio.run(main())
