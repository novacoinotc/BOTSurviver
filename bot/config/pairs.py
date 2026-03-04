"""Auto-detect top 20 USDT futures pairs by 24h volume."""

import logging
from binance.client import Client
from config.settings import settings

logger = logging.getLogger(__name__)

# Fallback list if API call fails
DEFAULT_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "SUIUSDT", "ARBUSDT", "OPUSDT", "APTUSDT",
    "NEARUSDT", "LTCUSDT", "ATOMUSDT", "FILUSDT", "INJUSDT",
]
# MATICUSDT removed — migrated to POLUSDT on Binance


async def get_top_pairs(count: int = 19) -> list[str]:
    """Fetch top futures pairs by 24h volume from Binance."""
    try:
        client = Client(settings.binance_api_key, settings.binance_api_secret)
        tickers = client.futures_ticker()

        usdt_pairs = [
            t for t in tickers
            if t["symbol"].endswith("USDT")
            and not t["symbol"].endswith("_PERP")
            and float(t["quoteVolume"]) > 0
        ]

        usdt_pairs.sort(key=lambda t: float(t["quoteVolume"]), reverse=True)
        top = [t["symbol"] for t in usdt_pairs[:count]]

        logger.info(f"Top {count} pairs by volume: {top}")
        client.close_connection()
        return top
    except Exception as e:
        logger.warning(f"Failed to fetch pairs from Binance: {e}. Using defaults.")
        return DEFAULT_PAIRS[:count]
