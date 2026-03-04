"""Download historical kline data from Binance Futures API.

Handles geo-restriction via proxy. Caches results to CSV to avoid re-downloading.
Paginates through the API (max 1500 candles per request).
"""

import time
from pathlib import Path

import httpx
import pandas as pd

CACHE_DIR = Path(__file__).parent / "cache"


def load_proxy_url() -> str:
    """Load proxy URL from bot/.env file."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("PROXY_URL=") or line.startswith("proxy_url="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def download_klines(
    symbol: str,
    interval: str = "1m",
    days: int = 14,
    proxy_url: str = None,
) -> pd.DataFrame:
    """Download historical klines from Binance Futures.

    Paginates through the API (max 1500 per request).
    Caches to CSV files to avoid re-downloading.

    Args:
        symbol: e.g. "BTCUSDT"
        interval: "1m", "5m", "15m", "1h", "4h"
        days: number of days of history
        proxy_url: HTTP proxy URL for geo-restricted access

    Returns:
        DataFrame with columns:
        [timestamp, open, high, low, close, volume, quote_volume,
         taker_buy_quote_volume, num_trades]
    """
    # Check cache first
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{symbol}_{interval}_{days}d.csv"
    if cache_file.exists():
        # Use cache if less than 1 hour old
        if time.time() - cache_file.stat().st_mtime < 3600:
            df = pd.read_csv(cache_file)
            print(f"  Loaded {len(df)} candles from cache for {symbol}")
            return df

    # Calculate pagination
    # 1m: 1440/day, 5m: 288/day, 15m: 96/day, 1h: 24/day, 4h: 6/day
    intervals_per_day = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6}
    total_candles = days * intervals_per_day.get(interval, 1440)

    base_url = "https://fapi.binance.com"
    endpoint = "/fapi/v1/klines"

    all_candles = []
    # Start from (now - days) in milliseconds
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    client_kwargs = {"timeout": 30.0}
    if proxy_url:
        client_kwargs["proxy"] = proxy_url

    with httpx.Client(**client_kwargs) as client:
        current_start = start_time
        while current_start < end_time:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "limit": 1500,
            }
            try:
                resp = client.get(f"{base_url}{endpoint}", params=params)
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break
                all_candles.extend(data)
                # Move start to after last candle
                current_start = data[-1][0] + 1
                if len(data) < 1500:
                    break
            except Exception as e:
                print(f"  Error fetching {symbol}: {e}")
                break

    if not all_candles:
        return pd.DataFrame()

    # Parse into DataFrame
    df = pd.DataFrame(
        all_candles,
        columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "num_trades",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore",
        ],
    )

    # Convert types
    for col in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote_volume"]:
        df[col] = df[col].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Drop unnecessary columns
    df = df[
        ["timestamp", "open", "high", "low", "close", "volume",
         "quote_volume", "taker_buy_quote_volume", "num_trades"]
    ]

    # Remove duplicates
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Save cache
    df.to_csv(cache_file, index=False)
    print(f"  Downloaded {len(df)} candles for {symbol} ({interval}, {days}d)")

    return df
