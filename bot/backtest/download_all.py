#!/usr/bin/env python3
"""
Pre-download all 12-month data sequentially with rate limit delays.
Run this FIRST, then run_12month_production.py loads from cache.
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.data_loader import download_klines, load_proxy_url

ALL_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "SUIUSDT", "ARBUSDT", "OPUSDT", "APTUSDT",
    "NEARUSDT", "LTCUSDT", "ATOMUSDT", "FILUSDT", "INJUSDT",
]
# MATICUSDT removed (migrated to POLUSDT)

DAYS = 365
INTERVAL = "5m"
DELAY_BETWEEN_PAIRS = 3  # seconds between pairs to avoid 429

def main():
    proxy_url = load_proxy_url()
    print(f"Downloading {len(ALL_PAIRS)} pairs, {DAYS} days of {INTERVAL} candles")
    print(f"Delay between pairs: {DELAY_BETWEEN_PAIRS}s\n")

    success = []
    failed = []

    for i, pair in enumerate(ALL_PAIRS, 1):
        print(f"[{i}/{len(ALL_PAIRS)}] {pair}...", end=" ", flush=True)
        try:
            df = download_klines(pair, interval=INTERVAL, days=DAYS, proxy_url=proxy_url)
            if df.empty or len(df) < 1000:
                print(f"FAIL ({len(df) if not df.empty else 0} candles)")
                failed.append(pair)
            else:
                print(f"OK - {len(df)} candles")
                success.append(pair)
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(pair)

        # Rate limit delay (skip for last pair)
        if i < len(ALL_PAIRS):
            time.sleep(DELAY_BETWEEN_PAIRS)

    print(f"\nSuccess: {len(success)} pairs: {success}")
    print(f"Failed: {len(failed)} pairs: {failed}")

    if failed:
        print(f"\nRetrying failed pairs with longer delay...")
        time.sleep(10)
        retry_success = []
        for pair in failed:
            print(f"  Retry {pair}...", end=" ", flush=True)
            try:
                df = download_klines(pair, interval=INTERVAL, days=DAYS, proxy_url=proxy_url)
                if df.empty or len(df) < 1000:
                    print(f"FAIL again")
                else:
                    print(f"OK - {len(df)} candles")
                    retry_success.append(pair)
            except Exception as e:
                print(f"ERROR: {e}")
            time.sleep(5)

        final_failed = [p for p in failed if p not in retry_success]
        print(f"\nFinal success: {len(success) + len(retry_success)} pairs")
        if final_failed:
            print(f"Still failed: {final_failed}")

if __name__ == "__main__":
    main()
