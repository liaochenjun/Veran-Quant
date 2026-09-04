from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Make the repo root importable when run as `python scripts/download_klines.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.binance_client import BinanceClient
from src.data.downloader import BinanceDownloader
from src.data.storage import DuckDBStorage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Binance historical klines")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True, choices=["1m", "5m", "15m", "1h", "4h"])
    parser.add_argument("--start", required=True, help="ISO datetime")
    parser.add_argument("--end", required=True, help="ISO datetime")
    parser.add_argument(
        "--request-interval", type=float, default=0.0,
        help="sleep seconds between requests to stay under rate limits (default: 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    storage = DuckDBStorage(root_dir=Path("data/raw"), database_path=Path("data/database/market.duckdb"))
    downloader = BinanceDownloader(
        client=BinanceClient(), storage=storage, request_interval=args.request_interval
    )
    downloader.download_historical_klines(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_time=datetime.fromisoformat(args.start),
        end_time=datetime.fromisoformat(args.end),
    )


if __name__ == "__main__":
    main()
