from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src.data.binance_client import BinanceClient
from src.data.storage import DuckDBStorage

logger = logging.getLogger(__name__)


def _as_aware_utc(dt: datetime) -> datetime:
    """Normalize to tz-aware UTC, interpreting naive datetimes as UTC."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(slots=True)
class BinanceDownloader:
    client: BinanceClient
    storage: DuckDBStorage
    # Optional pacing between requests to stay comfortably under rate limits.
    request_interval: float = 0.0

    def download_historical_klines(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        # The client returns tz-aware UTC open_times; keep pagination cursor
        # in the same space or `current < end_time` raises on naive input.
        start_time = _as_aware_utc(start_time)
        end_time = _as_aware_utc(end_time)
        current = start_time
        total = 0

        while current < end_time:
            rows = self.client.get_klines(
                symbol=symbol,
                interval=timeframe,
                start_time=current,
                end_time=end_time,
                limit=1000,
            )
            if not rows:
                break

            self.storage.write_klines(symbol=symbol, timeframe=timeframe, rows=rows)
            total += len(rows)
            latest_open_time = rows[-1]["open_time"]
            current = latest_open_time + timedelta(milliseconds=1)
            if self.request_interval > 0:
                time.sleep(self.request_interval)

        logger.info("Downloaded %s klines for %s %s", total, symbol, timeframe)
        return total
