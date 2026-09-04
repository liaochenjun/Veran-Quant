from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.data.binance_client import BinanceClient
from src.data.storage import DuckDBStorage

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BinanceDownloader:
    client: BinanceClient
    storage: DuckDBStorage

    def download_historical_klines(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
    ) -> int:
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

        logger.info("Downloaded %s klines for %s %s", total, symbol, timeframe)
        return total
