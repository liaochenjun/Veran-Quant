from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BinanceClient:
    base_url: str = "https://api.binance.com"

    def __post_init__(self) -> None:
        self.session = requests.Session()
        api_key = os.getenv("BINANCE_API_KEY")
        if api_key:
            self.session.headers.update({"X-MBX-APIKEY": api_key})

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[dict]:
        params: dict[str, object] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
        }
        if start_time is not None:
            params["startTime"] = int(start_time.replace(tzinfo=timezone.utc).timestamp() * 1000)
        if end_time is not None:
            params["endTime"] = int(end_time.replace(tzinfo=timezone.utc).timestamp() * 1000)

        response = self.session.get(f"{self.base_url}/api/v3/klines", params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        logger.info("Fetched %s klines for %s %s", len(payload), symbol, interval)

        rows: list[dict] = []
        for item in payload:
            rows.append(
                {
                    "open_time": datetime.fromtimestamp(item[0] / 1000, tz=timezone.utc),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                    "close_time": datetime.fromtimestamp(item[6] / 1000, tz=timezone.utc),
                    "quote_volume": float(item[7]),
                    "number_of_trades": int(item[8]),
                    "taker_buy_base_volume": float(item[9]),
                    "taker_buy_quote_volume": float(item[10]),
                }
            )
        return rows
