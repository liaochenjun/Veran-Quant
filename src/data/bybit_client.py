"""Bybit linear-perp kline client.

Bybit lists the tokenized-stock / gold / alt perps the KOL trades that are
absent from Binance (e.g. XAUUSDT, SKHYNIXUSDT, MSTRUSDT). Returns rows in
the same schema as BinanceClient so the shared downloader/storage can
consume either exchange.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 2.0
RETRYABLE_STATUS_CODES = (429, 403)  # Bybit uses 429/403 for rate limits

# project timeframe -> Bybit interval code (minutes)
_INTERVALS = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240"}
_INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}


def _to_utc_millis(dt: datetime) -> int:
    if dt.tzinfo is None or dt.utcoffset() is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1000)


@dataclass(slots=True)
class BybitClient:
    base_url: str = "https://api.bybit.com"
    session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.session = requests.Session()

    def _request_page(self, symbol: str, interval_code: str, params: dict) -> list:
        response: Optional[requests.Response] = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(f"{self.base_url}/v5/market/kline", params=params, timeout=30)
            except requests.exceptions.RequestException as exc:
                if attempt >= MAX_RETRIES - 1:
                    raise
                delay = RETRY_BACKOFF_SECONDS * (attempt + 1)
                logger.warning(
                    "Bybit network error (%s) for %s %s, retrying in %.1fs (attempt %d/%d)",
                    exc.__class__.__name__, symbol, interval_code, delay, attempt + 2, MAX_RETRIES,
                )
                time.sleep(delay)
                continue
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES - 1:
                retry_after = float(response.headers.get("Retry-After", 0))
                delay = max(retry_after, RETRY_BACKOFF_SECONDS * (attempt + 1))
                logger.warning(
                    "Bybit rate limited (HTTP %s) for %s %s, retrying in %.1fs (attempt %d/%d)",
                    response.status_code, symbol, interval_code, delay, attempt + 2, MAX_RETRIES,
                )
                time.sleep(delay)
                continue
            break
        assert response is not None
        response.raise_for_status()
        return response.json().get("result", {}).get("list", [])

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[dict]:
        if interval not in _INTERVALS:
            raise ValueError(f"Unsupported interval {interval!r}; use one of {sorted(_INTERVALS)}")
        symbol = symbol.upper()
        interval_code = _INTERVALS[interval]
        start_ms = _to_utc_millis(start_time) if start_time is not None else None
        end_ms = _to_utc_millis(end_time) if end_time is not None else None

        # Bybit returns the NEWEST ``limit`` rows inside [start, end], so the
        # range must be paginated by moving the END backward until the start
        # of the requested window is covered (or history is exhausted).
        collected: dict[int, dict] = {}
        window_end = end_ms
        while True:
            params: dict[str, object] = {
                "category": "linear",
                "symbol": symbol,
                "interval": interval_code,
                "limit": min(limit, 1000),
            }
            if start_ms is not None:
                params["start"] = start_ms
            if window_end is not None:
                params["end"] = window_end
            items = self._request_page(symbol, interval_code, params)
            if not items:
                break
            interval_ms = _INTERVAL_MINUTES[interval] * 60_000
            for item in items:
                open_ms = int(item[0])
                open_time = datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc)
                collected[open_ms] = {
                    "open_time": open_time,
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                    "close_time": open_time + timedelta(milliseconds=interval_ms - 1),
                    "quote_volume": float(item[6]),
                    "number_of_trades": 0,  # not provided by this endpoint
                    "taker_buy_base_volume": 0.0,
                    "taker_buy_quote_volume": 0.0,
                }
            oldest_open_ms = min(int(item[0]) for item in items)
            if start_ms is not None and oldest_open_ms <= start_ms:
                break  # requested window covered
            if len(items) < min(limit, 1000):
                break  # history exhausted
            window_end = oldest_open_ms - 1

        logger.info("Fetched %s klines for %s %s (bybit)", len(collected), symbol, interval)
        return [collected[key] for key in sorted(collected)]
