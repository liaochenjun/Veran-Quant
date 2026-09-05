"""Hyperliquid kline client.

HYPEUSDT exists on Hyperliquid but not on Binance/Bybit. Caveat: HL serves
1m/5m/15m candles only for recent windows; 1h/4h/1d history is deep. Rows
are returned in the same schema as BinanceClient so the shared downloader
and storage consume this exchange identically.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 2.0
RETRYABLE_STATUS_CODES = (429, 503)

_INTERVALS = ("1m", "5m", "15m", "1h", "4h")


def _to_utc_millis(dt: datetime) -> int:
    if dt.tzinfo is None or dt.utcoffset() is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1000)


@dataclass(slots=True)
class HyperliquidClient:
    base_url: str = "https://api.hyperliquid.xyz"
    session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.session = requests.Session()

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
        # HL coins carry no -USDT suffix (HYPEUSDT -> HYPE).
        coin = symbol.upper()[:-4] if symbol.upper().endswith("USDT") else symbol.upper()
        req: dict[str, object] = {"coin": coin, "interval": interval}
        if start_time is not None:
            req["startTime"] = _to_utc_millis(start_time)
        if end_time is not None:
            req["endTime"] = _to_utc_millis(end_time)

        payload: Optional[list] = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.post(
                    f"{self.base_url}/info", json={"type": "candleSnapshot", "req": req}, timeout=30
                )
            except requests.exceptions.RequestException as exc:
                if attempt >= MAX_RETRIES - 1:
                    raise
                delay = RETRY_BACKOFF_SECONDS * (attempt + 1)
                logger.warning(
                    "Hyperliquid network error (%s) for %s %s, retrying in %.1fs (attempt %d/%d)",
                    exc.__class__.__name__, coin, interval, delay, attempt + 2, MAX_RETRIES,
                )
                time.sleep(delay)
                continue
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES - 1:
                delay = RETRY_BACKOFF_SECONDS * (attempt + 1)
                logger.warning(
                    "Hyperliquid rate limited (HTTP %s) for %s %s, retrying in %.1fs (attempt %d/%d)",
                    response.status_code, coin, interval, delay, attempt + 2, MAX_RETRIES,
                )
                time.sleep(delay)
                continue
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                break
        assert payload is not None
        logger.info("Fetched %s klines for %s %s (hyperliquid)", len(payload), coin, interval)

        rows: list[dict] = []
        for candle in payload:
            # HL candle: {t: open_ms, T: close_ms, o, c, h, l, v, n}
            open_time = datetime.fromtimestamp(int(candle["t"]) / 1000, tz=timezone.utc)
            close_time = datetime.fromtimestamp(int(candle["T"]) / 1000, tz=timezone.utc)
            rows.append(
                {
                    "open_time": open_time,
                    "open": float(candle["o"]),
                    "high": float(candle["h"]),
                    "low": float(candle["l"]),
                    "close": float(candle["c"]),
                    "volume": float(candle["v"]),
                    "close_time": close_time,
                    "quote_volume": 0.0,  # not provided by this endpoint
                    "number_of_trades": int(candle["n"]),
                    "taker_buy_base_volume": 0.0,
                    "taker_buy_quote_volume": 0.0,
                }
            )
        return rows
