from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from src.data.storage import DuckDBStorage


def _as_aware_utc(dt: datetime) -> datetime:
    """Normalize to tz-aware UTC, interpreting naive datetimes as UTC."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(slots=True)
class MarketState:
    symbol: str
    as_of_timestamp: datetime
    frames: dict[str, pd.DataFrame]


@dataclass(slots=True)
class PointInTimeMarketState:
    storage: DuckDBStorage
    timeframes: tuple[str, ...] = ("1m", "5m", "15m", "1h", "4h")
    lookback_candles: int = 300

    def get_market_state(self, symbol: str, as_of_timestamp: datetime) -> MarketState:
        as_of = _as_aware_utc(as_of_timestamp)
        frames: dict[str, pd.DataFrame] = {}
        for timeframe in self.timeframes:
            frame = self.storage.read_klines(
                symbol=symbol,
                timeframe=timeframe,
                end_before=as_of,
                limit=self.lookback_candles,
            )
            if not frame.empty:
                close_times = frame["close_time"]
                if close_times.dt.tz is None:
                    # Naive parquet data is assumed to be UTC wall time.
                    close_times = close_times.dt.tz_localize("UTC")
                if (close_times >= as_of).any():
                    raise ValueError("Point-in-time violation: candle close_time is not strictly before as_of_timestamp")
            frames[timeframe] = frame

        return MarketState(symbol=symbol, as_of_timestamp=as_of, frames=frames)
