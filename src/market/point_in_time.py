from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.data.storage import DuckDBStorage


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
        frames: dict[str, pd.DataFrame] = {}
        for timeframe in self.timeframes:
            frame = self.storage.read_klines(
                symbol=symbol,
                timeframe=timeframe,
                end_before=as_of_timestamp,
                limit=self.lookback_candles,
            )
            if not frame.empty:
                if (frame["close_time"] >= as_of_timestamp).any():
                    raise ValueError("Point-in-time violation: candle close_time is not strictly before as_of_timestamp")
            frames[timeframe] = frame

        return MarketState(symbol=symbol, as_of_timestamp=as_of_timestamp, frames=frames)
