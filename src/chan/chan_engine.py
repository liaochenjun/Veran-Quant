from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from src.market.point_in_time import MarketState, PointInTimeMarketState


class ChanEngine(ABC):
    @abstractmethod
    def get_state(
        self,
        symbol: str,
        timeframe: str,
        as_of_timestamp: datetime,
        market_state: MarketState | None = None,
    ) -> dict:
        raise NotImplementedError


@dataclass(slots=True)
class SimpleChanEngine(ChanEngine):
    point_in_time: PointInTimeMarketState

    def get_state(
        self,
        symbol: str,
        timeframe: str,
        as_of_timestamp: datetime,
        market_state: MarketState | None = None,
    ) -> dict:
        # Reuse a precomputed point-in-time state when available to avoid
        # fetching the full multi-timeframe snapshot twice per trade.
        if market_state is None:
            market_state = self.point_in_time.get_market_state(symbol=symbol, as_of_timestamp=as_of_timestamp)
        candles = market_state.frames.get(timeframe)
        if candles is None or candles.empty:
            return {"trend": "unknown", "swing_high": None, "swing_low": None}

        first_close = float(candles.iloc[0]["close"])
        last_close = float(candles.iloc[-1]["close"])
        trend = "up" if last_close > first_close else "down" if last_close < first_close else "flat"

        return {
            "trend": trend,
            "swing_high": float(candles["high"].max()),
            "swing_low": float(candles["low"].min()),
            "as_of_timestamp": as_of_timestamp.isoformat(),
        }
