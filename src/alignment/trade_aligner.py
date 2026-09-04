from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src.market.point_in_time import MarketState, PointInTimeMarketState


@dataclass(slots=True)
class KOLTrade:
    kol: str
    symbol: str
    timestamp: datetime
    side: str
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    mfe: Optional[float] = None
    mae: Optional[float] = None
    holding_time: Optional[float] = None
    position_size: Optional[float] = None
    confidence: Optional[float] = None
    macro_dependency: Optional[str] = None

    def normalized_side(self) -> str:
        side = self.side.upper()
        if side not in {"LONG", "SHORT"}:
            raise ValueError(f"Unsupported trade side: {self.side}")
        return side


@dataclass(slots=True)
class AlignedTrade:
    trade: KOLTrade
    market_state: MarketState


@dataclass(slots=True)
class TradeAligner:
    point_in_time: PointInTimeMarketState

    def align_trade(self, trade: KOLTrade) -> AlignedTrade:
        trade.normalized_side()
        # Normalize into a local variable: never mutate the caller's trade.
        if trade.timestamp.tzinfo is None or trade.timestamp.tzinfo.utcoffset(trade.timestamp) is None:
            timestamp = trade.timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = trade.timestamp.astimezone(timezone.utc)
        market_state = self.point_in_time.get_market_state(
            symbol=trade.symbol,
            as_of_timestamp=timestamp,
        )
        return AlignedTrade(trade=trade, market_state=market_state)
