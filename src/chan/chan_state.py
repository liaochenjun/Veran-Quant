"""Stable, serialization-friendly Chan state snapshot.

Every structural field follows the pattern ``value`` + explicit presence
flag (``*_present``).  A structure that does not exist at the snapshot
moment keeps its value fields as ``None`` with ``*_present=False`` —
missing structures are never silently turned into 0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from typing import Any, Optional


def _as_aware_utc(dt: datetime) -> datetime:
    """Normalize to tz-aware UTC, interpreting naive datetimes as UTC."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(slots=True)
class ChanState:
    """Point-in-time chan snapshot for one symbol/timeframe.

    Immutable by convention: consumers must treat a snapshot as frozen data,
    never as a live view into the chan engine.
    """

    symbol: str
    timeframe: str
    as_of_timestamp: str  # ISO 8601, aware UTC
    supported: bool = True  # False when chan.py has no level for this timeframe
    last_bar_close_time: Optional[str] = None  # ISO 8601, close time of last observed bar

    # --- fractal (last merged kline) ---
    fractal_present: bool = False
    fractal_type: Optional[str] = None  # "TOP" | "BOTTOM"
    fractal_price: Optional[float] = None  # top -> high, bottom -> low

    # --- latest bi ---
    bi_count: int = 0
    bi_direction: Optional[str] = None  # "UP" | "DOWN"
    bi_is_sure: Optional[bool] = None
    bi_amplitude: Optional[float] = None  # abs(begin_val - end_val)
    bi_length: Optional[int] = None  # merged klines covered by the bi

    # --- latest segment ---
    segment_count: int = 0
    segment_direction: Optional[str] = None  # "UP" | "DOWN"
    segment_is_sure: Optional[bool] = None

    # --- latest zhongshu ---
    zs_count: int = 0
    zhongshu_present: bool = False
    zhongshu_high: Optional[float] = None
    zhongshu_low: Optional[float] = None
    zhongshu_is_sure: Optional[bool] = None
    # >0 when price is above the zhongshu, <0 when below, 0 when inside
    distance_to_zhongshu: Optional[float] = None

    # --- divergence (back-chi) ---
    # chan.py computes divergence internally for buy/sell points but does not
    # expose it as a standalone field in the open API; kept masked in v1.
    divergence_present: bool = False
    divergence_type: Optional[str] = None
    divergence_strength: Optional[float] = None

    # --- latest buy/sell point ---
    buy_sell_point_present: bool = False
    buy_sell_point_types: Optional[list[str]] = None  # e.g. ["1"], ["3b"]
    buy_sell_point_is_buy: Optional[bool] = None
    buy_sell_point_bi_is_sure: Optional[bool] = None  # chan.py has no bsp-level is_sure
    buy_sell_point_time: Optional[str] = None  # ISO 8601, bar time of the bsp

    @classmethod
    def empty(cls, symbol: str, timeframe: str, as_of_timestamp: datetime, supported: bool = True) -> "ChanState":
        """State with no structures (no data / unsupported timeframe)."""
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            as_of_timestamp=_as_aware_utc(as_of_timestamp).isoformat(),
            supported=supported,
        )

    def to_dict(self) -> dict[str, Any]:
        """Stable dict with keys in declaration order (JSON-friendly)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChanState":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})
