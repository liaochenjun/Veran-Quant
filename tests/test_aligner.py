from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.alignment.trade_aligner import KOLTrade, TradeAligner
from src.market.point_in_time import PointInTimeMarketState

EMPTY_FRAME = pd.DataFrame()


class StubStorage:
    def __init__(self):
        self.calls: list[tuple[str, str, object, object]] = []

    def read_klines(self, symbol, timeframe, end_before=None, limit=None):
        self.calls.append((symbol, timeframe, end_before, limit))
        return EMPTY_FRAME


def _make_aligner() -> tuple[StubStorage, TradeAligner]:
    storage = StubStorage()
    point_in_time = PointInTimeMarketState(storage=storage, timeframes=("1m",))
    return storage, TradeAligner(point_in_time=point_in_time)


def _make_trade(timestamp: datetime, side: str = "LONG") -> KOLTrade:
    return KOLTrade(kol="kolA", symbol="BTCUSDT", timestamp=timestamp, side=side, entry_price=100.0)


def test_naive_timestamp_treated_as_utc():
    storage, aligner = _make_aligner()
    aligned = aligner.align_trade(_make_trade(datetime(2026, 8, 20, 14, 32, 0)))

    assert aligned.market_state.symbol == "BTCUSDT"
    assert aligned.market_state.as_of_timestamp == datetime(2026, 8, 20, 14, 32, 0, tzinfo=timezone.utc)
    # end_before passed to storage is aware UTC
    assert storage.calls[0][2] == datetime(2026, 8, 20, 14, 32, 0, tzinfo=timezone.utc)


def test_align_does_not_mutate_input_trade():
    storage, aligner = _make_aligner()
    trade = _make_trade(datetime(2026, 8, 20, 14, 32, 0))
    aligner.align_trade(trade)

    assert trade.timestamp == datetime(2026, 8, 20, 14, 32, 0)
    assert trade.timestamp.tzinfo is None


def test_aware_timestamp_converted_to_utc_instant():
    storage, aligner = _make_aligner()
    trade = _make_trade(datetime(2026, 8, 20, 22, 32, 0, tzinfo=timezone(timedelta(hours=8))))
    aligned = aligner.align_trade(trade)

    assert aligned.market_state.as_of_timestamp == datetime(2026, 8, 20, 14, 32, 0, tzinfo=timezone.utc)


def test_side_normalization_and_validation():
    assert _make_trade(datetime(2026, 8, 20), side="long").normalized_side() == "LONG"
    assert _make_trade(datetime(2026, 8, 20), side="SHORT").normalized_side() == "SHORT"
    with pytest.raises(ValueError):
        _make_trade(datetime(2026, 8, 20), side="BOTH").normalized_side()
