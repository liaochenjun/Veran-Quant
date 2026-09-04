from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.chan.chan_engine import SimpleChanEngine
from src.market.point_in_time import MarketState, PointInTimeMarketState

AS_OF = datetime.fromisoformat("2026-08-20T14:00:00").replace(tzinfo=timezone.utc)


def _candles(closes) -> pd.DataFrame:
    rows = []
    for i, close in enumerate(closes):
        rows.append(
            {
                "open_time": datetime.fromisoformat("2026-08-20T13:58:00") + pd.Timedelta(minutes=i),
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 10.0,
                "close_time": datetime.fromisoformat("2026-08-20T13:58:59") + pd.Timedelta(minutes=i),
            }
        )
    return pd.DataFrame(rows)


class StubStorage:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.calls = 0

    def read_klines(self, symbol, timeframe, end_before=None, limit=None):
        self.calls += 1
        return self.frame


def test_get_state_without_market_state_fetches_once():
    storage = StubStorage(_candles([100.0, 100.5]))
    engine = SimpleChanEngine(point_in_time=PointInTimeMarketState(storage=storage, timeframes=("1m",)))

    state = engine.get_state(symbol="BTCUSDT", timeframe="1m", as_of_timestamp=AS_OF)

    assert storage.calls == 1
    assert state["trend"] == "up"
    assert state["swing_high"] == 101.5
    assert state["swing_low"] == 99.0
    assert state["as_of_timestamp"] == AS_OF.isoformat()


def test_get_state_reuses_provided_market_state():
    storage = StubStorage(_candles([100.0, 100.5]))
    engine = SimpleChanEngine(point_in_time=PointInTimeMarketState(storage=storage, timeframes=("1m",)))

    market_state = MarketState(
        symbol="BTCUSDT",
        as_of_timestamp=AS_OF,
        frames={"1m": _candles([100.0, 100.5])},
    )
    state = engine.get_state(
        symbol="BTCUSDT", timeframe="1m", as_of_timestamp=AS_OF, market_state=market_state
    )

    assert storage.calls == 0  # regression: no second fetch when state is provided
    assert state["trend"] == "up"


def test_get_state_unknown_for_empty_frame():
    storage = StubStorage(pd.DataFrame())
    engine = SimpleChanEngine(point_in_time=PointInTimeMarketState(storage=storage, timeframes=("1m",)))

    state = engine.get_state(symbol="BTCUSDT", timeframe="1m", as_of_timestamp=AS_OF)

    assert state == {"trend": "unknown", "swing_high": None, "swing_low": None}


def test_trend_detection_up_down_flat():
    storage = StubStorage(_candles([100.0, 101.0]))
    engine = SimpleChanEngine(point_in_time=PointInTimeMarketState(storage=storage, timeframes=("1m",)))

    assert engine.get_state("BTCUSDT", "1m", AS_OF)["trend"] == "up"

    storage.frame = _candles([101.0, 100.0])
    assert engine.get_state("BTCUSDT", "1m", AS_OF)["trend"] == "down"

    storage.frame = _candles([100.0, 100.0])
    assert engine.get_state("BTCUSDT", "1m", AS_OF)["trend"] == "flat"
