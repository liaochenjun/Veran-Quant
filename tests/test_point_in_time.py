from __future__ import annotations

from datetime import datetime

from src.data.storage import DuckDBStorage
from src.market.point_in_time import PointInTimeMarketState


def test_point_in_time_excludes_incomplete_candles(tmp_path):
    storage = DuckDBStorage(
        root_dir=tmp_path / "raw",
        database_path=tmp_path / "db" / "market.duckdb",
    )

    rows = [
        {
            "open_time": datetime.fromisoformat("2026-08-20T14:31:00"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
            "close_time": datetime.fromisoformat("2026-08-20T14:31:59"),
            "quote_volume": 1000.0,
            "number_of_trades": 100,
            "taker_buy_base_volume": 5.0,
            "taker_buy_quote_volume": 500.0,
        },
        {
            "open_time": datetime.fromisoformat("2026-08-20T14:32:00"),
            "open": 100.5,
            "high": 102.0,
            "low": 100.0,
            "close": 101.8,
            "volume": 12.0,
            "close_time": datetime.fromisoformat("2026-08-20T14:32:59"),
            "quote_volume": 1200.0,
            "number_of_trades": 120,
            "taker_buy_base_volume": 6.0,
            "taker_buy_quote_volume": 600.0,
        },
    ]

    storage.write_klines(symbol="BTCUSDT", timeframe="1m", rows=rows)

    as_of = datetime.fromisoformat("2026-08-20T14:32:18")
    point_in_time = PointInTimeMarketState(storage=storage, timeframes=("1m",))
    state = point_in_time.get_market_state(symbol="BTCUSDT", as_of_timestamp=as_of)

    one_minute = state.frames["1m"]
    assert len(one_minute) == 1
    assert one_minute.iloc[0]["close_time"] < as_of
