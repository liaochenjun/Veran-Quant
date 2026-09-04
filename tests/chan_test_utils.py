"""Shared helpers for chan tests (not a test module itself)."""

from __future__ import annotations

from datetime import datetime, timedelta

from src.data.storage import DuckDBStorage

START = datetime(2026, 8, 1, 0, 0)


def zigzag_rows(n: int = 40, minutes: int = 5) -> list[dict]:
    """Deterministic zigzag series: direction flips every 6 bars."""
    rows: list[dict] = []
    price, direction = 100.0, 1
    for i in range(n):
        open_time = START + timedelta(minutes=minutes * i)
        open_ = price
        close = price + 1.0 * direction
        rows.append(
            {
                "open_time": open_time,
                "open": open_,
                "high": max(open_, close) + 0.2,
                "low": min(open_, close) - 0.2,
                "close": close,
                "volume": 10.0,
                "close_time": open_time + timedelta(minutes=minutes) - timedelta(seconds=1),
                "quote_volume": 1000.0,
                "number_of_trades": 100,
                "taker_buy_base_volume": 5.0,
                "taker_buy_quote_volume": 500.0,
            }
        )
        price = close
        if (i + 1) % 6 == 0:
            direction *= -1
    return rows


def bar_close(i: int, minutes: int = 5) -> datetime:
    return START + timedelta(minutes=minutes * (i + 1)) - timedelta(seconds=1)


def make_storage(tmp_path) -> DuckDBStorage:
    return DuckDBStorage(root_dir=tmp_path / "raw", database_path=tmp_path / "db" / "m.duckdb")


def write_zigzag_history(storage: DuckDBStorage, symbol: str = "BTCUSDT") -> None:
    for timeframe, minutes in [("1m", 1), ("5m", 5), ("15m", 15), ("1h", 60)]:
        storage.write_klines(symbol, timeframe, zigzag_rows(minutes=minutes))
