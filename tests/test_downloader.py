from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.data.binance_client import BinanceClient
from src.data.downloader import BinanceDownloader
from src.data.storage import DuckDBStorage


class StubClient:
    """Emulates Binance pagination: aware-UTC rows filtered by [start, end)."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.calls = 0

    def get_klines(self, symbol, interval, start_time=None, end_time=None, limit=1000):
        self.calls += 1
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        return [
            row
            for row in self.rows
            if start_ms <= int(row["open_time"].timestamp() * 1000) < end_ms
        ][:limit]


def _make_rows(n: int) -> list[dict]:
    rows = []
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    for i in range(n):
        open_time = start + timedelta(minutes=i)
        rows.append(
            {
                "open_time": open_time,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10.0,
                "close_time": open_time + timedelta(seconds=59),
                "quote_volume": 1000.0,
                "number_of_trades": 100,
                "taker_buy_base_volume": 5.0,
                "taker_buy_quote_volume": 500.0,
            }
        )
    return rows


def test_downloader_paginates_naive_input_over_aware_bars(tmp_path):
    # Regression: naive CLI start/end vs aware open_time used to raise
    # "can't compare offset-naive and offset-aware datetimes" on page 2.
    storage = DuckDBStorage(root_dir=tmp_path / "raw", database_path=tmp_path / "db" / "m.duckdb")
    client = StubClient(_make_rows(1500))  # spans two 1000-limit pages
    downloader = BinanceDownloader(client=client, storage=storage)

    total = downloader.download_historical_klines(
        symbol="BTCUSDT",
        timeframe="1m",
        start_time=datetime(2026, 8, 1),  # naive, as the CLI passes
        end_time=datetime(2026, 8, 2, 1),
    )

    assert total == 1500
    assert client.calls == 3  # two data pages + one empty probe
    assert len(storage.read_klines("BTCUSDT", "1m")) == 1500


def test_downloader_pacing_sleeps_between_requests(tmp_path, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", sleeps.append)

    storage = DuckDBStorage(root_dir=tmp_path / "raw", database_path=tmp_path / "db" / "m.duckdb")
    client = StubClient(_make_rows(1500))
    downloader = BinanceDownloader(client=client, storage=storage, request_interval=0.5)

    downloader.download_historical_klines(
        symbol="BTCUSDT",
        timeframe="1m",
        start_time=datetime(2026, 8, 1),
        end_time=datetime(2026, 8, 2, 1),
    )

    assert sleeps == [0.5, 0.5]  # once per data page; the final empty probe does not sleep
