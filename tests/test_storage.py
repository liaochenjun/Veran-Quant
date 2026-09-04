from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.data.storage import DuckDBStorage


def _make_rows(open_times: list[datetime], tz_aware: bool = False) -> list[dict]:
    rows = []
    base = 100.0
    for open_time in open_times:
        if tz_aware and open_time.tzinfo is None:
            open_time = open_time.replace(tzinfo=timezone.utc)
        rows.append(
            {
                "open_time": open_time,
                "open": base,
                "high": base + 1.0,
                "low": base - 1.0,
                "close": base + 0.5,
                "volume": 10.0,
                "close_time": open_time + timedelta(seconds=59),
                "quote_volume": 1000.0,
                "number_of_trades": 100,
                "taker_buy_base_volume": 5.0,
                "taker_buy_quote_volume": 500.0,
            }
        )
        base += 0.5
    return rows


def _storage(tmp_path) -> DuckDBStorage:
    return DuckDBStorage(root_dir=tmp_path / "raw", database_path=tmp_path / "db" / "market.duckdb")


def test_write_read_roundtrip_sorted(tmp_path):
    storage = _storage(tmp_path)
    rows = _make_rows([datetime(2026, 8, 20, 14, 31), datetime(2026, 8, 20, 14, 30)])
    storage.write_klines("BTCUSDT", "1m", rows)

    out = storage.read_klines("BTCUSDT", "1m")
    assert list(out["open_time"]) == [datetime(2026, 8, 20, 14, 30), datetime(2026, 8, 20, 14, 31)]


def test_read_end_before_is_strict(tmp_path):
    storage = _storage(tmp_path)
    # close_time is 14:31:59
    storage.write_klines("BTCUSDT", "1m", _make_rows([datetime(2026, 8, 20, 14, 31)]))

    out = storage.read_klines("BTCUSDT", "1m", end_before=datetime(2026, 8, 20, 14, 31, 59))
    assert out.empty
    out = storage.read_klines("BTCUSDT", "1m", end_before=datetime(2026, 8, 20, 14, 32, 0))
    assert len(out) == 1


def test_read_limit_returns_last_n_ascending(tmp_path):
    storage = _storage(tmp_path)
    open_times = [datetime(2026, 8, 20, 14, m) for m in range(30, 35)]
    storage.write_klines("BTCUSDT", "1m", _make_rows(open_times))

    out = storage.read_klines("BTCUSDT", "1m", limit=3)
    assert list(out["open_time"]) == open_times[-3:]
    # end_before applies before limit: the 14:33 candle closes at 14:33:59,
    # which is not < 14:33:30, so the last 3 eligible candles are 14:30-14:32
    out = storage.read_klines("BTCUSDT", "1m", end_before=datetime(2026, 8, 20, 14, 33, 30), limit=3)
    assert list(out["open_time"]) == [datetime(2026, 8, 20, 14, 30), datetime(2026, 8, 20, 14, 31), datetime(2026, 8, 20, 14, 32)]


def test_write_dedupes_by_open_time_keep_last(tmp_path):
    storage = _storage(tmp_path)
    open_time = datetime(2026, 8, 20, 14, 31)
    first = _make_rows([open_time])
    first[0]["close"] = 1.0
    second = _make_rows([open_time])
    second[0]["close"] = 2.0

    storage.write_klines("BTCUSDT", "1m", first)
    storage.write_klines("BTCUSDT", "1m", second)

    out = storage.read_klines("BTCUSDT", "1m")
    assert len(out) == 1
    assert out.iloc[0]["close"] == 2.0


def test_monthly_partitions_merge_across_months(tmp_path):
    storage = _storage(tmp_path)
    storage.write_klines("BTCUSDT", "1m", _make_rows([datetime(2026, 8, 31, 23, 59)]))
    storage.write_klines("BTCUSDT", "1m", _make_rows([datetime(2026, 9, 1, 0, 0)]))

    partition_dir = tmp_path / "raw" / "BTCUSDT" / "1m"
    assert (partition_dir / "2026-08.parquet").exists()
    assert (partition_dir / "2026-09.parquet").exists()

    out = storage.read_klines("BTCUSDT", "1m")
    assert list(out["open_time"]) == [datetime(2026, 8, 31, 23, 59), datetime(2026, 9, 1, 0, 0)]


def test_read_missing_symbol_returns_empty(tmp_path):
    storage = _storage(tmp_path)
    out = storage.read_klines("BTCUSDT", "1m")
    assert out.empty


def test_read_does_not_create_database_file(tmp_path):
    storage = _storage(tmp_path)
    storage.write_klines("BTCUSDT", "1m", _make_rows([datetime(2026, 8, 20, 14, 31)]))
    storage.read_klines("BTCUSDT", "1m")
    assert not storage.database_path.exists()


def test_naive_and_aware_data_are_both_filterable(tmp_path):
    storage = _storage(tmp_path)
    open_time = datetime(2026, 8, 20, 14, 31)
    storage.write_klines("NAIVE", "1m", _make_rows([open_time], tz_aware=False))
    storage.write_klines("AWARE", "1m", _make_rows([open_time], tz_aware=True))

    aware_as_of = datetime(2026, 8, 20, 14, 32, 0, tzinfo=timezone.utc)
    naive_as_of = datetime(2026, 8, 20, 14, 32, 0)

    assert len(storage.read_klines("NAIVE", "1m", end_before=aware_as_of)) == 1
    assert len(storage.read_klines("AWARE", "1m", end_before=naive_as_of)) == 1
    assert storage.read_klines("NAIVE", "1m", end_before=datetime(2026, 8, 20, 14, 31, 59)).empty
    assert storage.read_klines("AWARE", "1m", end_before=datetime(2026, 8, 20, 14, 31, 59, tzinfo=timezone.utc)).empty
