from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.data.binance_client import BinanceClient, _to_utc_millis

KLINE = [
    1726819200000,  # open time
    "100.0",
    "101.0",
    "99.0",
    "100.5",
    "10.0",
    1726819259999,  # close time
    "1000.0",
    100,
    "5.0",
    "500.0",
    "0",
]


class StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class StubSession:
    def __init__(self, payload):
        self._payload = payload
        self.headers = {}
        self.calls: list[tuple[str, dict, int]] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        return StubResponse(self._payload)


def _make_client() -> BinanceClient:
    client = BinanceClient()
    client.session = StubSession([KLINE])  # slots allow replacing the session
    return client


def test_get_klines_maps_payload_rows():
    rows = _make_client().get_klines("btcusdt", "1m")
    assert len(rows) == 1
    row = rows[0]
    assert row["open_time"] == datetime.fromtimestamp(KLINE[0] / 1000, tz=timezone.utc)
    assert row["open"] == 100.0
    assert row["high"] == 101.0
    assert row["low"] == 99.0
    assert row["close"] == 100.5
    assert row["volume"] == 10.0
    assert row["close_time"] == datetime.fromtimestamp(KLINE[6] / 1000, tz=timezone.utc)
    assert row["quote_volume"] == 1000.0
    assert row["number_of_trades"] == 100
    assert row["taker_buy_base_volume"] == 5.0
    assert row["taker_buy_quote_volume"] == 500.0


def test_get_klines_request_params():
    client = _make_client()
    client.get_klines("btcusdt", "5m")
    url, params, timeout = client.session.calls[0]
    assert url.endswith("/api/v3/klines")
    assert params["symbol"] == "BTCUSDT"
    assert params["interval"] == "5m"
    assert params["limit"] == 1000
    assert timeout == 30


def test_naive_start_end_interpreted_as_utc():
    client = _make_client()
    start = datetime(2026, 8, 20, 14, 0, 0)
    end = datetime(2026, 8, 20, 15, 0, 0)
    client.get_klines("BTCUSDT", "1m", start_time=start, end_time=end)
    _, params, _ = client.session.calls[0]
    assert params["startTime"] == int(datetime(2026, 8, 20, 14, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    assert params["endTime"] == int(datetime(2026, 8, 20, 15, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)


def test_aware_start_converts_instant_not_wall_clock():
    client = _make_client()
    # 22:00+08:00 is the same instant as 14:00 UTC; a naive wall-clock
    # rewrite would silently send 22:00 UTC instead.
    start = datetime(2026, 8, 20, 22, 0, 0, tzinfo=timezone(timedelta(hours=8)))
    client.get_klines("BTCUSDT", "1m", start_time=start)
    _, params, _ = client.session.calls[0]
    assert params["startTime"] == int(datetime(2026, 8, 20, 14, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)


def test_to_utc_millis_helper():
    naive = datetime(2026, 8, 20, 14, 0, 0)
    assert _to_utc_millis(naive) == int(naive.replace(tzinfo=timezone.utc).timestamp() * 1000)
    aware = datetime(2026, 8, 20, 22, 0, 0, tzinfo=timezone(timedelta(hours=8)))
    assert _to_utc_millis(aware) == _to_utc_millis(naive)
