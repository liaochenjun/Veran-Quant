from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import requests

from src.data.bybit_client import BybitClient
from src.data.hyperliquid_client import HyperliquidClient


class StubResponse:
    status_code = 200
    headers: dict = {}

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class RateLimitedResponse:
    status_code = 429
    headers = {"Retry-After": "0"}

    def raise_for_status(self):
        raise requests.HTTPError("429 Client Error")

    def json(self):
        return {}


class ScriptedSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[tuple[str, str, dict, dict]] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(("get", url, params or {}, {}))
        return self._responses.pop(0)

    def post(self, url, json=None, timeout=None):
        self.calls.append(("post", url, {}, json or {}))
        return self._responses.pop(0)


BYBIT_PAYLOAD = {
    "result": {
        "list": [
            ["1788566400000", "101.0", "102.0", "99.0", "100.5", "10.0", "1000.0"],  # newest
            ["1788566100000", "100.0", "101.0", "99.0", "100.0", "9.0", "900.0"],  # oldest
        ]
    }
}


def test_bybit_params_and_row_mapping():
    client = BybitClient()
    client.session = ScriptedSession([StubResponse(BYBIT_PAYLOAD)])

    start = datetime(2026, 8, 20, 14, 0, 0)
    rows = client.get_klines("xauusdt", "1h", start_time=start)

    method, url, params, _ = client.session.calls[0]
    assert method == "get"
    assert url.endswith("/v5/market/kline")
    assert params["category"] == "linear"
    assert params["symbol"] == "XAUUSDT"
    assert params["interval"] == "60"  # 1h -> Bybit code
    assert params["start"] == int(start.replace(tzinfo=timezone.utc).timestamp() * 1000)
    assert params["limit"] == 1000

    # newest-first input is normalized to ascending order
    assert rows[0]["open_time"] < rows[1]["open_time"]
    row = rows[0]
    assert row["open"] == 100.0 and row["close"] == 100.0
    assert row["close_time"] == row["open_time"] + timedelta(minutes=60) - timedelta(milliseconds=1)
    assert row["quote_volume"] == 900.0
    assert row["number_of_trades"] == 0  # not provided by endpoint


def test_bybit_retries_on_rate_limit(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = BybitClient()
    client.session = ScriptedSession([RateLimitedResponse(), StubResponse(BYBIT_PAYLOAD)])

    rows = client.get_klines("XAUUSDT", "1h")
    assert len(rows) == 2
    assert len(client.session.calls) == 2


HL_PAYLOAD = [
    {"t": 1788566400000, "T": 1788566459999, "s": "HYPE", "i": "1m",
     "o": "84.1", "c": "84.2", "h": "84.5", "l": "83.9", "v": "122557.85", "n": 8041},
]


def test_hyperliquid_params_and_row_mapping():
    client = HyperliquidClient()
    client.session = ScriptedSession([StubResponse(HL_PAYLOAD)])

    start = datetime(2026, 8, 20, 14, 0, 0, tzinfo=timezone.utc)
    rows = client.get_klines("HYPEUSDT", "1m", start_time=start)

    method, url, _, body = client.session.calls[0]
    assert method == "post"
    assert url.endswith("/info")
    assert body["type"] == "candleSnapshot"
    assert body["req"]["coin"] == "HYPE"  # -USDT suffix stripped
    assert body["req"]["interval"] == "1m"
    assert body["req"]["startTime"] == int(start.timestamp() * 1000)

    row = rows[0]
    assert row["open_time"] == datetime.fromtimestamp(1788566400000 / 1000, tz=timezone.utc)
    assert row["close"] == 84.2
    assert row["volume"] == 122557.85
    assert row["number_of_trades"] == 8041
    assert row["quote_volume"] == 0.0  # not provided by endpoint


def test_hyperliquid_retries_on_rate_limit(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = HyperliquidClient()
    client.session = ScriptedSession([RateLimitedResponse(), StubResponse(HL_PAYLOAD)])

    rows = client.get_klines("HYPEUSDT", "1m")
    assert len(rows) == 1
    assert len(client.session.calls) == 2


def test_bybit_rejects_unknown_interval():
    with pytest.raises(ValueError):
        BybitClient().get_klines("XAUUSDT", "2h")


def test_bybit_paginates_backward_over_the_window():
    # Bybit returns the NEWEST limit rows of [start, end]; the client must
    # move the window end backward until the requested start is covered.
    def make_page(newest_open_ms: int, n: int) -> dict:
        rows = []
        for i in range(n):
            open_ms = newest_open_ms - i * 3_600_000  # 1h steps, newest first
            rows.append([str(open_ms), "100.0", "101.0", "99.0", "100.5", "10.0", "1000.0"])
        return {"result": {"list": rows}}

    page1 = make_page(1788566400000, 1000)  # full page -> paginate further
    page2 = make_page(1788566400000 - 1000 * 3_600_000, 500)  # partial -> exhausted
    client = BybitClient()
    client.session = ScriptedSession([StubResponse(page1), StubResponse(page2)])

    start = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)  # before both pages
    rows = client.get_klines("XAUUSDT", "1h", start_time=start)

    assert len(rows) == 1500
    assert [r["open_time"] for r in rows] == sorted(r["open_time"] for r in rows)
    # second request's end moved to just before the oldest row of page 1
    _, _, params2, _ = client.session.calls[1]
    assert params2["end"] == 1788566400000 - 999 * 3_600_000 - 1
