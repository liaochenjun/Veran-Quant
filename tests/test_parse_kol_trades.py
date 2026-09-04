from __future__ import annotations

import csv

import pytest

from scripts.build_dataset import load_trades
from scripts.parse_kol_trades import (
    FIELDS,
    ParseError,
    detect_format,
    parse_event_stream,
    parse_positions,
)

FULL_BLOCK = """\
ZECUSDT
Perp
2x
Cross
Long
Closed
Opened
2026-09-03 22:59:07
Entry Price
898.68 USDT
Max. Open Interest
556.327 ZEC
Closing PNL
+45,724.41 USDT
Closed
2026-09-04 22:43:29
Avg. Close Price
981.42 USDT
Closed Vol.
556.327 ZEC
"""

TRUNCATED_BLOCK = """\
MRVLUSDT
Perp
10x
Cross
Long
Closed
Opened
2026-06-03 15:07:41
Entry Price
334.19851 USDT
Max. Open Interest
146.96 MRVL
Closing PNL
-2,385.64 USDT
Closed
2026-06-03 17:48:45
"""


def test_parse_full_block():
    trades, warnings = parse_positions(FULL_BLOCK, kol="aoying_capital")

    assert warnings == []
    assert len(trades) == 1
    trade = trades[0]
    assert trade["kol"] == "aoying_capital"
    assert trade["symbol"] == "ZECUSDT"
    assert trade["timestamp"] == "2026-09-03 22:59:07"
    assert trade["side"] == "LONG"
    assert trade["entry_price"] == 898.68
    assert trade["leverage"] == 2
    assert trade["margin_mode"] == "Cross"
    assert trade["exit_price"] == 981.42
    assert trade["close_timestamp"] == "2026-09-04 22:43:29"
    assert trade["pnl"] == 45724.41  # commas and sign parsed
    assert trade["position_size"] == 556.327
    assert trade["holding_time_seconds"] == pytest.approx(85462.0)


def test_parse_truncated_block_still_yields_sample():
    # Behavior label (side at open) is complete even without the close tail.
    trades, warnings = parse_positions(TRUNCATED_BLOCK, kol="aoying_capital")

    assert len(trades) == 1
    assert trades[0]["side"] == "LONG"
    assert trades[0]["entry_price"] == 334.19851
    assert trades[0]["close_timestamp"] == "2026-06-03 17:48:45"
    assert trades[0]["exit_price"] is None
    assert "missing close-price tail" in warnings[0]


def test_parse_skips_image_tokens_and_blank_lines():
    text = FULL_BLOCK.replace("ZECUSDT\n", "image\n\nZECUSDT\n")
    trades, warnings = parse_positions(text, kol="k")
    assert len(trades) == 1
    assert trades[0]["symbol"] == "ZECUSDT"


def test_parse_short_side_is_normalized():
    text = FULL_BLOCK.replace("Long", "Short")
    trades, _ = parse_positions(text, kol="k")
    assert trades[0]["side"] == "SHORT"


def test_parse_bad_block_raises():
    with pytest.raises(ParseError):
        parse_positions("NOTASYMBOL\nPerp\n2x\nCross\nLong\n", kol="k")


def test_parser_output_is_loadable_by_dataset_builder(tmp_path):
    # Contract: first five FIELDS are exactly what build_dataset.load_trades reads.
    assert FIELDS[:5] == ["kol", "symbol", "timestamp", "side", "entry_price"]

    trades, _ = parse_positions(FULL_BLOCK + TRUNCATED_BLOCK, kol="aoying_capital")
    csv_path = tmp_path / "trades.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for trade in trades:
            writer.writerow({field: trade[field] for field in FIELDS})

    loaded = load_trades(csv_path)
    assert len(loaded) == 2
    assert loaded[0].symbol == "ZECUSDT"
    assert loaded[0].side == "LONG"
    assert loaded[0].entry_price == 898.68
    assert loaded[1].symbol == "MRVLUSDT"


EVENT_STREAM = """\
09-04, 23:09:09
Open Long
Open a Long position of SKHYNIXUSDT Perpetual at a price of 1,243.29364 USDT, amount of 240.96 SKHYNIX for a total value of 299,584.03578 USDT.
09-04, 22:43:29
Close Long
Close a Long position of ZECUSDT Perpetual at a price of 974.19 USDT, amount of 127.605 ZEC for a total value of 124,311.65 USDT, Realized PNL is 9,635.82 USDT.
09-04, 22:08:40
Open Short
Open a Short position of ZECUSDT Perpetual at a price of 986.71 USDT, amount of 122.600 ZEC for a total value of 120,970.93 USDT.
"""


def test_parse_event_stream_opens_become_samples_closes_are_listed():
    trades, closes, warnings = parse_event_stream(EVENT_STREAM, kol="aoying_capital", year=2026)

    assert warnings == []
    assert len(trades) == 2
    assert trades[0]["symbol"] == "SKHYNIXUSDT"
    assert trades[0]["side"] == "LONG"
    assert trades[0]["timestamp"] == "2026-09-04 23:09:09"
    assert trades[0]["entry_price"] == 1243.29364
    assert trades[0]["position_size"] == 240.96
    assert trades[1]["side"] == "SHORT"

    assert len(closes) == 1
    assert closes[0]["symbol"] == "ZECUSDT"
    assert closes[0]["pnl"] == 9635.82
    assert closes[0]["total_value"] == 124311.65


def test_parse_event_stream_year_rollover():
    text = """\
01-01, 00:05:00
Open Long
Open a Long position of BTCUSDT Perpetual at a price of 100.0 USDT, amount of 1.0 BTC for a total value of 100.0 USDT.
12-31, 23:55:00
Open Short
Open a Short position of BTCUSDT Perpetual at a price of 99.0 USDT, amount of 1.0 BTC for a total value of 99.0 USDT.
"""
    trades, _, warnings = parse_event_stream(text, kol="k", year=2026)
    # newest-first: 12-31 event belongs to the previous year (2025)
    assert trades[0]["timestamp"].startswith("2026-01-01")
    assert trades[1]["timestamp"].startswith("2025-12-31")
    assert warnings == ["year rollover at 12-31 -> 2025"]


def test_detect_format():
    assert detect_format(FULL_BLOCK) == "position"
    assert detect_format(EVENT_STREAM) == "event"


def test_parse_event_stream_mismatched_action_raises():
    text = """\
09-04, 23:09:09
Open Long
Close a Long position of ZECUSDT Perpetual at a price of 974.19 USDT, amount of 127.605 ZEC for a total value of 124,311.65 USDT, Realized PNL is 9,635.82 USDT.
"""
    with pytest.raises(ParseError):
        parse_event_stream(text, kol="k", year=2026)
