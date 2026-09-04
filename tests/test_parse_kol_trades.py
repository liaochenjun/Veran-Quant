from __future__ import annotations

import csv

import pytest

from scripts.build_dataset import load_trades
from scripts.parse_kol_trades import FIELDS, ParseError, parse_positions

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
