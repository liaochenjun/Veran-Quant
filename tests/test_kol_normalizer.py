from __future__ import annotations

import csv
import json
from dataclasses import fields
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.data.kol.normalizer import (
    NormalizedKOLEvent,
    ParseError,
    TradeOutcome,
    detect_format,
    parse_chinese_event_stream,
    parse_chinese_position_dump,
    parse_event_stream_auto,
    parse_event_stream_text,
    parse_position_dump,
    parse_position_dump_auto,
    read_events,
    to_trades_rows,
    write_events,
)
from src.dataset.outcome_dataset import OutcomeDataset

_EVENT_FIELD_NAMES = {f.name for f in fields(NormalizedKOLEvent)}

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


def _assert_event_carries_no_outcome(event: NormalizedKOLEvent) -> None:
    # Structural guarantee: outcome fields do not exist on behavior events.
    for forbidden in ("pnl", "closed_price", "closed_at", "holding_time"):
        assert forbidden not in _EVENT_FIELD_NAMES
        assert forbidden not in event.extra


def test_naive_timestamps_are_interpreted_as_beijing_time():
    # Platform dumps carry Beijing wall clock; the pipeline converts to UTC.
    # 2026-09-03 22:59:07 Beijing = 2026-09-03 14:59:07 UTC
    events, outcomes, _ = parse_position_dump(FULL_BLOCK, trader_id="aoying_capital")
    assert events[0].timestamp_utc == "2026-09-03T14:59:07+00:00"
    assert outcomes[0].closed_at_utc == "2026-09-04T14:43:29+00:00"
    # epoch ms must match the UTC instant
    assert events[0].timestamp_ms == int(
        datetime.fromisoformat("2026-09-03T14:59:07+00:00").timestamp() * 1000
    )


def test_position_dump_separates_events_and_outcomes():
    events, outcomes, warnings = parse_position_dump(FULL_BLOCK, trader_id="aoying_capital")

    assert warnings == []
    assert len(events) == 1
    event = events[0]
    assert event.action == "OPEN"
    assert event.side == "LONG"
    assert event.symbol == "ZECUSDT"
    assert event.price == 898.68
    assert event.quantity == 556.327
    assert event.timestamp_utc == "2026-09-03T14:59:07+00:00"
    assert event.leverage == "2x"
    assert event.margin_mode == "Cross"
    _assert_event_carries_no_outcome(event)

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.pnl == 45724.41
    assert outcome.closed_price == 981.42
    assert outcome.closed_at_utc == "2026-09-04T14:43:29+00:00"
    assert outcome.holding_time_seconds == pytest.approx(85462.0)


def test_event_stream_opens_events_closes_outcomes():
    events, outcomes, warnings = parse_event_stream_text(EVENT_STREAM, trader_id="k", year=2026)

    assert warnings == []
    assert [e.action for e in events] == ["OPEN", "OPEN"]
    assert [e.side for e in events] == ["LONG", "SHORT"]
    for event in events:
        _assert_event_carries_no_outcome(event)

    assert len(outcomes) == 1
    assert outcomes[0].pnl == 9635.82
    assert outcomes[0].closed_price == 974.19
    assert outcomes[0].extra["quantity"] == 127.605


def test_trades_rows_join_outcomes_as_metadata_only():
    events, outcomes, _ = parse_position_dump(FULL_BLOCK, trader_id="aoying_capital")
    rows = to_trades_rows(events, outcomes)

    assert len(rows) == 1
    row = rows[0]
    assert row["timestamp"] == "2026-09-03 14:59:07"  # build_dataset contract format
    assert row["side"] == "LONG"
    assert row["entry_price"] == 898.68
    assert row["pnl"] == 45724.41  # metadata column only, never a feature


def test_detect_format():
    assert detect_format(FULL_BLOCK) == "position"
    assert detect_format(EVENT_STREAM) == "event"


def test_write_read_json_roundtrip(tmp_path):
    events, outcomes, _ = parse_event_stream_text(EVENT_STREAM, trader_id="k", year=2026)
    path = tmp_path / "events.json"
    write_events(path, events, outcomes)

    loaded_events, loaded_outcomes = read_events(path)
    assert len(loaded_events) == 2
    assert loaded_events[0].symbol == "SKHYNIXUSDT"
    assert loaded_events[0].timestamp_utc == events[0].timestamp_utc
    assert len(loaded_outcomes) == 1
    assert loaded_outcomes[0].pnl == 9635.82


def test_read_events_csv_splits_outcome_columns(tmp_path):
    path = tmp_path / "events.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trader_id", "symbol", "action", "side", "price", "quantity",
                "timestamp_ms", "timestamp_utc", "source", "closed_price",
                "closed_at_utc", "pnl",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "trader_id": "k",
                "symbol": "BTCUSDT",
                "action": "OPEN",
                "side": "LONG",
                "price": 100.0,
                "quantity": 1.0,
                "timestamp_ms": 1784476800000,
                "timestamp_utc": "2026-08-01T00:00:00+00:00",
                "source": "manual",
                "closed_price": 105.0,
                "closed_at_utc": "2026-08-02T00:00:00+00:00",
                "pnl": 5.0,
            }
        )

    events, outcomes = read_events(path)
    assert len(events) == 1
    _assert_event_carries_no_outcome(events[0])
    assert len(outcomes) == 1
    assert outcomes[0].pnl == 5.0
    assert outcomes[0].closed_price == 105.0


def test_read_events_parquet(tmp_path):
    rows = [
        {
            "trader_id": "k", "symbol": "BTCUSDT", "action": "OPEN", "side": "LONG",
            "price": 100.0, "quantity": 1.0, "timestamp_ms": 1784476800000,
            "timestamp_utc": "2026-08-01T00:00:00+00:00", "source": "manual",
        }
    ]
    path = tmp_path / "events.parquet"
    pd.DataFrame(rows).to_parquet(path)

    events, outcomes = read_events(path)
    assert len(events) == 1
    assert events[0].symbol == "BTCUSDT"
    assert outcomes == []


def test_outcome_dataset_lookup_and_isolation():
    _, outcomes, _ = parse_position_dump(FULL_BLOCK, trader_id="aoying_capital")
    dataset = OutcomeDataset.from_normalizer(outcomes)

    hit = dataset.lookup("ZECUSDT", "2026-09-03T14:59:07+00:00")
    assert hit is not None
    assert hit.pnl == 45724.41
    assert dataset.lookup("ZECUSDT", "2099-01-01T00:00:00+00:00") is None


def test_outcome_dataset_from_json(tmp_path):
    _, outcomes, _ = parse_position_dump(FULL_BLOCK, trader_id="aoying_capital")
    path = tmp_path / "outcomes.json"
    write_events(path, [], outcomes)

    dataset = OutcomeDataset.from_json(path)
    assert len(dataset.outcomes) == 1
    assert dataset.outcomes[0].pnl == 45724.41


# ---------------------------------------------------------------------------
# Chinese formats
# ---------------------------------------------------------------------------

CN_FULL_BLOCK = """\
影响因子：0.5
BTCUSDT
永续
30x
全仓
做空
全部平仓
开仓时间
2026-09-03 22:18:32
开仓价格
80,240.53 USDT
最大持仓量
50.888 BTC
平仓盈亏
+31,490.16 USDT
全部平仓
2026-09-04 03:10:00
平仓均价
79,401.40 USDT
已平仓量
36.639 BTC
"""

CN_PARTIAL_BLOCK = """\
ETHUSDT
永续
20x
全仓
做多
部分平仓
开仓时间
2026-09-03 22:37:48
开仓价格
2,473.77 USDT
最大持仓量
588.888 ETH
平仓盈亏
+12,678.59 USDT
全部平仓
--
平仓均价
2,445.62 USDT
已平仓量
423.999 ETH
"""

CN_PARTIAL_NO_DASH_BLOCK = """\
CLUSDT
永续
24x
全仓
做多
全部平仓
开仓时间
2026-07-29 00:33:17
开仓价格
90.24000 USDT
最大持仓量
5,000.00 CL
平仓盈亏
+600.00 USDT
全部平仓
平仓均价
90.00000 USDT
已平仓量
1,000.00 CL
"""

CN_EVENT_STREAM = """\
09-04, 20:53:30
开多
以均价为90.00000USDT 买入开多CLUSDT永续合约 ，成交数量为 1,000.00CL，总价值为 90,000.00000USDT 。
09-04, 20:00:00
平空
以均价为2,445.62USDT 买入平空ETHUSDT永续合约 ，成交数量为 423.999ETH，总价值为 1,036,941.66USDT ，已实现盈亏为12,986.05USDT。
"""


def test_chinese_position_dump_full_and_partial_variants():
    events, outcomes, warnings = parse_chinese_position_dump(
        CN_FULL_BLOCK + CN_PARTIAL_BLOCK + CN_PARTIAL_NO_DASH_BLOCK, trader_id="xijiuye"
    )

    assert len(events) == 3
    assert events[0].symbol == "BTCUSDT"
    assert events[0].side == "SHORT"
    assert events[0].price == 80240.53
    assert events[0].quantity == 50.888
    assert events[0].leverage == "30x"
    assert events[0].timestamp_utc == "2026-09-03T14:18:32+00:00"
    _assert_event_carries_no_outcome(events[0])

    assert events[1].side == "LONG"  # partial close: 部分平仓 status line handled
    assert events[2].symbol == "CLUSDT"

    # full-close block has an outcome; partial blocks have none
    assert len(outcomes) == 1
    assert outcomes[0].pnl == 31490.16
    assert outcomes[0].closed_at_utc == "2026-09-03T19:10:00+00:00"


def test_chinese_event_stream():
    events, outcomes, warnings = parse_chinese_event_stream(CN_EVENT_STREAM, trader_id="liuyuan", year=2026)

    assert warnings == []
    assert len(events) == 1
    assert events[0].symbol == "CLUSDT"
    assert events[0].side == "LONG"
    assert events[0].price == 90.0
    assert events[0].quantity == 1000.0
    assert events[0].timestamp_utc == "2026-09-04T12:53:30+00:00"
    _assert_event_carries_no_outcome(events[0])

    assert len(outcomes) == 1
    assert outcomes[0].pnl == 12986.05
    assert outcomes[0].closed_price == 2445.62
    assert outcomes[0].side == "SHORT"


def test_auto_dispatch_ignores_impact_factor_header():
    # Regression: the header line used to break language dispatch,
    # routing Chinese files into the English parser.
    events, outcomes, warnings = parse_position_dump_auto(CN_FULL_BLOCK, trader_id="xijiuye")
    assert len(events) == 1
    assert events[0].symbol == "BTCUSDT"
    assert events[0].side == "SHORT"
    assert len(outcomes) == 1

    en_events, en_outcomes, _ = parse_position_dump_auto(FULL_BLOCK, trader_id="aoying_capital")
    assert en_events[0].symbol == "ZECUSDT"


def test_auto_dispatch_events():
    cn_events, cn_outcomes, _ = parse_event_stream_auto(CN_EVENT_STREAM, trader_id="liuyuan", year=2026)
    assert cn_events[0].symbol == "CLUSDT"

    en_events, en_outcomes, _ = parse_event_stream_auto(EVENT_STREAM, trader_id="aoying_capital", year=2026)
    assert en_events[0].symbol == "SKHYNIXUSDT"


# ---------------------------------------------------------------------------
# Timezone regression: Beijing -> UTC semantics (business-critical)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "beijing, utc",
    [
        ("2026-09-03 22:59:07", "2026-09-03T14:59:07+00:00"),
        ("2026-09-04 22:43:29", "2026-09-04T14:43:29+00:00"),
        ("2026-09-04 20:53:30", "2026-09-04T12:53:30+00:00"),
        ("2026-06-03 17:48:45", "2026-06-03T09:48:45+00:00"),
        # cross-year: Beijing 2026-01-01 00:05 -> UTC 2025-12-31 16:05
        ("2026-01-01 00:05:00", "2025-12-31T16:05:00+00:00"),
    ],
)
def test_beijing_to_utc_conversion(beijing, utc):
    from src.data.kol.normalizer import _naive_to_utc

    timestamp_ms, timestamp_utc = _naive_to_utc(beijing)
    assert timestamp_utc == utc
    assert timestamp_ms == int(datetime.fromisoformat(utc).timestamp() * 1000)


def test_pit_uses_converted_utc_trade_time(tmp_path):
    # Business chain: KOL platform "2026-09-03 22:59:07" (Beijing)
    # -> normalizer converts to UTC "2026-09-03 14:59:07" written into the
    #    trades CSV as naive UTC wall clock
    # -> aligner treats the naive CSV timestamp as UTC -> T = 14:59:07 UTC
    # -> bars must satisfy close_time < T (14:59:59 bar excluded).
    from src.alignment.trade_aligner import KOLTrade, TradeAligner
    from src.data.storage import DuckDBStorage
    from src.market.point_in_time import PointInTimeMarketState

    storage = DuckDBStorage(root_dir=tmp_path / "raw", database_path=tmp_path / "db" / "m.duckdb")
    storage.write_klines("BTCUSDT", "1m", [
        {
            "open_time": datetime(2026, 9, 3, 14, 58, tzinfo=timezone.utc),
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0,
            "close_time": datetime(2026, 9, 3, 14, 58, 59, tzinfo=timezone.utc),
            "quote_volume": 1000.0, "number_of_trades": 100,
            "taker_buy_base_volume": 5.0, "taker_buy_quote_volume": 500.0,
        },
        {
            "open_time": datetime(2026, 9, 3, 14, 59, tzinfo=timezone.utc),
            "open": 100.5, "high": 101.5, "low": 100.0, "close": 101.0, "volume": 12.0,
            "close_time": datetime(2026, 9, 3, 14, 59, 59, tzinfo=timezone.utc),  # T's own minute
            "quote_volume": 1200.0, "number_of_trades": 120,
            "taker_buy_base_volume": 6.0, "taker_buy_quote_volume": 600.0,
        },
    ])

    aligner = TradeAligner(point_in_time=PointInTimeMarketState(storage=storage, timeframes=("1m",)))
    trade = KOLTrade(
        kol="aoying_capital", symbol="BTCUSDT",
        timestamp=datetime(2026, 9, 3, 14, 59, 7),  # UTC wall clock from the CSV
        side="SHORT", entry_price=100.0,
    )
    aligned = aligner.align_trade(trade)

    assert aligned.market_state.as_of_timestamp == datetime(2026, 9, 3, 14, 59, 7, tzinfo=timezone.utc)
    one_minute = aligned.market_state.frames["1m"]
    assert len(one_minute) == 1  # 14:59:59 bar is excluded (close_time > T)
    assert one_minute.iloc[0]["close_time"] == datetime(2026, 9, 3, 14, 58, 59, tzinfo=timezone.utc)
