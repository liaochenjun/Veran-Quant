"""Unified KOL data import interface.

Whatever the source (manual text dumps, CSV, JSON, Parquet exports), the
pipeline normalizes everything into two STRICTLY SEPARATED structures:

- ``NormalizedKOLEvent`` — what the KOL did and what was observable at the
  event time. This is the ONLY input to Behavior Cloning.
- ``TradeOutcome`` — what happened AFTER the open (close price/time, PNL,
  holding time). Outcome data may only be used for after-the-fact
  evaluation, never as behavior input (see prompt.txt: 未来数据可以用于
  "评分"，不能用于"出题时给答案").

The normalizer never mixes outcome fields into behavior events; callers
that want both get them as two separate lists.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
_SKIP_TOKENS = {"image", "--"}

# KOL platform dumps carry naive wall-clock timestamps in BEIJING time
# (Asia/Shanghai, UTC+8, no DST). Everything downstream must be UTC, so
# naive timestamps are converted here: beijing wall clock - 8h = UTC.
_BEIJING_OFFSET = timedelta(hours=8)

# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass(slots=True)
class NormalizedKOLEvent:
    """One KOL trading action, normalized across sources.

    Only information available AT the event time may live here. Outcome
    fields (close price/time, PNL, ...) are forbidden by construction —
    they live in ``TradeOutcome`` instead.
    """

    trader_id: str
    symbol: str
    action: str  # "OPEN" | "CLOSE" (future: ADD / REDUCE)
    side: str  # "LONG" | "SHORT"
    price: float
    quantity: float
    timestamp_ms: int  # epoch milliseconds, UTC
    timestamp_utc: str  # ISO 8601 aware UTC
    source: str  # provenance tag, e.g. "manual_position_dump"
    leverage: Optional[str] = None  # e.g. "2x", if known at event time
    margin_mode: Optional[str] = None
    extra: dict = field(default_factory=dict)  # e.g. total_value; NEVER outcome


@dataclass(slots=True)
class TradeOutcome:
    """After-the-fact result of a trade. Evaluation only, never input."""

    trader_id: str
    symbol: str
    side: str
    opened_at_utc: str
    closed_price: Optional[float] = None
    closed_at_utc: Optional[str] = None
    pnl: Optional[float] = None
    holding_time_seconds: Optional[float] = None
    source: str = ""
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


class ParseError(ValueError):
    """Raised when input does not match the expected format."""


def _parse_number(text: str) -> float:
    return float(text.replace(",", ""))


# Impact-factor header lines (影响因子：X) are tolerated and SKIPPED at parse
# time; the value is intentionally not extracted — it must never reach
# behavior inputs.
_IMPACT_FACTOR_RE = re.compile(r"^影响因子[：:]\s*([0-9.]+)\s*$")


def _naive_to_utc(timestamp: str) -> tuple[int, str]:
    """Parse a naive wall-clock timestamp as BEIJING time -> UTC instant."""
    dt_beijing = datetime.strptime(timestamp, _TIMESTAMP_FORMAT)
    # attach tzinfo BEFORE .timestamp(): naive.timestamp() would otherwise
    # be interpreted in the machine's LOCAL timezone (+8 on this host),
    # silently shifting epoch ms by another 8 hours.
    dt_utc = (dt_beijing - _BEIJING_OFFSET).replace(tzinfo=timezone.utc)
    return int(dt_utc.timestamp() * 1000), dt_utc.isoformat()


def _to_timestamp_ms(timestamp_utc: str) -> int:
    dt = datetime.fromisoformat(timestamp_utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


# --------------------------------------------------------------------------
# Format 1: position-history blocks (copy-trading UI dump)
# --------------------------------------------------------------------------

_LEVERAGE_RE = re.compile(r"^\d+x$")
_BLOCK_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


class _BlockReader:
    def __init__(self, lines: list[str], block_start: int):
        self.lines = lines
        self.i = block_start
        self.block_start = block_start

    def take(self, what: str) -> str:
        if self.i >= len(self.lines):
            raise ParseError(f"block starting at line {self.block_start + 1} ends before '{what}'")
        token = self.lines[self.i]
        self.i += 1
        return token

    def expect(self, marker: str) -> None:
        token = self.take(marker)
        if token != marker:
            raise ParseError(
                f"line {self.i}: expected {marker!r}, got {token!r} "
                f"(block starts at line {self.block_start + 1})"
            )

    def take_price(self, what: str) -> float:
        self.expect(what)
        value, _, _ = self.take(f"value of {what}").partition(" ")
        return _parse_number(value)

    def take_quantity(self, what: str) -> tuple[float, str]:
        self.expect(what)
        value, _, unit = self.take(f"value of {what}").partition(" ")
        return _parse_number(value), unit


def parse_position_dump(
    text: str, trader_id: str, source: str = "manual_position_dump"
) -> tuple[list[NormalizedKOLEvent], list[TradeOutcome], list[str]]:
    """Parse the position-history block format into events + outcomes."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line and line not in _SKIP_TOKENS]

    events: list[NormalizedKOLEvent] = []
    outcomes: list[TradeOutcome] = []
    warnings: list[str] = []

    i = 0
    while i < len(lines):
        # Header lines (e.g. 影响因子：1.5) sit between blocks; anything whose
        # next line is not "Perp" is not a block start and is skipped.
        if i + 1 >= len(lines) or lines[i + 1] != "Perp":
            warnings.append(f"skipped non-block line {i + 1}: {lines[i]!r}")
            i += 1
            continue
        block_start = i
        reader = _BlockReader(lines, block_start)
        symbol = reader.take("symbol").upper()
        reader.expect("Perp")
        leverage = reader.take("leverage")
        if not _LEVERAGE_RE.match(leverage):
            raise ParseError(f"line {reader.i}: bad leverage {leverage!r}")
        margin_mode = reader.take("margin mode")
        side = reader.take("side").upper()
        if side not in {"LONG", "SHORT"}:
            raise ParseError(f"line {reader.i}: bad side {side!r}")
        reader.take("status")  # Closed / Opened, kept as info
        reader.expect("Opened")
        opened_at = reader.take("open timestamp")
        if not _BLOCK_TS_RE.match(opened_at):
            raise ParseError(f"line {reader.i}: bad open timestamp {opened_at!r}")
        entry_price = reader.take_price("Entry Price")
        max_oi, _ = reader.take_quantity("Max. Open Interest")
        pnl = reader.take_price("Closing PNL")

        # Outcome tail; truncated dumps may end anywhere after the PNL line.
        exit_price: Optional[float] = None
        closed_at: Optional[str] = None
        closed_vol: Optional[float] = None
        if reader.i < len(lines) and lines[reader.i] == "Closed":
            reader.i += 1
            closed_at = reader.take("close timestamp")
            if not _BLOCK_TS_RE.match(closed_at):
                raise ParseError(f"line {reader.i}: bad close timestamp {closed_at!r}")
            if reader.i < len(lines):
                if lines[reader.i] == "Avg. Close Price":
                    exit_price = reader.take_price("Avg. Close Price")
                    closed_vol, _ = reader.take_quantity("Closed Vol.")
                else:
                    warnings.append(f"{symbol}: truncated after close timestamp")
            else:
                warnings.append(f"{symbol}: missing close-price tail")
        else:
            warnings.append(f"{symbol}: truncated before close timestamp")

        opened_ms, opened_iso = _naive_to_utc(opened_at)
        events.append(
            NormalizedKOLEvent(
                trader_id=trader_id,
                symbol=symbol,
                action="OPEN",
                side=side,
                price=entry_price,
                quantity=max_oi,
                timestamp_ms=opened_ms,
                timestamp_utc=opened_iso,
                source=source,
                leverage=leverage,
                margin_mode=margin_mode,
            )
        )

        if closed_at:
            closed_dt = datetime.strptime(closed_at, _TIMESTAMP_FORMAT) - _BEIJING_OFFSET
            opened_dt = datetime.strptime(opened_at, _TIMESTAMP_FORMAT) - _BEIJING_OFFSET
            outcomes.append(
                TradeOutcome(
                    trader_id=trader_id,
                    symbol=symbol,
                    side=side,
                    opened_at_utc=opened_iso,
                    closed_price=exit_price,
                    closed_at_utc=closed_dt.isoformat() + "+00:00",
                    pnl=pnl,
                    holding_time_seconds=(closed_dt - opened_dt).total_seconds(),
                    source=source,
                    extra={"closed_volume": closed_vol},
                )
            )
        i = reader.i

    return events, outcomes, warnings


# --------------------------------------------------------------------------
# Format 2: per-action event stream (signal feed)
# --------------------------------------------------------------------------

_EVENT_TS_RE = re.compile(r"^(\d{2})-(\d{2}), (\d{2}:\d{2}:\d{2})$")
_EVENT_ACTION_RE = re.compile(r"^(Open|Close) (Long|Short)$")
_EVENT_DETAIL_RE = re.compile(
    r"^(Open|Close) a (Long|Short) position of (\S+) Perpetual at a price of "
    r"([0-9][0-9,.]*) USDT, amount of ([0-9][0-9,.]*) ([A-Z0-9]+) for a total value of "
    r"([0-9][0-9,.]*) USDT(?:\.|,)(?: Realized PNL is (-?[0-9][0-9,.]*) USDT\.)?$"
)


def parse_event_stream_text(
    text: str, trader_id: str, year: int = 2026, source: str = "manual_event_stream"
) -> tuple[list[NormalizedKOLEvent], list[TradeOutcome], list[str]]:
    """Parse the per-action signal feed; opens -> events, closes -> outcomes."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    events: list[NormalizedKOLEvent] = []
    outcomes: list[TradeOutcome] = []
    warnings: list[str] = []

    previous_month: Optional[int] = None
    i = 0
    while i + 2 < len(lines):
        ts_match = _EVENT_TS_RE.match(lines[i])
        if not ts_match:
            raise ParseError(f"line {i + 1}: expected event timestamp, got {lines[i]!r}")
        action_match = _EVENT_ACTION_RE.match(lines[i + 1])
        if not action_match:
            raise ParseError(f"line {i + 2}: expected action line, got {lines[i + 1]!r}")
        detail_match = _EVENT_DETAIL_RE.match(lines[i + 2])
        if not detail_match:
            raise ParseError(f"line {i + 3}: unparsable detail line, got {lines[i + 2]!r}")

        month, day, hms = ts_match.groups()
        month_int = int(month)
        if previous_month is not None and month_int == 12 and previous_month <= 6:
            # File is newest-first; crossing BACK into December from the
            # first half of the year means the previous year (e.g. ... 01-xx
            # then 12-xx). Small out-of-order month bumps (8 -> 9) are NOT
            # year crossings and must not roll the year back.
            year -= 1
            warnings.append(f"year rollover at {month}-{day} -> {year}")
        previous_month = month_int

        detail_action, detail_side = detail_match.group(1), detail_match.group(2)
        if detail_action != action_match.group(1) or detail_side != action_match.group(2):
            raise ParseError(
                f"line {i + 2}: action line {action_match.group(0)!r} "
                f"does not match detail {detail_match.group(0)!r}"
            )

        symbol = detail_match.group(3)
        price = _parse_number(detail_match.group(4))
        amount = _parse_number(detail_match.group(5))
        total_value = _parse_number(detail_match.group(7))
        pnl = _parse_number(detail_match.group(8)) if detail_match.group(8) else None
        side = "LONG" if detail_side == "Long" else "SHORT"
        timestamp = f"{year}-{month}-{day} {hms}"
        event_ms, event_iso = _naive_to_utc(timestamp)

        if detail_action == "Open":
            events.append(
                NormalizedKOLEvent(
                    trader_id=trader_id,
                    symbol=symbol,
                    action="OPEN",
                    side=side,
                    price=price,
                    quantity=amount,
                    timestamp_ms=event_ms,
                    timestamp_utc=event_iso,
                    source=source,
                    extra={"total_value": total_value},
                )
            )
        else:
            outcomes.append(
                TradeOutcome(
                    trader_id=trader_id,
                    symbol=symbol,
                    side=side,
                    opened_at_utc="",  # FIFO matching is portfolio accounting, done elsewhere
                    closed_price=price,
                    closed_at_utc=event_iso,
                    pnl=pnl,
                    holding_time_seconds=None,
                    source=source,
                    extra={"total_value": total_value, "quantity": amount},
                )
            )
        i += 3

    if i < len(lines):
        raise ParseError(f"line {i + 1}: trailing unparsed lines: {lines[i:]!r}")

    return events, outcomes, warnings


def detect_format(text: str) -> str:
    """'event' if the first content line looks like an event timestamp."""
    for line in text.splitlines():
        line = line.strip()
        if not line or line in _SKIP_TOKENS:
            continue
        return "event" if _EVENT_TS_RE.match(line) else "position"
    return "position"


# --------------------------------------------------------------------------
# Format 3: Chinese position-history blocks (永续/全仓/做多/做空/...)
# --------------------------------------------------------------------------

_CH_SIDES = {"做多": "LONG", "做空": "SHORT"}
_CH_STATUSES = {"部分平仓", "全部平仓"}


def parse_chinese_position_dump(
    text: str, trader_id: str, source: str = "manual_position_dump_cn"
) -> tuple[list[NormalizedKOLEvent], list[TradeOutcome], list[str]]:
    """Parse the Chinese position-history block format.

    Block layout (status line after the side is optional):

        SYMBOL / 永续 / Nx / 全仓 / 做多|做空 / [部分平仓|全部平仓]
        开仓时间 ts / 开仓价格 p USDT / 最大持仓量 q U / 平仓盈亏 pnl USDT
        全部平仓 [close_ts | "--"] / 平仓均价 p USDT / 已平仓量 q U
    """
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line and line not in _SKIP_TOKENS]

    events: list[NormalizedKOLEvent] = []
    outcomes: list[TradeOutcome] = []
    warnings: list[str] = []

    i = 0
    while i < len(lines):
        if i + 1 >= len(lines) or lines[i + 1] != "永续":
            warnings.append(f"skipped non-block line {i + 1}: {lines[i]!r}")
            i += 1
            continue
        block_start = i
        symbol = lines[i].upper()
        i += 2  # symbol + 永续
        leverage = lines[i]
        if not _LEVERAGE_RE.match(leverage):
            raise ParseError(f"line {i + 1}: bad leverage {leverage!r} (block starts at {block_start + 1})")
        i += 1
        if lines[i] != "全仓":
            raise ParseError(f"line {i + 1}: expected 全仓, got {lines[i]!r}")
        i += 1
        side_token = lines[i]
        if side_token not in _CH_SIDES:
            raise ParseError(f"line {i + 1}: bad side {side_token!r}")
        side = _CH_SIDES[side_token]
        i += 1
        if i < len(lines) and lines[i] in _CH_STATUSES:
            i += 1  # 部分平仓 / 全部平仓 status line
        if lines[i] != "开仓时间":
            raise ParseError(f"line {i + 1}: expected 开仓时间, got {lines[i]!r}")
        i += 1
        opened_at = lines[i]
        if not _BLOCK_TS_RE.match(opened_at):
            raise ParseError(f"line {i + 1}: bad open timestamp {opened_at!r}")
        i += 1
        if lines[i] != "开仓价格":
            raise ParseError(f"line {i + 1}: expected 开仓价格, got {lines[i]!r}")
        i += 1
        entry_price = _parse_number(lines[i].partition(" ")[0])
        i += 1
        if lines[i] != "最大持仓量":
            raise ParseError(f"line {i + 1}: expected 最大持仓量, got {lines[i]!r}")
        i += 1
        max_oi = _parse_number(lines[i].partition(" ")[0])
        i += 1
        if lines[i] != "平仓盈亏":
            raise ParseError(f"line {i + 1}: expected 平仓盈亏, got {lines[i]!r}")
        i += 1
        pnl = _parse_number(lines[i].partition(" ")[0])
        i += 1
        if lines[i] != "全部平仓":
            raise ParseError(f"line {i + 1}: expected 全部平仓, got {lines[i]!r}")
        i += 1
        closed_at: Optional[str] = None
        if lines[i] == "--":
            i += 1  # partial close, no full close time
        elif lines[i] != "平仓均价":
            # fully closed: a close timestamp follows 全部平仓
            closed_at = lines[i]
            if not _BLOCK_TS_RE.match(closed_at):
                raise ParseError(f"line {i + 1}: bad close timestamp {closed_at!r}")
            i += 1
        if lines[i] != "平仓均价":
            raise ParseError(f"line {i + 1}: expected 平仓均价, got {lines[i]!r}")
        i += 1
        exit_price = _parse_number(lines[i].partition(" ")[0])
        i += 1
        if lines[i] != "已平仓量":
            raise ParseError(f"line {i + 1}: expected 已平仓量, got {lines[i]!r}")
        i += 1
        closed_vol = _parse_number(lines[i].partition(" ")[0])
        i += 1

        opened_ms, opened_iso = _naive_to_utc(opened_at)
        events.append(
            NormalizedKOLEvent(
                trader_id=trader_id,
                symbol=symbol,
                action="OPEN",
                side=side,
                price=entry_price,
                quantity=max_oi,
                timestamp_ms=opened_ms,
                timestamp_utc=opened_iso,
                source=source,
                leverage=leverage,
                margin_mode="Cross",
            )
        )
        if closed_at:
            closed_dt = datetime.strptime(closed_at, _TIMESTAMP_FORMAT) - _BEIJING_OFFSET
            opened_dt = datetime.strptime(opened_at, _TIMESTAMP_FORMAT) - _BEIJING_OFFSET
            outcomes.append(
                TradeOutcome(
                    trader_id=trader_id,
                    symbol=symbol,
                    side=side,
                    opened_at_utc=opened_iso,
                    closed_price=exit_price,
                    closed_at_utc=closed_dt.isoformat() + "+00:00",
                    pnl=pnl,
                    holding_time_seconds=(closed_dt - opened_dt).total_seconds(),
                    source=source,
                    extra={"closed_volume": closed_vol},
                )
            )
        else:
            warnings.append(f"{symbol}: partial close, no full close time")

    return events, outcomes, warnings


# --------------------------------------------------------------------------
# Format 4: Chinese per-action event stream (开多/开空/平多/平空)
# --------------------------------------------------------------------------

_CH_EVENT_ACTION_RE = re.compile(r"^(开多|开空|平多|平空)$")
_CH_EVENT_DETAIL_RE = re.compile(
    r"^以均价为([0-9][0-9,.]*)USDT\s*(买入|卖出)\s*(开多|开空|平多|平空)(\S+)永续合约\s*，\s*成交数量为\s*"
    r"([0-9][0-9,.]*)([A-Z0-9]+)，总价值为\s*([0-9][0-9,.]*)USDT\s*(?:，|,)?\s*"
    r"(?:已实现盈亏为(-?[0-9][0-9,.]*)USDT)?\s*。?\s*$"
)

_CH_ACTION_MAP = {
    "开多": ("Open", "LONG"),
    "开空": ("Open", "SHORT"),
    "平多": ("Close", "LONG"),
    "平空": ("Close", "SHORT"),
}


def parse_chinese_event_stream(
    text: str, trader_id: str, year: int = 2026, source: str = "manual_event_stream_cn"
) -> tuple[list[NormalizedKOLEvent], list[TradeOutcome], list[str]]:
    """Parse the Chinese signal feed; opens -> events, closes -> outcomes."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    lines = [line for line in lines if line not in _SKIP_TOKENS]

    events: list[NormalizedKOLEvent] = []
    outcomes: list[TradeOutcome] = []
    warnings: list[str] = []

    previous_month: Optional[int] = None
    i = 0
    while i + 2 < len(lines):
        ts_match = _EVENT_TS_RE.match(lines[i])
        if not ts_match:
            raise ParseError(f"line {i + 1}: expected event timestamp, got {lines[i]!r}")
        action_match = _CH_EVENT_ACTION_RE.match(lines[i + 1])
        if not action_match:
            raise ParseError(f"line {i + 2}: expected action line, got {lines[i + 1]!r}")
        detail_match = _CH_EVENT_DETAIL_RE.match(lines[i + 2])
        if not detail_match:
            raise ParseError(f"line {i + 3}: unparsable detail line, got {lines[i + 2]!r}")

        month, day, hms = ts_match.groups()
        month_int = int(month)
        if previous_month is not None and month_int == 12 and previous_month <= 6:
            # Same rule as the English event parser: only a jump back into
            # December from the first half of the year is a year crossing.
            year -= 1
            warnings.append(f"year rollover at {month}-{day} -> {year}")
        previous_month = month_int

        price = _parse_number(detail_match.group(1))
        detail_action = detail_match.group(3)
        symbol = detail_match.group(4)
        amount = _parse_number(detail_match.group(5))
        total_value = _parse_number(detail_match.group(7))
        pnl = _parse_number(detail_match.group(8)) if detail_match.group(8) else None
        action, side = _CH_ACTION_MAP[detail_action]
        timestamp = f"{year}-{month}-{day} {hms}"
        event_ms, event_iso = _naive_to_utc(timestamp)

        if action == "Open":
            events.append(
                NormalizedKOLEvent(
                    trader_id=trader_id,
                    symbol=symbol,
                    action="OPEN",
                    side=side,
                    price=price,
                    quantity=amount,
                    timestamp_ms=event_ms,
                    timestamp_utc=event_iso,
                    source=source,
                    extra={"total_value": total_value},
                )
            )
        else:
            outcomes.append(
                TradeOutcome(
                    trader_id=trader_id,
                    symbol=symbol,
                    side=side,
                    opened_at_utc="",  # FIFO matching is portfolio accounting
                    closed_price=price,
                    closed_at_utc=event_iso,
                    pnl=pnl,
                    holding_time_seconds=None,
                    source=source,
                    extra={"total_value": total_value, "quantity": amount},
                )
            )
        i += 3

    if i < len(lines):
        raise ParseError(f"line {i + 1}: trailing unparsed lines: {lines[i:]!r}")

    return events, outcomes, warnings


# --------------------------------------------------------------------------
# Auto-dispatch helpers
# --------------------------------------------------------------------------


def _first_content_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        # impact-factor headers sit before the first block and must not
        # confuse format dispatch
        if not line or line in _SKIP_TOKENS or _IMPACT_FACTOR_RE.match(line):
            continue
        lines.append(line)
        if len(lines) >= 2:
            break
    return lines


def parse_position_dump_auto(
    text: str, trader_id: str, source: str | None = None
) -> tuple[list[NormalizedKOLEvent], list[TradeOutcome], list[str]]:
    first = _first_content_lines(text)
    if len(first) >= 2 and first[1] == "永续":
        return parse_chinese_position_dump(
            text, trader_id, source or "manual_position_dump_cn"
        )
    return parse_position_dump(text, trader_id, source or "manual_position_dump")


def parse_event_stream_auto(
    text: str, trader_id: str, year: int = 2026, source: str | None = None
) -> tuple[list[NormalizedKOLEvent], list[TradeOutcome], list[str]]:
    first = _first_content_lines(text)
    if len(first) >= 2 and _CH_EVENT_ACTION_RE.match(first[1]):
        return parse_chinese_event_stream(
            text, trader_id, year, source or "manual_event_stream_cn"
        )
    return parse_event_stream_text(text, trader_id, year, source or "manual_event_stream")


# --------------------------------------------------------------------------
# Generic import interface (CSV / JSON / Parquet)
# --------------------------------------------------------------------------

# Columns of the normalized event schema; outcome columns may optionally
# appear alongside and are split out into TradeOutcome records.
_EVENT_COLUMNS = ("trader_id", "symbol", "action", "side", "price", "quantity",
                  "timestamp_ms", "timestamp_utc")
_EVENT_OPTIONAL_COLUMNS = ("source", "leverage", "margin_mode")
_OUTCOME_COLUMNS = ("closed_price", "closed_at_utc", "pnl", "holding_time_seconds")


def _row_to_event(row: dict) -> NormalizedKOLEvent:
    missing = [c for c in _EVENT_COLUMNS if c not in row or row[c] in (None, "")]
    if missing:
        raise ParseError(f"normalized event row missing columns: {missing}")
    return NormalizedKOLEvent(
        trader_id=str(row["trader_id"]),
        symbol=str(row["symbol"]).upper(),
        action=str(row["action"]).upper(),
        side=str(row["side"]).upper(),
        price=float(row["price"]),
        quantity=float(row["quantity"]),
        timestamp_ms=int(float(row["timestamp_ms"])),
        timestamp_utc=str(row["timestamp_utc"]),
        source=str(row.get("source") or ""),
        leverage=row.get("leverage") or None,
        margin_mode=row.get("margin_mode") or None,
    )


def _row_to_outcome(row: dict) -> Optional[TradeOutcome]:
    has_outcome = any(row.get(c) not in (None, "") for c in _OUTCOME_COLUMNS)
    if not has_outcome:
        return None
    return TradeOutcome(
        trader_id=str(row.get("trader_id", "")),
        symbol=str(row.get("symbol", "")).upper(),
        side=str(row.get("side", "")).upper(),
        opened_at_utc=str(row.get("timestamp_utc", "")),
        closed_price=float(row["closed_price"]) if row.get("closed_price") not in (None, "") else None,
        closed_at_utc=str(row["closed_at_utc"]) if row.get("closed_at_utc") not in (None, "") else None,
        pnl=float(row["pnl"]) if row.get("pnl") not in (None, "") else None,
        holding_time_seconds=(
            float(row["holding_time_seconds"])
            if row.get("holding_time_seconds") not in (None, "")
            else None
        ),
        source=str(row.get("source") or ""),
    )


def read_events(path: str | Path) -> tuple[list[NormalizedKOLEvent], list[TradeOutcome]]:
    """Import normalized events from CSV / JSON / Parquet.

    Outcome columns, when present, are split into TradeOutcome records —
    they never enter the behavior event list.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    elif path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, dict):  # {"events": [...], "outcomes": [...]} form
            return _from_events_outcomes_dict(rows, source=path.stem)
    elif path.suffix.lower() == ".parquet":
        rows = pd.read_parquet(path).to_dict(orient="records")
    else:
        raise ParseError(f"unsupported import format: {path.suffix}")

    events, outcomes = [], []
    for row in rows:
        events.append(_row_to_event(row))
        outcome = _row_to_outcome(row)
        if outcome is not None:
            outcomes.append(outcome)
    return events, outcomes


def _from_events_outcomes_dict(
    data: dict, source: str
) -> tuple[list[NormalizedKOLEvent], list[TradeOutcome]]:
    events = [_row_to_event(row) for row in data.get("events", [])]
    outcomes = [
        TradeOutcome(**{k: v for k, v in row.items() if k in TradeOutcome.__dataclass_fields__})
        for row in data.get("outcomes", [])
    ]
    return events, outcomes


def write_events(path: str | Path, events: Iterable[NormalizedKOLEvent],
                 outcomes: Iterable[TradeOutcome] | None = None) -> None:
    """Write events (and optionally outcomes) in JSON form with separate lists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "events": [asdict(event) for event in events],
        "outcomes": [asdict(outcome) for outcome in (outcomes or [])],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# build_dataset contract rows
# --------------------------------------------------------------------------

# First five columns are exactly what build_dataset.load_trades consumes;
# the rest are outcome metadata and must NOT become model inputs.
FIELDS = [
    "kol",
    "symbol",
    "timestamp",
    "side",
    "entry_price",
    "leverage",
    "margin_mode",
    "exit_price",
    "close_timestamp",
    "pnl",
    "position_size",
    "holding_time_seconds",
]

EVENT_FIELDS = ["timestamp", "symbol", "side", "price", "amount", "total_value", "pnl"]


def to_trades_rows(
    events: Iterable[NormalizedKOLEvent],
    outcomes: Iterable[TradeOutcome] | None = None,
) -> list[dict]:
    """Open events -> behavior rows; outcomes joined back ONLY as metadata
    columns (evaluated-after-the-fact), never as behavior features."""
    outcome_by_key = {
        (o.symbol, o.opened_at_utc): o for o in (outcomes or []) if o.opened_at_utc
    }
    rows = []
    for event in events:
        if event.action != "OPEN":
            continue
        outcome = outcome_by_key.get((event.symbol, event.timestamp_utc))
        closed_dt = (
            datetime.fromisoformat(outcome.closed_at_utc).strftime(_TIMESTAMP_FORMAT)
            if outcome and outcome.closed_at_utc
            else None
        )
        rows.append(
            {
                "kol": event.trader_id,
                "symbol": event.symbol,
                "timestamp": event.timestamp_utc[:19].replace("T", " "),
                "side": event.side,
                "entry_price": event.price,
                "leverage": int(event.leverage[:-1]) if event.leverage else None,
                "margin_mode": event.margin_mode,
                "exit_price": outcome.closed_price if outcome else None,
                "close_timestamp": closed_dt,
                "pnl": outcome.pnl if outcome else None,
                "position_size": event.quantity,
                "holding_time_seconds": outcome.holding_time_seconds if outcome else None,
            }
        )
    return rows


def outcomes_to_rows(outcomes: Iterable[TradeOutcome]) -> list[dict]:
    """Outcome records as close-event rows (EVENT_FIELDS contract)."""
    rows = []
    for outcome in outcomes:
        if not outcome.closed_at_utc:
            continue
        closed_naive = datetime.fromisoformat(outcome.closed_at_utc).strftime(_TIMESTAMP_FORMAT)
        rows.append(
            {
                "timestamp": closed_naive,
                "symbol": outcome.symbol,
                "side": outcome.side,
                "price": outcome.closed_price,
                "amount": outcome.extra.get("closed_volume", outcome.extra.get("quantity")),
                "total_value": outcome.extra.get("total_value"),
                "pnl": outcome.pnl,
            }
        )
    return rows
