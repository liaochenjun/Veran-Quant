"""Parse KOL trading-history dumps into the normalized trades CSV.

Two input formats are supported (auto-detected):

1. Position-history blocks (copy-trading UI), one position per block:

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

2. Event stream (per-action signal feed), one event per triple:

       09-04, 23:09:09
       Open Long
       Open a Long position of SKHYNIXUSDT Perpetual at a price of
       1,243.29364 USDT, amount of 240.96 SKHYNIX for a total value of
       299,584.03578 USDT.

   Every "Open" event becomes one behavior sample; "Close" events are
   listed separately (write them with --events-output to keep them).

Output CSV columns (first five are what build_dataset.py consumes; the rest
are outcome metadata that must NOT become model inputs):

    kol, symbol, timestamp, side, entry_price, leverage, margin_mode,
    exit_price, close_timestamp, pnl, position_size, holding_time_seconds

- Timestamps are naive wall clock; downstream (TradeAligner) treats them
  as UTC, matching Binance exchange time. Event timestamps carry no year;
  the year is inferred from --year and month rollover.
- A trailing block truncated at the close timestamp still yields a sample
  (behavior label = open side only needs the entry); outcome metadata is
  left empty and a warning is printed.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

# Make the repo root importable when run as `python scripts/parse_kol_trades.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
SKIP_TOKENS = {"image"}

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

_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
_LEVERAGE_RE = re.compile(r"^\d+x$")

# Event-stream format: "09-04, 23:09:09" / "Open Long" / detail sentence.
_EVENT_TS_RE = re.compile(r"^(\d{2})-(\d{2}), (\d{2}:\d{2}:\d{2})$")
_EVENT_ACTION_RE = re.compile(r"^(Open|Close) (Long|Short)$")
_EVENT_DETAIL_RE = re.compile(
    r"^(Open|Close) a (Long|Short) position of (\S+) Perpetual at a price of "
    r"([0-9][0-9,.]*) USDT, amount of ([0-9][0-9,.]*) ([A-Z0-9]+) for a total value of "
    r"([0-9][0-9,.]*) USDT(?:\.|,)(?: Realized PNL is (-?[0-9][0-9,.]*) USDT\.)?$"
)


class ParseError(ValueError):
    """Raised when a block does not match the expected format."""


def _parse_number(text: str) -> float:
    return float(text.replace(",", ""))


def _parse_timestamp(text: str) -> datetime:
    return datetime.strptime(text, TIMESTAMP_FORMAT)


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
        value, _, unit = self.take(f"value of {what}").partition(" ")
        return _parse_number(value)

    def take_quantity(self, what: str) -> tuple[float, str]:
        self.expect(what)
        value, _, unit = self.take(f"value of {what}").partition(" ")
        return _parse_number(value), unit


def detect_format(text: str) -> str:
    """'event' if the first content line looks like an event timestamp."""
    for line in text.splitlines():
        line = line.strip()
        if not line or line in SKIP_TOKENS:
            continue
        return "event" if _EVENT_TS_RE.match(line) else "position"
    return "position"


def parse_event_stream(
    text: str, kol: str, year: int = 2026
) -> tuple[list[dict], list[dict], list[str]]:
    """Parse the per-action signal feed; opens become samples, closes are
    listed separately (FIFO matching is portfolio accounting, not needed
    for behavior labels)."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    trades: list[dict] = []
    closes: list[dict] = []
    warnings: list[str] = []

    previous_month: int | None = None
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
        if previous_month is not None and month_int > previous_month:
            # File is newest-first; month increasing means crossing into
            # the previous year (e.g. ... 01-xx then 12-xx).
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

        if detail_action == "Open":
            trades.append(
                {
                    "kol": kol,
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "side": side,
                    "entry_price": price,
                    "leverage": None,
                    "margin_mode": None,
                    "exit_price": None,
                    "close_timestamp": None,
                    "pnl": None,
                    "position_size": amount,
                    "holding_time_seconds": None,
                }
            )
        else:
            closes.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "side": side,
                    "price": price,
                    "amount": amount,
                    "total_value": total_value,
                    "pnl": pnl,
                }
            )
        i += 3

    if i < len(lines):
        raise ParseError(f"line {i + 1}: trailing unparsed lines: {lines[i:]!r}")

    return trades, closes, warnings


def parse_positions(text: str, kol: str) -> tuple[list[dict], list[str]]:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line and line not in SKIP_TOKENS]

    trades: list[dict] = []
    warnings: list[str] = []
    i = 0
    while i < len(lines):
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
        if not _TIMESTAMP_RE.match(opened_at):
            raise ParseError(f"line {reader.i}: bad open timestamp {opened_at!r}")
        entry_price = reader.take_price("Entry Price")
        max_oi, _ = reader.take_quantity("Max. Open Interest")
        pnl = reader.take_price("Closing PNL")

        # Outcome tail; truncated dumps may end anywhere after the PNL line.
        exit_price = None
        closed_at = None
        closed_vol = None
        if reader.i < len(lines) and lines[reader.i] == "Closed":
            reader.i += 1
            closed_at = reader.take("close timestamp")
            if not _TIMESTAMP_RE.match(closed_at):
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

        opened_dt = _parse_timestamp(opened_at)
        closed_dt = _parse_timestamp(closed_at) if closed_at else None
        trades.append(
            {
                "kol": kol,
                "symbol": symbol,
                "timestamp": opened_at,
                "side": side,
                "entry_price": entry_price,
                "leverage": int(leverage[:-1]),
                "margin_mode": margin_mode,
                "exit_price": exit_price,
                "close_timestamp": closed_at,
                "pnl": pnl,
                "position_size": max_oi,
                "holding_time_seconds": (
                    (closed_dt - opened_dt).total_seconds() if closed_dt else None
                ),
            }
        )
        i = reader.i

    return trades, warnings


def _print_stats(trades: list[dict]) -> None:
    wins = [t for t in trades if t["pnl"] is not None and t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] is not None and t["pnl"] <= 0]
    sides: dict[str, int] = {}
    symbols: dict[str, int] = {}
    for trade in trades:
        sides[trade["side"]] = sides.get(trade["side"], 0) + 1
        symbols[trade["symbol"]] = symbols.get(trade["symbol"], 0) + 1

    print(f"trades: {len(trades)}")
    if any(t["close_timestamp"] for t in trades):
        print(f"  complete (has close data): {sum(1 for t in trades if t['close_timestamp'])}")
    if wins or losses:
        total_pnl = sum(t["pnl"] for t in trades if t["pnl"] is not None)
        print(f"  wins/losses: {len(wins)}/{len(losses)}")
        print(f"  total PNL: {total_pnl:,.2f} USDT")
    print(f"  sides: {sides}")
    print(f"  symbols ({len(symbols)}): {dict(sorted(symbols.items(), key=lambda kv: -kv[1]))}")


EVENT_FIELDS = ["timestamp", "symbol", "side", "price", "amount", "total_value", "pnl"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="raw history text dump")
    parser.add_argument("--output", required=True, help="normalized trades CSV path")
    parser.add_argument("--kol", default="aoying_capital", help="KOL identifier (default: aoying_capital)")
    parser.add_argument(
        "--format",
        choices=["auto", "position", "event"],
        default="auto",
        help="input format (default: auto-detect)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2026,
        help="year for event-stream timestamps, which carry no year (default: 2026)",
    )
    parser.add_argument(
        "--events-output",
        help="if given and input is an event stream, write close events here",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = Path(args.input).read_text(encoding="utf-8")
    fmt = args.format if args.format != "auto" else detect_format(text)

    if fmt == "event":
        trades, close_events, warnings = parse_event_stream(text, kol=args.kol, year=args.year)
        if args.events_output:
            events_path = Path(args.events_output)
            events_path.parent.mkdir(parents=True, exist_ok=True)
            with events_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
                writer.writeheader()
                for event in close_events:
                    writer.writerow(event)
            print(f"wrote {events_path} ({len(close_events)} close events)")
        elif close_events:
            print(f"[INFO] {len(close_events)} close events dropped (use --events-output to keep them)")
    else:
        trades, warnings = parse_positions(text, kol=args.kol)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for trade in trades:
            writer.writerow({field: trade[field] for field in FIELDS})

    for warning in warnings:
        print(f"[WARN] {warning}")
    _print_stats(trades)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
