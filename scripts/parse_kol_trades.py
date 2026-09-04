"""Parse KOL position-history text dumps into the normalized trades CSV.

Input format (copy-trading UI dump), one position per block:

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

Output CSV columns (first five are what build_dataset.py consumes; the rest
are outcome metadata that must NOT become model inputs):

    kol, symbol, timestamp, side, entry_price, leverage, margin_mode,
    exit_price, close_timestamp, pnl, position_size, holding_time_seconds

- Timestamps are naive wall clock; downstream (TradeAligner) treats them
  as UTC, matching Binance exchange time.
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
    total_pnl = sum(t["pnl"] for t in trades if t["pnl"] is not None)
    sides: dict[str, int] = {}
    symbols: dict[str, int] = {}
    for trade in trades:
        sides[trade["side"]] = sides.get(trade["side"], 0) + 1
        symbols[trade["symbol"]] = symbols.get(trade["symbol"], 0) + 1

    print(f"trades: {len(trades)}")
    print(f"  complete (has close data): {sum(1 for t in trades if t['close_timestamp'])}")
    print(f"  wins/losses: {len(wins)}/{len(losses)}")
    print(f"  total PNL: {total_pnl:,.2f} USDT")
    print(f"  sides: {sides}")
    print(f"  symbols ({len(symbols)}): {dict(sorted(symbols.items(), key=lambda kv: -kv[1]))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="raw position-history text dump")
    parser.add_argument("--output", required=True, help="normalized trades CSV path")
    parser.add_argument("--kol", default="aoying_capital", help="KOL identifier (default: aoying_capital)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = Path(args.input).read_text(encoding="utf-8")
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
