"""CLI for KOL trading-history dumps -> normalized trades CSV.

Thin wrapper over src/data/kol/normalizer.py; the parser logic lives there.

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

   Every "Open" event becomes one behavior sample; "Close" events carry
   realized PNL and are written with --events-output (outcome data).

Output CSV columns (first five are what build_dataset.py consumes; the rest
are outcome metadata that must NOT become model inputs):

    kol, symbol, timestamp, side, entry_price, leverage, margin_mode,
    exit_price, close_timestamp, pnl, position_size, holding_time_seconds

Timestamps are naive wall clock; downstream (TradeAligner) treats them
as UTC, matching Binance exchange time.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Make the repo root importable when run as `python scripts/parse_kol_trades.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.kol.normalizer import (  # noqa: E402
    EVENT_FIELDS,
    FIELDS,
    ParseError,
    detect_format,
    outcomes_to_rows,
    parse_event_stream_auto,
    parse_position_dump_auto,
    to_trades_rows,
)


def parse_positions(text: str, kol: str) -> tuple[list[dict], list[str]]:
    """Backward-compatible wrapper: (trades rows, warnings)."""
    events, outcomes, warnings = parse_position_dump_auto(text, trader_id=kol)
    return to_trades_rows(events, outcomes), warnings


def parse_event_stream(text: str, kol: str, year: int = 2026) -> tuple[list[dict], list[dict], list[str]]:
    """Backward-compatible wrapper: (trades rows, close rows, warnings)."""
    events, outcomes, warnings = parse_event_stream_auto(text, trader_id=kol, year=year)
    return to_trades_rows(events), outcomes_to_rows(outcomes), warnings


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
