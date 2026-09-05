"""Parse every raw KOL file under data/raw/kols/ into one combined dataset.

Conventions:
- ``<kol>.txt``            position-history dump (English or Chinese blocks)
- ``<kol>_events.txt``     per-action event stream (English or Chinese)

Outputs:
- data/processed/trades_all_kols.csv   combined OPEN behavior trades
- data/processed/events_all_kols.csv   combined CLOSE events (outcome side)

Impact-factor header lines (影响因子) are tolerated and skipped at parse
time; they are NOT recorded, NOT output, and never enter any model input.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Make the repo root importable when run as `python scripts/parse_all_kols.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.kol.normalizer import (  # noqa: E402
    EVENT_FIELDS,
    FIELDS,
    outcomes_to_rows,
    parse_event_stream_auto,
    parse_position_dump_auto,
    to_trades_rows,
)

KOLS_DIR = Path("data/raw/kols")
OUT_DIR = Path("data/processed")


def main() -> None:
    trades: list[dict] = []
    events: list[dict] = []
    warnings: list[str] = []

    for path in sorted(KOLS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        is_events = path.stem.endswith("_events")
        kol = path.stem.removesuffix("_events")
        file_warnings: list[str]

        if is_events:
            parsed_events, outcomes, file_warnings = parse_event_stream_auto(text, trader_id=kol)
            trades.extend(to_trades_rows(parsed_events))
            events.extend(outcomes_to_rows(outcomes))
        else:
            parsed_events, outcomes, file_warnings = parse_position_dump_auto(text, trader_id=kol)
            trades.extend(to_trades_rows(parsed_events, outcomes))
            events.extend(outcomes_to_rows(outcomes))

        for warning in file_warnings:
            warnings.append(f"{path.name}: {warning}")
        kol_trades = len([t for t in trades if t["kol"] == kol])
        print(f"{path.name}: trades={kol_trades}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "trades_all_kols.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for trade in trades:
            writer.writerow({field: trade[field] for field in FIELDS})
    with (OUT_DIR / "events_all_kols.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        for event in events:
            writer.writerow({field: event.get(field) for field in EVENT_FIELDS})

    for warning in warnings:
        print(f"[WARN] {warning}")
    print(f"\ntotal trades: {len(trades)} | total close events: {len(events)}")
    print(f"wrote {OUT_DIR / 'trades_all_kols.csv'} and {OUT_DIR / 'events_all_kols.csv'}")


if __name__ == "__main__":
    main()
