"""K-line coverage acceptance check for every symbol in the trades CSV.

For each (symbol, timeframe) checks:
1. data exists and reads cleanly from DuckDB/Parquet
2. row count, earliest/latest bar
3. coverage of the symbol's KOL trade time range (each trade T needs
   at least one fully closed bar with close_time < T)
4. duplicate open_times
5. time ordering
6. large gaps between consecutive bars
7. timezone consistency

Outputs a per-(symbol, tf) table plus symbol-level status:
complete / partial / missing.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

# Make the repo root importable when run as `python scripts/check_kline_coverage.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.storage import DuckDBStorage  # noqa: E402

TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h")
TRADES_CSV = Path("data/processed/trades_all_kols.csv")
TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400}


def _load_trades() -> dict[str, list[str]]:
    by_symbol: dict[str, list[str]] = defaultdict(list)
    with TRADES_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_symbol[row["symbol"]].append(row["timestamp"])
    for symbol in by_symbol:
        by_symbol[symbol].sort()
    return dict(by_symbol)


def _check_frame(df: pd.DataFrame, tf: str, trade_times: list[str]) -> tuple[str, list[str]]:
    if df.empty:
        return "MISSING", ["no kline data"]

    issues: list[str] = []
    close_times = df["close_time"]
    if close_times.dt.tz is not None:
        tz_note = "aware-UTC"
    else:
        tz_note = "naive(assumed UTC)"
        issues.append("close_time is tz-naive")

    # duplicates + ordering
    open_times = df["open_time"]
    if open_times.duplicated().any():
        issues.append(f"{int(open_times.duplicated().sum())} duplicate open_times")
    if not open_times.is_monotonic_increasing:
        issues.append("open_time not monotonically increasing")

    # gaps: > 3x timeframe between consecutive bars
    gaps = close_times.diff().dt.total_seconds().dropna()
    big_gaps = int((gaps > 3 * TF_SECONDS[tf]).sum())
    if big_gaps:
        issues.append(f"{big_gaps} gaps > 3x timeframe (max {gaps.max():.0f}s)")

    # trade-range coverage: every trade T needs at least one bar with close_time < T
    uncovered = 0
    for ts in trade_times:
        t = pd.Timestamp(ts.replace(" ", "T"))
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        if not (close_times < t).any():
            uncovered += 1
    if uncovered:
        issues.append(f"{uncovered}/{len(trade_times)} trades have no bar before T")
        return "PARTIAL", issues

    hard_issues = [i for i in issues if not i.startswith("close_time is tz-naive")]
    notes = issues + ([tz_note] if tz_note != "aware-UTC" else [])
    return ("PASS" if not hard_issues else "PARTIAL"), notes


def main() -> None:
    trades_by_symbol = _load_trades()
    storage = DuckDBStorage(root_dir=Path("data/raw"), database_path=Path("data/database/market.duckdb"))

    symbol_status: dict[str, str] = {}
    print(f"{'SYMBOL':<14} {'TF':<5} {'ROWS':>8} {'START':<20} {'END':<20} {'STATUS':<8} NOTES")
    for symbol in sorted(trades_by_symbol):
        statuses = []
        for tf in TIMEFRAMES:
            df = storage.read_klines(symbol, tf)
            status, notes = _check_frame(df, tf, trades_by_symbol[symbol])
            statuses.append(status)
            start = str(df["close_time"].min()) if not df.empty else "-"
            end = str(df["close_time"].max()) if not df.empty else "-"
            print(f"{symbol:<14} {tf:<5} {len(df):>8} {start:<20} {end:<20} {status:<8} {'; '.join(notes)}")
        if all(s == "PASS" for s in statuses):
            symbol_status[symbol] = "complete"
        elif all(s == "MISSING" for s in statuses):
            symbol_status[symbol] = "missing"
        else:
            symbol_status[symbol] = "partial"

    complete = [s for s, st in symbol_status.items() if st == "complete"]
    partial = [s for s, st in symbol_status.items() if st == "partial"]
    missing = [s for s, st in symbol_status.items() if st == "missing"]
    print(f"\nsymbols_with_trades: {len(trades_by_symbol)}")
    print(f"symbols_complete:    {len(complete)} {sorted(complete)}")
    print(f"symbols_partial:     {len(partial)} {sorted(partial)}")
    print(f"symbols_missing:     {len(missing)} {sorted(missing)}")


if __name__ == "__main__":
    main()
