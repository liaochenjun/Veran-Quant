from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


def _as_naive_utc(dt: datetime) -> datetime:
    """Strip tz info, converting aware datetimes to UTC wall time first."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


@dataclass(slots=True)
class DuckDBStorage:
    """Parquet-backed storage with DuckDB query access.

    Klines are partitioned by month under ``<root>/<SYMBOL>/<timeframe>/<YYYY-MM>.parquet``
    so that incremental writes only rewrite the affected month instead of the full history.
    """

    root_dir: Path
    database_path: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def _partition_dir(self, symbol: str, timeframe: str) -> Path:
        symbol_dir = self.root_dir / symbol.upper() / timeframe
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir

    def write_klines(self, symbol: str, timeframe: str, rows: list[dict]) -> None:
        if not rows:
            return

        incoming = pd.DataFrame(rows)
        partition_dir = self._partition_dir(symbol=symbol, timeframe=timeframe)
        # Bucket by UTC month; to_period drops tz silently, so make UTC
        # wall time explicit first.
        month_keys = incoming["open_time"]
        if month_keys.dt.tz is not None:
            month_keys = month_keys.dt.tz_convert("UTC").dt.tz_localize(None)
        for month, group in incoming.groupby(month_keys.dt.to_period("M")):
            path = partition_dir / f"{month}.parquet"
            if path.exists():
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, group], ignore_index=True)
            else:
                combined = group

            combined = combined.drop_duplicates(subset=["open_time"], keep="last")
            combined = combined.sort_values("open_time").reset_index(drop=True)
            combined.to_parquet(path, index=False)

    def read_klines(
        self,
        symbol: str,
        timeframe: str,
        end_before: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        partition_dir = self._partition_dir(symbol=symbol, timeframe=timeframe)
        paths = sorted(partition_dir.glob("*.parquet"))
        if not paths:
            return pd.DataFrame()

        where_clause = ""
        params: list[object] = [[str(p) for p in paths]]
        if end_before is not None:
            # Compare in naive-UTC space so both tz-aware and tz-naive parquet
            # columns are handled identically (DuckDB's binder rejects mixed
            # TIMESTAMP/TIMESTAMPTZ comparisons).
            where_clause = " WHERE CAST(close_time AS TIMESTAMP) < ?"
            params.append(_as_naive_utc(end_before))

        limit_clause = ""
        if limit is not None:
            limit_clause = f" LIMIT {int(limit)}"

        query = (
            "SET TimeZone='UTC';"  # make TIMESTAMPTZ -> TIMESTAMP casts UTC-based
            "SELECT * FROM read_parquet(?)"
            f"{where_clause}"
            " ORDER BY close_time DESC"
            f"{limit_clause}"
        )

        # Reads never touch the DuckDB file itself; an in-memory connection
        # avoids creating the database as a side effect.
        with duckdb.connect() as con:
            out = con.execute(query, params).df()

        return out.sort_values("close_time").reset_index(drop=True)
