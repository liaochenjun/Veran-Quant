from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


@dataclass(slots=True)
class DuckDBStorage:
    """Parquet-backed storage with DuckDB query access."""

    root_dir: Path
    database_path: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def _parquet_path(self, symbol: str, timeframe: str) -> Path:
        symbol_clean = symbol.upper()
        symbol_dir = self.root_dir / symbol_clean
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir / f"{timeframe}.parquet"

    def write_klines(self, symbol: str, timeframe: str, rows: list[dict]) -> None:
        if not rows:
            return

        incoming = pd.DataFrame(rows)
        path = self._parquet_path(symbol=symbol, timeframe=timeframe)

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, incoming], ignore_index=True)
        else:
            combined = incoming

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
        path = self._parquet_path(symbol=symbol, timeframe=timeframe)
        if not path.exists():
            return pd.DataFrame()

        where_clause = ""
        params: list[object] = [str(path)]
        if end_before is not None:
            where_clause = " WHERE close_time < ?"
            params.append(end_before)

        limit_clause = ""
        if limit is not None:
            limit_clause = f" LIMIT {int(limit)}"

        query = (
            "SELECT * FROM read_parquet(?)"
            f"{where_clause}"
            " ORDER BY close_time DESC"
            f"{limit_clause}"
        )

        with duckdb.connect(str(self.database_path), read_only=False) as con:
            out = con.execute(query, params).df()

        out = out.sort_values("close_time").reset_index(drop=True)
        return out
