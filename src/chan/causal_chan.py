"""Causal (point-in-time-safe) Chan engine built on chan.py.

Guarantees:
- a state queried at ``as_of_timestamp = T`` is computed exclusively from
  bars whose ``close_time < T`` (bars are fed in close-time order and the
  storage layer already enforces the strict filter);
- feeding bars after T never changes a snapshot already returned for T
  (snapshots are plain frozen data, not live views into chan.py);
- bars are fed incrementally (chan.py ``trigger_load``), so the T-state is
  exactly what a strategy running at T would have computed — future bars
  can never rewrite it.

Backward queries (as_of earlier than the most recently fed bar) are handled
by rebuilding the adapter from the bars visible at that earlier moment.
Forward queries only feed the delta since the last feed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from src.chan.chan_adapter import SUPPORTED_TIMEFRAMES, ChanPyAdapter
from src.chan.chan_engine import ChanEngine
from src.chan.chan_state import ChanState
from src.data.storage import DuckDBStorage
from src.market.point_in_time import _as_aware_utc


def _close_times_naive_utc(frame: pd.DataFrame) -> pd.Series:
    """Normalize close_time to naive-UTC wall clock for comparisons."""
    close_times = frame["close_time"]
    if close_times.dt.tz is not None:
        return close_times.dt.tz_convert("UTC").dt.tz_localize(None)
    return close_times


@dataclass(slots=True)
class CausalChanEngine(ChanEngine):
    """Point-in-time Chan engine over DuckDBStorage data.

    One chan.py adapter is kept per (symbol, timeframe) and advanced only
    forward in bar time; snapshots are frozen ChanState dataclasses.
    """

    storage: DuckDBStorage
    timeframes: tuple[str, ...] = SUPPORTED_TIMEFRAMES
    config: dict | None = None

    _adapters: dict[tuple[str, str], ChanPyAdapter] = field(default_factory=dict)
    _last_bar_close: dict[tuple[str, str], datetime] = field(default_factory=dict)

    def get_state(
        self,
        symbol: str,
        timeframe: str,
        as_of_timestamp: datetime,
        market_state: object | None = None,
    ) -> dict:
        """Chan state at as_of_timestamp as a stable dict (ChanState.to_dict)."""
        as_of = _as_aware_utc(as_of_timestamp)

        if timeframe not in SUPPORTED_TIMEFRAMES:
            # chan.py has no 4h (or other) level; report honestly, never fake.
            return ChanState.empty(symbol, timeframe, as_of, supported=False).to_dict()

        # Bars strictly closed before as_of (storage enforces close_time < as_of).
        frame = self.storage.read_klines(symbol=symbol, timeframe=timeframe, end_before=as_of)
        if frame.empty:
            return ChanState.empty(symbol, timeframe, as_of).to_dict()

        close_times = _close_times_naive_utc(frame)
        frame_last = close_times.max().to_pydatetime()

        key = (symbol, timeframe)
        last_fed = self._last_bar_close.get(key)

        if last_fed is None or frame_last < last_fed:
            # First query or backward query: rebuild from everything visible.
            adapter = self._build_adapter(timeframe, frame)
            self._adapters[key] = adapter
        elif frame_last > last_fed:
            # Forward query: feed only the delta.
            adapter = self._adapters[key]
            delta = frame[close_times > last_fed]
            self._feed_rows(adapter, delta)
        else:
            # frame_last == last_fed: nothing new to feed.
            adapter = self._adapters[key]

        self._last_bar_close[key] = frame_last
        return adapter.snapshot(symbol=symbol, timeframe=timeframe, as_of_timestamp=as_of).to_dict()

    def _build_adapter(self, timeframe: str, frame: pd.DataFrame) -> ChanPyAdapter:
        adapter = ChanPyAdapter(timeframe=timeframe, config=self.config)
        self._feed_rows(adapter, frame)
        return adapter

    @staticmethod
    def _feed_rows(adapter: ChanPyAdapter, frame: pd.DataFrame) -> None:
        for row in frame.itertuples(index=False):
            adapter.feed_bar(
                bar_close_time=row.close_time.to_pydatetime(),
                open_=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
            )
