"""Multi-timeframe chan collection around a single point-in-time moment.

These helpers freeze the chan state (or the flattened feature vector) of
every requested timeframe at ``as_of_timestamp``.  Each state is a plain
``ChanState`` dict snapshot — later bars can never rewrite it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from src.chan.chan_engine import ChanEngine
from src.chan.chan_state import ChanState
from src.chan.feature_encoder import ChanFeatureEncoder


def collect_chan_states(
    chan_engine: ChanEngine,
    symbol: str,
    as_of_timestamp: datetime,
    timeframes: Iterable[str],
) -> dict[str, dict]:
    """Freeze the chan state of every timeframe at as_of_timestamp."""
    return {
        timeframe: chan_engine.get_state(
            symbol=symbol,
            timeframe=timeframe,
            as_of_timestamp=as_of_timestamp,
        )
        for timeframe in timeframes
    }


def collect_chan_features(
    chan_engine: ChanEngine,
    symbol: str,
    as_of_timestamp: datetime,
    timeframes: Iterable[str],
    encoder: ChanFeatureEncoder | None = None,
) -> dict[str, float]:
    """Flatten multi-timeframe chan states into prefixed numeric features."""
    encoder = encoder or ChanFeatureEncoder()
    states = {
        timeframe: ChanState.from_dict(state_dict)
        for timeframe, state_dict in collect_chan_states(
            chan_engine, symbol, as_of_timestamp, timeframes
        ).items()
    }
    return encoder.encode_multi(states)
