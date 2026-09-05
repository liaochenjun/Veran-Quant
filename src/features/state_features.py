"""State feature assembly: STATE_AT_T -> flat numeric feature dict.

This is the ONLY way model inputs are built:

- ``from_sample`` — training/validation/test path (sample stores the frozen
  T-visible data built by the point-in-time pipeline);
- ``from_components`` — live/future prediction path (market state + chan
  snapshots collected at T through the same causal pipeline).

Both paths produce the same deterministic feature keys, so a model trained
on samples can consume live features unchanged. Features contain INPUT
only: no KOL action, no outcome fields.

Key groups (sorted, deterministic order):
- ``market__<tf>__...``   per-timeframe market/geometry features
- ``chan__<tf>__...``     per-timeframe chan structure features
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.chan.chan_adapter import SUPPORTED_TIMEFRAMES
from src.chan.chan_state import ChanState
from src.chan.feature_encoder import ChanFeatureEncoder
from src.dataset.behavior_dataset import BehaviorSample
from src.market.features import MarketFeatureBuilder
from src.market.geometry import GeometryFeatureExtractor
from src.market.point_in_time import MarketState

# Bump whenever the feature key set or encoding changes; recorded in every
# model artifact so predictions stay traceable to their feature definition.
FEATURE_VERSION = "state-features-v2"

_TREND_VALUES = ("uptrend", "downtrend", "flat")

# Traders one-hot encoded into the input (P(Action | State, Trader)).
# New traders must be added here (and FEATURE_VERSION bumped).
KNOWN_TRADERS = ("aoying_capital", "liuyuan", "xijiuye")


def _flatten(prefix: str, value: object) -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(_flatten(f"{prefix}__{key}", item))
    elif isinstance(value, str):
        # trend-style categorical -> one-hot
        for hot in _TREND_VALUES:
            out[f"{prefix}__{hot}"] = 1.0 if value == hot else 0.0
    elif isinstance(value, bool):
        out[prefix] = float(value)
    elif value is None:
        out[prefix] = float("nan")
    else:
        out[prefix] = float(value)
    return out


class StateFeatureAssembler:
    """Assemble the pure-INPUT feature vector for one point-in-time moment."""

    def __init__(
        self,
        market_builder: MarketFeatureBuilder | None = None,
        chan_encoder: ChanFeatureEncoder | None = None,
        chan_timeframes: tuple[str, ...] = SUPPORTED_TIMEFRAMES,
        trader_ids: tuple[str, ...] = KNOWN_TRADERS,
    ) -> None:
        self.market_builder = market_builder or MarketFeatureBuilder(
            geometry_extractor=GeometryFeatureExtractor(lookback=50)
        )
        self.chan_encoder = chan_encoder or ChanFeatureEncoder()
        self.chan_timeframes = chan_timeframes
        self.trader_ids = trader_ids

    def _trader_features(self, trader_id: str | None) -> dict[str, float]:
        """One-hot trader identity: P(Action | State, Trader)."""
        features: dict[str, float] = {}
        for trader in self.trader_ids:
            features[f"trader__{trader}"] = 1.0 if trader_id == trader else 0.0
        known = trader_id in self.trader_ids
        features["trader__other"] = 1.0 if (trader_id is not None and not known) else 0.0
        return features

    def from_components(
        self,
        market_state: MarketState,
        chan_states: dict[str, dict],
        trader_id: str | None = None,
    ) -> dict[str, float]:
        """Live path: assemble from a point-in-time MarketState + chan snapshots."""
        features: dict[str, float] = {}
        market_features = self.market_builder.build(market_state)
        for timeframe in sorted(market_features):
            features.update(_flatten(f"market__{timeframe}", market_features[timeframe]))
        encoded = self.chan_encoder.encode_multi(
            {tf: ChanState.from_dict(state) for tf, state in chan_states.items()}
        )
        features.update({f"chan__{key}": value for key, value in encoded.items()})
        features.update(self._trader_features(trader_id))
        return dict(sorted(features.items()))

    def from_sample(self, sample: BehaviorSample) -> dict[str, float]:
        """Training/validation/test path: assemble from a frozen sample."""
        frames: dict[str, pd.DataFrame] = {}
        for timeframe, records in sample.market_state.items():
            frame = pd.DataFrame(records)
            # JSON-serialized records carry string timestamps; restore the
            # datetime columns the feature builders rely on.
            for column in ("open_time", "close_time"):
                if column in frame.columns:
                    frame[column] = pd.to_datetime(frame[column], utc=True)
            frames[timeframe] = frame
        market_state = MarketState(
            symbol=sample.symbol,
            as_of_timestamp=datetime.fromisoformat(sample.timestamp),
            frames=frames,
        )
        return self.from_components(market_state, sample.chan_states, trader_id=sample.kol)
