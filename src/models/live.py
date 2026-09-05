"""Live / future prediction interface.

The signature takes ONLY (symbol, timestamp) plus causal pipeline
components — it structurally cannot receive the real KOL action, because
at prediction time the future action does not exist.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from src.chan.chan_engine import ChanEngine
from src.chan.collector import collect_chan_states
from src.features.state_features import StateFeatureAssembler
from src.market.point_in_time import PointInTimeMarketState
from src.models.behavior_model import BehaviorModel
from src.models.prediction import Prediction


def predict_kol_behavior(
    symbol: str,
    timestamp: datetime,
    point_in_time: PointInTimeMarketState,
    chan_engine: ChanEngine,
    assembler: StateFeatureAssembler,
    model: BehaviorModel,
    model_version: str = "unknown",
    trader_id: Optional[str] = None,
    prediction_log: Optional[object] = None,
) -> dict:
    """Predict P(KOL action | STATE_AT_T, trader) for a live moment.

    Internal flow:
        T -> Point-in-Time Market State -> Causal Chan snapshots
          -> StateFeatureAssembler (+ trader identity) -> model.predict_proba

    No actual_kol_action parameter exists — feedback comes later through
    record_actual_kol_action().
    """
    market_state = point_in_time.get_market_state(symbol=symbol, as_of_timestamp=timestamp)
    chan_states = collect_chan_states(
        chan_engine, symbol=symbol, as_of_timestamp=timestamp, timeframes=assembler.chan_timeframes
    )
    features = assembler.from_components(
        market_state=market_state, chan_states=chan_states, trader_id=trader_id
    )

    prediction: Prediction = model.predict_proba([features])[0]
    entry = {
        "timestamp": timestamp.isoformat(),
        "symbol": symbol,
        "trader_id": trader_id,
        **prediction.to_dict(),
        "model_version": model_version,
    }
    if prediction_log is not None:
        prediction_log.record(entry)
    return entry
