from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from chan_test_utils import START, bar_close, make_storage, write_zigzag_history, zigzag_rows
from src.chan.causal_chan import CausalChanEngine
from src.chan.chan_state import ChanState
from src.chan.feature_encoder import ChanFeatureEncoder
from src.data.storage import DuckDBStorage


def _utc_iso(dt: datetime) -> str:
    return dt.replace(tzinfo=timezone.utc).isoformat()


@pytest.fixture
def storage(tmp_path) -> DuckDBStorage:
    storage = make_storage(tmp_path)
    write_zigzag_history(storage)
    storage.write_klines("SPARSE", "5m", zigzag_rows(n=2))
    return storage


@pytest.fixture
def engine(storage) -> CausalChanEngine:
    return CausalChanEngine(storage=storage)


def _as_of_after_n_bars(n: int, minutes: int = 5) -> datetime:
    return bar_close(n - 1, minutes) + timedelta(seconds=1)


# 1. Basic chan computation --------------------------------------------------

def test_basic_chan_computation(engine):
    state = engine.get_state("BTCUSDT", "5m", _as_of_after_n_bars(40))

    assert state["supported"] is True
    assert state["last_bar_close_time"] == _utc_iso(bar_close(39))
    assert state["bi_count"] >= 1
    assert state["bi_direction"] in ("UP", "DOWN")
    assert state["bi_is_sure"] is not None
    assert state["bi_amplitude"] > 0
    assert state["bi_length"] >= 1
    assert state["segment_count"] >= 0
    assert state["zs_count"] >= 0
    if state["zhongshu_present"]:
        assert state["zhongshu_high"] >= state["zhongshu_low"]
        assert state["distance_to_zhongshu"] is not None


# 2. Point-in-time: state at T uses only bars strictly before T --------------

def test_point_in_time_state_uses_only_bars_before_as_of(engine):
    as_of = _as_of_after_n_bars(21)  # bars 0..20 included
    state = engine.get_state("BTCUSDT", "5m", as_of)
    assert state["last_bar_close_time"] == _utc_iso(bar_close(20))

    # a query just after bar 10's close must not see bar 10 yet (strict <)
    state_early = engine.get_state("BTCUSDT", "5m", _as_of_after_n_bars(10) - timedelta(seconds=1))
    assert state_early["last_bar_close_time"] == _utc_iso(bar_close(8))


# 3. Future leakage: feeding later bars must not rewrite a past snapshot -----

def test_future_bars_do_not_change_past_state(engine):
    as_of_t1 = _as_of_after_n_bars(21)
    state_t1 = engine.get_state("BTCUSDT", "5m", as_of_t1)

    # advance the engine far beyond T1
    engine.get_state("BTCUSDT", "5m", _as_of_after_n_bars(40))

    # re-query T1: must be identical (engine rebuilds from bars visible at T1)
    assert engine.get_state("BTCUSDT", "5m", as_of_t1) == state_t1


# 4. Incremental vs point-in-time (fresh) computation -------------------------

def test_incremental_equals_fresh_computation(storage):
    incremental = CausalChanEngine(storage=storage)
    for n in (10, 20, 30, 40):
        incremental.get_state("BTCUSDT", "5m", _as_of_after_n_bars(n))

    fresh = CausalChanEngine(storage=storage)
    final_as_of = _as_of_after_n_bars(40)
    assert incremental.get_state("BTCUSDT", "5m", final_as_of) == fresh.get_state("BTCUSDT", "5m", final_as_of)


# 5. Unclosed bar at T is never treated as completed --------------------------

def test_bar_closing_at_as_of_is_excluded(engine):
    # as_of exactly equals bar 20's close time: bar 20 must not be observed
    state = engine.get_state("BTCUSDT", "5m", bar_close(20))
    assert state["last_bar_close_time"] == _utc_iso(bar_close(19))


# 6. Multi-timeframe ----------------------------------------------------------

def test_multi_timeframe_all_supported_levels_produce_state(storage, engine):
    for timeframe, minutes in [("1m", 1), ("5m", 5), ("15m", 15), ("1h", 60)]:
        state = engine.get_state("BTCUSDT", timeframe, _as_of_after_n_bars(40, minutes))
        assert state["supported"] is True
        assert state["timeframe"] == timeframe
        assert state["bi_count"] >= 1


def test_unsupported_timeframe_is_masked_not_faked(engine):
    state = engine.get_state("BTCUSDT", "4h", _as_of_after_n_bars(40))
    assert state["supported"] is False
    assert state["bi_count"] == 0
    assert state["fractal_present"] is False
    assert state["zhongshu_present"] is False
    assert state["buy_sell_point_present"] is False


# 7. Serialization ------------------------------------------------------------

def test_chan_state_serialization_roundtrip(engine):
    state_dict = engine.get_state("BTCUSDT", "5m", _as_of_after_n_bars(40))
    # stable key order = dataclass declaration order
    assert list(state_dict.keys()) == list(ChanState.empty("X", "5m", START).to_dict().keys())

    raw = json.dumps(state_dict)
    assert isinstance(raw, str)
    assert ChanState.from_dict(json.loads(raw)).to_dict() == state_dict


# 8. Missing structures must not crash ----------------------------------------

def test_missing_structures_are_masked(engine):
    # only 2 bars: no fractal/bi/seg/zs possible yet
    state = engine.get_state("SPARSE", "5m", _as_of_after_n_bars(2))
    assert state["supported"] is True
    assert state["bi_count"] == 0
    assert state["fractal_present"] is False
    assert state["zhongshu_present"] is False
    assert state["buy_sell_point_present"] is False
    assert state["bi_direction"] is None
    assert state["bi_amplitude"] is None


def test_no_data_before_first_bar_is_empty(engine):
    state = engine.get_state("BTCUSDT", "5m", START - timedelta(seconds=1))
    assert state["supported"] is True
    assert state["bi_count"] == 0
    assert state["last_bar_close_time"] is None


# 9. Determinism ----------------------------------------------------------------

def test_determinism_same_input_same_state(storage):
    as_of = _as_of_after_n_bars(40)
    a = CausalChanEngine(storage=storage).get_state("BTCUSDT", "5m", as_of)
    b = CausalChanEngine(storage=storage).get_state("BTCUSDT", "5m", as_of)
    assert a == b


# 10. Feature encoder -----------------------------------------------------------

def test_feature_encoder_key_order_and_masks(engine):
    encoder = ChanFeatureEncoder()
    state = ChanState.from_dict(engine.get_state("BTCUSDT", "5m", _as_of_after_n_bars(40)))
    features = encoder.encode(state)

    assert list(features.keys()) == list(encoder.FEATURE_KEYS)
    assert features["bi_count"] == float(state.bi_count)
    assert features["fractal_present"] in (0.0, 1.0)
    # missing structure values are NaN, never 0
    if state.fractal_price is None:
        assert math.isnan(features["fractal_price"])
    if state.bi_direction is None:
        assert math.isnan(features["bi_direction_up"])
    # bsp one-hots
    for t in ("1", "1p", "2", "2s", "3a", "3b"):
        expected = 1.0 if t in (state.buy_sell_point_types or ()) else 0.0
        assert features[f"buy_sell_point_type_{t}"] == expected


def test_feature_encoder_multi_timeframe_prefixes(engine):
    encoder = ChanFeatureEncoder()
    states = {
        "1m": ChanState.from_dict(engine.get_state("BTCUSDT", "1m", _as_of_after_n_bars(40, 1))),
        "5m": ChanState.from_dict(engine.get_state("BTCUSDT", "5m", _as_of_after_n_bars(40))),
    }
    features = encoder.encode_multi(states)
    assert "1m__bi_count" in features
    assert "5m__bi_count" in features
    assert len(features) == 2 * len(encoder.FEATURE_KEYS)
