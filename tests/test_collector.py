from __future__ import annotations

from datetime import timedelta

import pytest

from chan_test_utils import START, make_storage, write_zigzag_history
from src.chan.chan_adapter import SUPPORTED_TIMEFRAMES
from src.chan.causal_chan import CausalChanEngine
from src.chan.collector import collect_chan_features, collect_chan_states
from src.chan.feature_encoder import ChanFeatureEncoder


@pytest.fixture
def storage(tmp_path):
    storage = make_storage(tmp_path)
    write_zigzag_history(storage)
    return storage


@pytest.fixture
def engine(storage) -> CausalChanEngine:
    return CausalChanEngine(storage=storage)


def test_collect_chan_states_all_supported_timeframes(engine):
    as_of = START + timedelta(hours=41)  # every timeframe has >= 40 bars by then
    states = collect_chan_states(engine, "BTCUSDT", as_of, SUPPORTED_TIMEFRAMES)

    assert set(states) == set(SUPPORTED_TIMEFRAMES)
    for timeframe, state in states.items():
        assert state["supported"] is True
        assert state["timeframe"] == timeframe
        assert state["bi_count"] >= 1


def test_collect_chan_states_unsupported_timeframe_is_masked(engine):
    states = collect_chan_states(engine, "BTCUSDT", START + timedelta(hours=41), ("4h",))
    assert states["4h"]["supported"] is False
    assert states["4h"]["bi_count"] == 0


def test_collect_chan_states_are_point_in_time_snapshots(engine):
    as_of_early = START + timedelta(hours=3)
    as_of_late = START + timedelta(hours=41)

    early = collect_chan_states(engine, "BTCUSDT", as_of_early, ("5m",))
    late = collect_chan_states(engine, "BTCUSDT", as_of_late, ("5m",))

    # advancing the engine must not rewrite the earlier snapshot
    assert early["5m"]["last_bar_close_time"] != late["5m"]["last_bar_close_time"]
    early_again = collect_chan_states(engine, "BTCUSDT", as_of_early, ("5m",))
    assert early_again == early


def test_collect_chan_features_prefixes_and_order(engine):
    encoder = ChanFeatureEncoder()
    features = collect_chan_features(engine, "BTCUSDT", START + timedelta(hours=41), ("1m", "5m"), encoder)

    assert set(features) == {f"{tf}__{key}" for tf in ("1m", "5m") for key in encoder.FEATURE_KEYS}
    # deterministic key order: timeframe blocks sorted, feature keys in fixed order
    assert list(features) == [
        f"{tf}__{key}" for tf in sorted(("1m", "5m")) for key in encoder.FEATURE_KEYS
    ]
    assert features["5m__bi_count"] >= 1.0
