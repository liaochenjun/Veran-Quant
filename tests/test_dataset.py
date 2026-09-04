from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timedelta

import pandas as pd
import pytest

from chan_test_utils import START, make_storage, zigzag_rows
from src.alignment.trade_aligner import KOLTrade, TradeAligner
from src.chan.causal_chan import CausalChanEngine
from src.chan.chan_engine import SimpleChanEngine
from src.dataset.behavior_dataset import BehaviorDataset, BehaviorSample
from src.market.geometry import GeometryFeatureExtractor
from src.market.point_in_time import PointInTimeMarketState


def _candles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "open_time": datetime(2026, 8, 20, 13, 58),
                "open": 99.5,
                "high": 100.6,
                "low": 99.6,
                "close": 100.0,
                "volume": 10.0,
                "close_time": datetime(2026, 8, 20, 13, 58, 59),
            },
            {
                "open_time": datetime(2026, 8, 20, 13, 59),
                "open": 100.0,
                "high": 101.0,
                "low": 100.0,
                "close": 100.5,
                "volume": 12.0,
                "close_time": datetime(2026, 8, 20, 13, 59, 59),
            },
        ]
    )


class StubStorage:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.calls: list[tuple[str, str]] = []

    def read_klines(self, symbol, timeframe, end_before=None, limit=None):
        self.calls.append((symbol, timeframe))
        return self.frame


def _make_components(frame):
    storage = StubStorage(frame)
    point_in_time = PointInTimeMarketState(storage=storage, timeframes=("15m",))
    aligner = TradeAligner(point_in_time=point_in_time)
    chan_engine = SimpleChanEngine(point_in_time=point_in_time)
    return storage, aligner, chan_engine


def _make_trade(kol: str, timestamp: datetime, side: str) -> KOLTrade:
    return KOLTrade(kol=kol, symbol="BTCUSDT", timestamp=timestamp, side=side, entry_price=100.0)


def test_from_trades_sorts_chronologically():
    storage, aligner, chan_engine = _make_components(_candles())
    geometry = GeometryFeatureExtractor(lookback=10)

    late = _make_trade("B", datetime(2026, 8, 20, 15, 0), "SHORT")
    early = _make_trade("A", datetime(2026, 8,20, 14, 0), "LONG")
    dataset = BehaviorDataset.from_trades([late, early], aligner, chan_engine, geometry)

    assert [s.kol for s in dataset.samples] == ["A", "B"]
    # timestamp serialized as aware UTC ISO
    assert dataset.samples[0].timestamp == "2026-08-20T14:00:00+00:00"
    assert dataset.samples[0].side == "LONG"
    assert dataset.samples[1].side == "SHORT"


def test_from_trades_does_not_mutate_input_trades():
    storage, aligner, chan_engine = _make_components(_candles())
    geometry = GeometryFeatureExtractor(lookback=10)

    trade = _make_trade("A", datetime(2026, 8, 20, 14, 0), "LONG")
    BehaviorDataset.from_trades([trade], aligner, chan_engine, geometry)

    assert trade.timestamp == datetime(2026, 8, 20, 14, 0)
    assert trade.timestamp.tzinfo is None


def test_from_trades_chan_reuses_aligned_market_state():
    # Regression: chan state used to trigger a second full market fetch per trade.
    storage, aligner, chan_engine = _make_components(_candles())
    geometry = GeometryFeatureExtractor(lookback=10)

    trades = [
        _make_trade("A", datetime(2026, 8, 20, 14, 0), "LONG"),
        _make_trade("B", datetime(2026, 8, 20, 15, 0), "SHORT"),
    ]
    dataset = BehaviorDataset.from_trades(trades, aligner, chan_engine, geometry)

    assert len(storage.calls) == len(trades)  # one read per trade, not two
    for sample in dataset.samples:
        assert sample.chan_state["trend"] == "up"
        assert sample.chan_state["swing_high"] == 101.0
        assert sample.chan_state["swing_low"] == 99.6


def test_from_trades_geometry_features_from_aligned_frames():
    storage, aligner, chan_engine = _make_components(_candles())
    geometry = GeometryFeatureExtractor(lookback=10)

    trade = _make_trade("A", datetime(2026, 8, 20, 14, 0), "LONG")
    dataset = BehaviorDataset.from_trades([trade], aligner, chan_engine, geometry)

    features = dataset.samples[0].geometry_features
    assert features["previous_high"] == 101.0
    assert features["previous_low"] == 99.6
    assert features["local_trend_structure"] == "uptrend"


def test_from_trades_missing_geometry_timeframe_is_tolerated():
    # geometry_timeframe not present in the state frames -> empty features
    storage, aligner, chan_engine = _make_components(_candles())
    geometry = GeometryFeatureExtractor(lookback=10)

    trade = _make_trade("A", datetime(2026, 8, 20, 14, 0), "LONG")
    dataset = BehaviorDataset.from_trades([trade], aligner, chan_engine, geometry, geometry_timeframe="1h")

    assert dataset.samples[0].geometry_features == {}


def test_from_trades_multi_timeframe_chan_states(tmp_path):
    # Real storage + CausalChanEngine: chan_timeframes populates sample.chan_states
    storage = make_storage(tmp_path)
    storage.write_klines("BTCUSDT", "5m", zigzag_rows(minutes=5))
    storage.write_klines("BTCUSDT", "1m", zigzag_rows(minutes=1))

    aligner = TradeAligner(point_in_time=PointInTimeMarketState(storage=storage))
    chan_engine = CausalChanEngine(storage=storage)
    geometry = GeometryFeatureExtractor(lookback=10)

    as_of = START + timedelta(hours=3, minutes=20)  # after all 40 5m bars
    trade = KOLTrade(kol="A", symbol="BTCUSDT", timestamp=as_of, side="LONG", entry_price=100.0)
    dataset = BehaviorDataset.from_trades(
        [trade],
        aligner,
        chan_engine,
        geometry,
        geometry_timeframe="5m",
        chan_timeframes=("5m", "1m"),
    )

    sample = dataset.samples[0]
    assert set(sample.chan_states) == {"5m", "1m"}
    # geometry timeframe state is reused, not queried twice
    assert sample.chan_states["5m"] == sample.chan_state
    assert sample.chan_states["5m"]["bi_count"] >= 1
    assert sample.chan_states["1m"]["supported"] is True
    # sample stays JSON-serializable (same default=str convention as the builder script)
    json.dumps(asdict(sample), default=str)


def test_from_trades_without_chan_timeframes_keeps_old_behavior():
    # chan_timeframes defaults to None: chan_states stays empty
    storage, aligner, chan_engine = _make_components(_candles())
    geometry = GeometryFeatureExtractor(lookback=10)

    trade = _make_trade("A", datetime(2026, 8, 20, 14, 0), "LONG")
    dataset = BehaviorDataset.from_trades([trade], aligner, chan_engine, geometry)

    assert dataset.samples[0].chan_states == {}


def _sample(timestamp: str) -> BehaviorSample:
    return BehaviorSample(
        kol="k", symbol="S", timestamp=timestamp, side="LONG",
        market_state={}, chan_state={}, geometry_features={},
    )


def test_chronological_split_ratios():
    samples = [_sample(f"2026-08-20T14:0{i}:00+00:00") for i in range(10)]
    dataset = BehaviorDataset(samples=samples)
    train, val, test = dataset.chronological_split()

    assert len(train) == 7
    assert len(val) == 1
    assert len(test) == 2
    # chronological, not shuffled
    assert train[-1].timestamp == "2026-08-20T14:06:00+00:00"
    assert test[0].timestamp == "2026-08-20T14:08:00+00:00"


def test_chronological_split_rejects_invalid_ratios():
    dataset = BehaviorDataset(samples=[_sample("2026-08-20T14:00:00+00:00")])
    with pytest.raises(ValueError):
        dataset.chronological_split(train_ratio=0)
    with pytest.raises(ValueError):
        dataset.chronological_split(train_ratio=1)
    with pytest.raises(ValueError):
        dataset.chronological_split(val_ratio=0)
    with pytest.raises(ValueError):
        dataset.chronological_split(train_ratio=0.8, val_ratio=0.3)
