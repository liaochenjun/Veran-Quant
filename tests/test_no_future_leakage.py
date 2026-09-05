"""Future-leakage tests: time capsule, leakage detector, replay discipline."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import pytest

from chan_test_utils import START, make_storage, zigzag_rows
from src.alignment.trade_aligner import KOLTrade, TradeAligner
from src.chan.causal_chan import CausalChanEngine
from src.data.leakage_detector import (
    LeakageReport,
    LeakageViolation,
    assert_no_leakage,
    scan_samples,
)
from src.dataset.behavior_dataset import BehaviorDataset, BehaviorSample
from src.market.geometry import GeometryFeatureExtractor
from src.market.point_in_time import PointInTimeMarketState
from src.models.behavior_model import BehaviorModel
from src.replay.causal_replay import CausalReplay, ReplayLeakageError

AS_OF = START + timedelta(hours=4)  # after all 40 5m zigzag bars


def _build_sample_dict(tmp_path, symbol: str = "BTCUSDT") -> dict:
    storage = make_storage(tmp_path)
    storage.write_klines(symbol, "5m", zigzag_rows(n=40))
    aligner = TradeAligner(point_in_time=PointInTimeMarketState(storage=storage))
    chan = CausalChanEngine(storage=storage)
    geometry = GeometryFeatureExtractor(lookback=10)

    trade = KOLTrade(kol="k", symbol=symbol, timestamp=AS_OF, side="LONG", entry_price=100.0)
    dataset = BehaviorDataset.from_trades(
        [trade], aligner, chan, geometry, geometry_timeframe="5m", chan_timeframes=("5m",)
    )
    return storage, asdict(dataset.samples[0])


def _build_samples(tmp_path, n: int = 2) -> tuple[list[BehaviorSample], object]:
    storage = make_storage(tmp_path)
    storage.write_klines("BTCUSDT", "5m", zigzag_rows(n=40))
    aligner = TradeAligner(point_in_time=PointInTimeMarketState(storage=storage))
    chan = CausalChanEngine(storage=storage)
    geometry = GeometryFeatureExtractor(lookback=10)

    trades = [
        KOLTrade(kol="k", symbol="BTCUSDT", timestamp=AS_OF, side="LONG", entry_price=100.0),
        KOLTrade(kol="k", symbol="BTCUSDT", timestamp=AS_OF + timedelta(hours=1), side="SHORT", entry_price=101.0),
    ]
    dataset = BehaviorDataset.from_trades(
        trades[:n], aligner, chan, geometry, geometry_timeframe="5m", chan_timeframes=("5m",)
    )
    return dataset.samples, storage


# ---------------------------------------------------------------------------
# Time capsule: future bars must not change a past state, field by field
# ---------------------------------------------------------------------------


def test_time_capsule_full_pipeline_state_unchanged_by_future_bars(tmp_path):
    storage, frozen_before = _build_sample_dict(tmp_path)

    # A large amount of future data arrives in the database.
    storage.write_klines("BTCUSDT", "5m", zigzag_rows(n=40, start=START + timedelta(hours=4)))

    _, frozen_after = _build_sample_dict(tmp_path)

    before_json = json.dumps(frozen_before, default=str, sort_keys=True)
    after_json = json.dumps(frozen_after, default=str, sort_keys=True)
    assert before_json == after_json


def test_time_capsule_future_price_tampering_does_not_change_past_state(tmp_path):
    # Spec test 2: future prices multiplied by +100000% must not affect T.
    storage, frozen_before = _build_sample_dict(tmp_path)

    future_rows = zigzag_rows(n=40, start=START + timedelta(hours=4))
    for row in future_rows:
        row["open"] *= 1_000_000
        row["high"] *= 1_000_000
        row["low"] *= 1_000_000
        row["close"] *= 1_000_000
    storage.write_klines("BTCUSDT", "5m", future_rows)

    _, frozen_after = _build_sample_dict(tmp_path)

    before_json = json.dumps(frozen_before, default=str, sort_keys=True)
    after_json = json.dumps(frozen_after, default=str, sort_keys=True)
    assert before_json == after_json


# ---------------------------------------------------------------------------
# Leakage detector
# ---------------------------------------------------------------------------


def test_leakage_detector_passes_clean_sample(tmp_path):
    _, sample = _build_sample_dict(tmp_path)
    report = scan_samples([sample])
    assert report.ok, report.to_dict()
    assert_no_leakage([sample])  # must not raise


def test_leakage_detector_fails_on_future_kline(tmp_path):
    _, sample = _build_sample_dict(tmp_path)
    # A kline that closes at/after the action timestamp appears in the input.
    sample["market_state"]["5m"][-1]["close_time"] = "2026-08-01T06:00:00+00:00"

    report = scan_samples([sample])
    assert not report.ok
    assert not report.checks[1].passed  # kline_close_before_action
    with pytest.raises(LeakageViolation):
        assert_no_leakage([sample])


def test_leakage_detector_fails_on_forbidden_outcome_key(tmp_path):
    _, sample = _build_sample_dict(tmp_path)
    sample["geometry_features"]["pnl"] = 123.0  # outcome smuggled into input

    report = scan_samples([sample])
    assert not report.ok
    with pytest.raises(LeakageViolation):
        assert_no_leakage([sample])


def test_leakage_detector_fails_on_chan_future_bar(tmp_path):
    _, sample = _build_sample_dict(tmp_path)
    # chan snapshot whose last bar closes after the action timestamp
    sample["chan_states"]["5m"]["last_bar_close_time"] = "2026-08-01T06:00:00+00:00"

    report = scan_samples([sample])
    assert not report.ok
    assert not report.checks[2].passed  # chan_state_before_action


def test_leakage_detector_reports_scaler_discipline():
    report = scan_samples([])
    scaler_check = report.checks[-1]
    assert scaler_check.name == "scaler_fit_discipline"
    assert scaler_check.passed  # no scaler configured -> nothing fits on val/test


# ---------------------------------------------------------------------------
# Behavior / Outcome architectural separation
# ---------------------------------------------------------------------------


def test_behavior_dataset_module_never_imports_outcome_dataset():
    # Enforced in a clean interpreter so in-process imports cannot mask it.
    code = (
        "import sys\n"
        "import src.dataset.behavior_dataset\n"
        "assert 'src.dataset.outcome_dataset' not in sys.modules, "
        "'BehaviorDataset must never import the Outcome dataset'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Causal replay: OBSERVE -> PREDICT -> RECORD -> ADVANCE -> EVALUATE
# ---------------------------------------------------------------------------


class RecordingModel(BehaviorModel):
    def __init__(self):
        self.events: list[tuple[str, str]] = []

    def fit(self, samples):
        return None

    def predict(self, samples):
        for sample in samples:
            self.events.append(("predict", sample.timestamp))
        return ["LONG"] * len(samples)


def test_replay_evaluates_only_after_prediction_recorded(tmp_path):
    samples, _ = _build_samples(tmp_path, n=2)

    order: list[tuple[str, str]] = []

    class OrderRecordingModel(RecordingModel):
        def predict(self, samples):
            for sample in samples:
                order.append(("predict", sample.timestamp))
            return ["LONG"] * len(samples)

    def outcome_lookup(trader_id, symbol, action_ts):
        order.append(("outcome", action_ts))
        return {"pnl": 42.0}

    records = CausalReplay(model=OrderRecordingModel()).run(samples, outcome_lookup=outcome_lookup)

    assert len(records) == 2
    assert records[0].predicted_side == "LONG"
    assert records[0].actual_side == "LONG"
    assert records[0].outcome == {"pnl": 42.0}  # attached after record
    # strict interleaving: predict(ts1) -> outcome(ts1) -> predict(ts2) -> outcome(ts2)
    assert order == [
        ("predict", samples[0].timestamp),
        ("outcome", samples[0].timestamp),
        ("predict", samples[1].timestamp),
        ("outcome", samples[1].timestamp),
    ]


def test_replay_refuses_leaky_dataset(tmp_path):
    samples, _ = _build_samples(tmp_path, n=1)
    # sneak future PNL into the geometry input
    samples[0].geometry_features["pnl"] = 999.0

    with pytest.raises(ReplayLeakageError):
        CausalReplay(model=RecordingModel()).run(samples)


def test_replay_rejects_out_of_order_samples(tmp_path):
    samples, _ = _build_samples(tmp_path, n=2)
    swapped = [samples[1], samples[0]]

    with pytest.raises(ValueError, match="chronological"):
        CausalReplay(model=RecordingModel()).run(swapped)
