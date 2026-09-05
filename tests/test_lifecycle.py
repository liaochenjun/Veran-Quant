"""Lifecycle tests: train/val/test separation, live prediction, feedback,
champion/challenger (spec section 30, tests 3-9)."""

from __future__ import annotations

import inspect
from datetime import timedelta

from chan_test_utils import START, make_storage, zigzag_rows
from src.alignment.trade_aligner import KOLTrade, TradeAligner
from src.chan.causal_chan import CausalChanEngine
from src.chan.collector import collect_chan_states
from src.dataset.behavior_dataset import BehaviorDataset, BehaviorSample
from src.features.state_features import FEATURE_VERSION, StateFeatureAssembler
from src.market.geometry import GeometryFeatureExtractor
from src.market.point_in_time import PointInTimeMarketState
from src.models.behavior_model import BaselineBehaviorModel, BehaviorModel
from src.models.feedback import ErrorMemory, PredictionLog, record_actual_kol_action
from src.models.live import predict_kol_behavior
from src.models.prediction import Prediction
from src.models.registry import (
    ChallengeDecision,
    ModelArtifact,
    ModelRegistry,
    evaluate_challenger,
)
from src.models.training import (
    TrainingConfig,
    evaluate_behavior_model,
    train_behavior_model,
)

AS_OF = START + timedelta(hours=4)


class FitCountingModel(BaselineBehaviorModel):
    def __init__(self):
        super().__init__()
        self.fit_calls = 0
        self.fit_timestamps: list[str] = []

    def fit(self, samples):
        self.fit_calls += 1
        self.fit_timestamps = [s.timestamp for s in samples]
        super().fit(samples)


class SpyPredictModel(BaselineBehaviorModel):
    def __init__(self):
        super().__init__()
        self.received: list[list[dict]] = []

    def predict_proba(self, states):
        self.received.append(list(states))
        return super().predict_proba(states)


class FixedModel(BehaviorModel):
    """Always predicts one side, ignoring inputs."""

    def __init__(self, side: str):
        self.side = side

    def fit(self, samples):
        return None

    def predict(self, samples):
        return [self.side] * len(samples)

    def predict_proba(self, states):
        if self.side == "LONG":
            return [Prediction(1.0, 0.0, "LONG", 1.0) for _ in states]
        return [Prediction(0.0, 1.0, "SHORT", 1.0) for _ in states]


def _build_real_samples(tmp_path, n=4) -> list[BehaviorSample]:
    storage = make_storage(tmp_path)
    storage.write_klines("BTCUSDT", "5m", zigzag_rows(n=40))
    aligner = TradeAligner(point_in_time=PointInTimeMarketState(storage=storage))
    chan = CausalChanEngine(storage=storage)
    geometry = GeometryFeatureExtractor(lookback=10)

    trades = [
        KOLTrade(
            kol="k", symbol="BTCUSDT", timestamp=AS_OF + timedelta(hours=i),
            side="LONG" if i % 2 == 0 else "SHORT", entry_price=100.0,
        )
        for i in range(n)
    ]
    dataset = BehaviorDataset.from_trades(
        trades, aligner, chan, geometry, geometry_timeframe="5m", chan_timeframes=("5m",)
    )
    return dataset.samples


def _fake_sample(timestamp: str, side: str) -> BehaviorSample:
    return BehaviorSample(
        kol="k", symbol="S", timestamp=timestamp, side=side,
        market_state={}, chan_state={}, geometry_features={}, chan_states={},
    )


# ---------------------------------------------------------------------------
# Test 3: model.predict_proba receives pure INPUT features only
# ---------------------------------------------------------------------------


def test_predict_proba_receives_only_state_features(tmp_path):
    samples = _build_real_samples(tmp_path, n=2)
    model = SpyPredictModel()
    assembler = StateFeatureAssembler(chan_timeframes=("5m",))

    evaluate_behavior_model(model, samples, assembler)

    assert len(model.received) == 2  # one call per sample
    for states in model.received:
        for features in states:
            assert list(features) == sorted(features)  # deterministic order
            # the KOL action label never appears in the input
            assert "action" not in features
            assert "side" not in features
            assert "pnl" not in features
            assert "actual" not in features


# ---------------------------------------------------------------------------
# Tests 4+5: validation and test never update model parameters
# ---------------------------------------------------------------------------


def test_evaluation_never_updates_model_parameters(tmp_path):
    samples = _build_real_samples(tmp_path, n=2)
    model = FitCountingModel()
    model.fit(samples)  # training happened once
    assert model.fit_calls == 1

    evaluate_behavior_model(model, samples, StateFeatureAssembler(chan_timeframes=("5m",)))  # validation
    evaluate_behavior_model(model, samples, StateFeatureAssembler(chan_timeframes=("5m",)))  # test

    assert model.fit_calls == 1  # evaluation never calls fit


# ---------------------------------------------------------------------------
# Test 6: test data never enters the training dataset
# ---------------------------------------------------------------------------


def test_test_samples_never_reach_fit(tmp_path):
    samples = _build_real_samples(tmp_path, n=4)
    dataset = BehaviorDataset(samples=samples)
    train, val, test = dataset.chronological_split()
    assert len(train) == 2 and len(val) == 1 and len(test) == 1  # 70/15/15 chronological

    model = FitCountingModel()
    trained = train_behavior_model(
        train, val,
        model_factory=lambda: model,
        assembler=StateFeatureAssembler(chan_timeframes=("5m",)),
        model_version="KOL-TWIN-v0",
    )

    test_timestamps = {s.timestamp for s in test}
    assert test_timestamps.isdisjoint(model.fit_timestamps)
    assert trained.model_version == "KOL-TWIN-v0"
    assert trained.feature_version == FEATURE_VERSION
    assert trained.training_data_version  # deterministic data version recorded


# ---------------------------------------------------------------------------
# Test 7: live prediction uses only timestamp + symbol
# ---------------------------------------------------------------------------


def test_live_prediction_uses_only_timestamp_and_symbol(tmp_path):
    storage = make_storage(tmp_path)
    storage.write_klines("BTCUSDT", "5m", zigzag_rows(n=40))
    pit = PointInTimeMarketState(storage=storage)
    chan = CausalChanEngine(storage=storage)
    assembler = StateFeatureAssembler(chan_timeframes=("5m",))
    model = BaselineBehaviorModel(default_side="SHORT")
    prediction_log = PredictionLog(tmp_path / "predictions.jsonl")

    entry = predict_kol_behavior(
        "BTCUSDT", AS_OF, pit, chan, assembler, model,
        model_version="KOL-TWIN-test", prediction_log=prediction_log,
    )

    assert entry["symbol"] == "BTCUSDT"
    assert entry["predicted_action"] == "SHORT"
    assert entry["confidence"] == 1.0
    assert entry["model_version"] == "KOL-TWIN-test"
    assert "actual_action" not in entry  # structurally impossible at T
    assert len(prediction_log.entries) == 1

    # live-path features == frozen-sample features (same assembler core)
    market_state = pit.get_market_state("BTCUSDT", AS_OF)
    chan_states = collect_chan_states(chan, "BTCUSDT", AS_OF, assembler.chan_timeframes)
    live_features = assembler.from_components(market_state, chan_states, trader_id="k")
    sample = _build_real_samples(tmp_path, n=1)[0]
    assert live_features == assembler.from_sample(sample)


# ---------------------------------------------------------------------------
# Test 8: feedback records correct/incorrect, never retrains
# ---------------------------------------------------------------------------


def test_feedback_records_without_touching_model(tmp_path):
    prediction_log = PredictionLog(tmp_path / "plog.jsonl")
    entry = {
        "timestamp": "2026-09-05T18:00:00+00:00",
        "symbol": "BTCUSDT",
        "long_probability": 0.1, "short_probability": 0.9,
        "predicted_action": "SHORT", "confidence": 0.9,
        "model_version": "KOL-TWIN-v1.0",
    }
    prediction_log.record(entry)
    memory = ErrorMemory(tmp_path / "feedback.jsonl")

    # The signature structurally has no model parameter: feedback cannot retrain.
    assert "model" not in inspect.signature(record_actual_kol_action).parameters

    feedback = record_actual_kol_action(prediction_log, memory, "2026-09-05T18:00:00+00:00", "BTCUSDT", "SHORT")
    assert feedback is not None
    assert feedback["correct"] is True
    assert feedback["actual_action"] == "SHORT"

    feedback_wrong = record_actual_kol_action(prediction_log, memory, "2026-09-05T18:00:00+00:00", "BTCUSDT", "LONG")
    assert feedback_wrong["correct"] is False  # wrong prediction fully preserved

    assert record_actual_kol_action(prediction_log, memory, "2099-01-01T00:00:00+00:00", "BTCUSDT", "LONG") is None
    assert memory.new_count() == 2  # correct AND incorrect both kept


def test_retrain_threshold_is_configurable(tmp_path):
    memory = ErrorMemory(tmp_path / "feedback.jsonl")
    config = TrainingConfig(retrain_min_new_samples=2)
    assert not memory.should_retrain(config)

    memory.add({"correct": True})
    memory.add({"correct": False})
    assert memory.should_retrain(config)
    assert TrainingConfig().retrain_min_new_samples == 500  # default, not hard-coded in logic


# ---------------------------------------------------------------------------
# Test 9: challenger must beat the champion before promotion
# ---------------------------------------------------------------------------


def _artifact(version: str, side: str) -> ModelArtifact:
    return ModelArtifact(
        model=FixedModel(side),
        model_version=version,
        training_data_version="t1",
        feature_version=FEATURE_VERSION,
        created_at="2026-09-05T00:00:00+00:00",
        validation_metrics={},
    )


def test_challenger_promotion_and_rejection():
    assembler = StateFeatureAssembler()
    # test set: 6 SHORT / 2 LONG -> majority (SHORT) accuracy 0.75
    test = [
        _fake_sample(f"2026-08-0{i}T00:00:00+00:00", "SHORT" if i % 4 else "LONG")
        for i in range(1, 9)
    ]
    champion = _artifact("KOL-TWIN-v1.0", "LONG")  # accuracy 0.25
    strong = _artifact("KOL-TWIN-v1.1", "SHORT")  # accuracy 0.75

    decision = evaluate_challenger(champion, strong, test, assembler)
    assert isinstance(decision, ChallengeDecision)
    assert decision.promoted is True  # +0.5 accuracy, f1 up, no class collapse

    weak = _artifact("KOL-TWIN-v1.2", "LONG")  # identical to champion
    rejected = evaluate_challenger(champion, weak, test, assembler)
    assert rejected.promoted is False  # no improvement -> discard


def test_model_registry_roundtrip(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    champion = _artifact("KOL-TWIN-v1.0", "LONG")
    registry.save(champion)
    registry.set_champion("KOL-TWIN-v1.0")

    loaded = registry.load("KOL-TWIN-v1.0")
    assert loaded.model_version == "KOL-TWIN-v1.0"
    assert loaded.feature_version == FEATURE_VERSION
    assert isinstance(loaded.model, FixedModel)

    current = registry.current_champion()
    assert current is not None
    assert current.model_version == "KOL-TWIN-v1.0"
    assert registry.list_versions() == ["KOL-TWIN-v1.0"]
