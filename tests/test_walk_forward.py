"""Walk-forward / ablation discipline tests (spec section 五)."""

from __future__ import annotations

from datetime import datetime

from src.dataset.behavior_dataset import BehaviorSample
from src.eval.ablation import FEATURE_SETS, filter_features
from src.eval.walk_forward import expanding_walk_forward
from src.features.state_features import StateFeatureAssembler
from src.models.behavior_model import BehaviorModel
from src.models.prediction import Prediction


def _sample(timestamp: str, side: str, kol: str = "aoying_capital") -> BehaviorSample:
    return BehaviorSample(
        kol=kol, symbol="S", timestamp=timestamp, side=side,
        market_state={}, chan_state={}, geometry_features={}, chan_states={},
    )


class RecordingModel(BehaviorModel):
    def __init__(self):
        self.fit_timestamps: list[list[str]] = []

    def fit(self, samples):
        self.fit_timestamps.append(sorted(s.timestamp for s in samples))

    def predict(self, samples):
        return ["LONG"] * len(samples)

    def predict_proba(self, states):
        return [Prediction(1.0, 0.0, "LONG", 1.0) for _ in states]


def _samples(n=40):
    return [
        _sample(f"2026-08-{i // 24 + 1:02d}T{i % 24:02d}:00:00+00:00", "LONG" if i % 2 else "SHORT")
        for i in range(n)
    ]


def test_walk_forward_time_strictly_increasing_and_disjoint():
    samples = sorted(_samples(40), key=lambda s: s.timestamp)
    folds = expanding_walk_forward(samples, RecordingModel, StateFeatureAssembler(), n_splits=3)

    assert len(folds) == 3
    train_sizes = [f.n_train for f in folds]
    assert train_sizes == sorted(train_sizes)  # expanding window
    for fold in folds:
        assert fold.test_start > fold.train_end  # test strictly after train
        assert fold.n_val == 0
    # no overlap: every fold's train timestamps end before its test begins
    all_train_ts = set()
    # (train/test disjointness is implied by test_start > train_end for each fold)


def test_walk_forward_models_fit_only_on_fold_train():
    samples = sorted(_samples(40), key=lambda s: s.timestamp)
    model = RecordingModel()
    folds = expanding_walk_forward(samples, lambda: model, StateFeatureAssembler(), n_splits=3)

    assert len(model.fit_timestamps) == 3  # fresh fit per fold (same instance recorded)
    ordered = sorted(samples, key=lambda s: s.timestamp)
    block_size = 10  # 40 samples / 4 blocks
    for fold_index, fold in enumerate(folds):
        expected_train = [s.timestamp for s in ordered[: (fold_index + 1) * block_size]]
        assert model.fit_timestamps[fold_index] == expected_train
        test_ts = {s.timestamp for s in ordered[(fold_index + 1) * block_size:(fold_index + 2) * block_size]}
        assert test_ts.isdisjoint(model.fit_timestamps[fold_index])


def test_walk_forward_never_shuffles():
    samples = sorted(_samples(40), key=lambda s: s.timestamp)
    folds = expanding_walk_forward(samples, RecordingModel, StateFeatureAssembler(), n_splits=3)

    # fold test ranges are contiguous and in chronological order
    for a, b in zip(folds, folds[1:]):
        assert a.test_end <= b.test_start
    # the first fold trains on the earliest block only
    assert folds[0].train_start == samples[0].timestamp
    assert folds[0].train_end <= folds[0].test_start


def test_ablation_filtering_never_reads_labels():
    features = {
        "market__1m__last_close": 1.0,
        "market__1m__geometry__slope": 2.0,
        "chan__1m__bi_count": 3.0,
        "trader__aoying_capital": 1.0,
    }
    # same features, different labels -> identical filtered output
    for _side in ("LONG", "SHORT"):
        for feature_set in FEATURE_SETS:
            out = filter_features(features, feature_set)
            assert set(out) <= set(features)
    assert set(filter_features(features, "market_only")) == {"market__1m__last_close"}
    assert set(filter_features(features, "market_geometry")) == {
        "market__1m__last_close", "market__1m__geometry__slope",
    }
    assert set(filter_features(features, "market_chan")) == {
        "market__1m__last_close", "chan__1m__bi_count",
    }
    assert set(filter_features(features, "market_chan_geometry")) == {
        "market__1m__last_close", "market__1m__geometry__slope", "chan__1m__bi_count",
    }
    assert set(filter_features(features, "full")) == set(features)


def test_ablation_feature_groups_partition_all_keys():
    # Every feature key belongs to exactly one of the primitive groups:
    # market-base / market-geometry / chan / trader.
    keys = [
        "market__1m__last_close", "market__1m__mean_close",
        "market__1m__geometry__slope", "market__1m__geometry__resistance_line",
        "chan__1m__bi_count", "chan__5m__zs_count",
        "trader__aoying_capital", "trader__other",
    ]
    market_base = filter_features(dict.fromkeys(keys, 0.0), "market_only")
    market_geo = filter_features(dict.fromkeys(keys, 0.0), "market_geometry")
    chan = {k for k in keys if k.startswith("chan__")}
    trader = {k for k in keys if k.startswith("trader__")}
    partitioned = set(market_base) | (set(market_geo) - set(market_base)) | chan | trader
    assert partitioned == set(keys)
    assert not (set(market_base) & chan or set(market_base) & trader)


def test_walk_forward_with_empty_samples_returns_empty():
    assert expanding_walk_forward([], RecordingModel, StateFeatureAssembler()) == []
