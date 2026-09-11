"""P2-1 recency/time-decay discipline tests (spec section 25)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.dataset.behavior_dataset import BehaviorSample
from src.eval.recency import (
    MIN_TRAIN_SAMPLES,
    DecayRandomForestModel,
    decay_weights,
    filter_window,
)
from src.eval.walk_forward import walk_forward_fold_blocks
from src.features.state_features import StateFeatureAssembler
from src.models.sklearn_models import RandomForestModel


def _sample(ts: datetime, side: str = "LONG", kol: str = "aoying_capital") -> BehaviorSample:
    return BehaviorSample(
        kol=kol, symbol="S", timestamp=ts.isoformat(), side=side,
        market_state={}, chan_state={}, geometry_features={}, chan_states={},
    )


def _samples(n=80) -> list[BehaviorSample]:
    base = datetime(2026, 6, 1, tzinfo=timezone.utc)
    return [
        _sample(base + timedelta(hours=6 * i), "LONG" if i % 2 else "SHORT",
                kol=["aoying_capital", "liuyuan", "xijiuye"][i % 3])
        for i in range(n)
    ]


def test_recency_window_excludes_future_data():
    samples = _samples()
    test_start = "2026-06-10T00:00:00+00:00"
    for window in ("all", "180d", "90d", "60d", "30d"):
        train = filter_window(samples, test_start, window)
        start = datetime.fromisoformat(test_start)
        assert all(datetime.fromisoformat(s.timestamp) < start for s in train)


def test_recency_window_boundary():
    samples = _samples()
    test_start = "2026-06-20T00:00:00+00:00"
    start = datetime.fromisoformat(test_start)
    train30 = filter_window(samples, test_start, "30d")
    assert all(datetime.fromisoformat(s.timestamp) >= start - timedelta(days=30) for s in train30)
    # 90d is a superset of 30d
    train90 = filter_window(samples, test_start, "90d")
    assert set(s.timestamp for s in train30) <= set(s.timestamp for s in train90)
    # a sample exactly at the boundary (test_start - 30d) is INCLUDED
    edge = _sample(start - timedelta(days=30), "LONG")
    assert edge.timestamp in {s.timestamp for s in filter_window([edge], test_start, "30d")}
    # a sample one second before the boundary is EXCLUDED
    too_old = _sample(start - timedelta(days=30, seconds=1), "LONG")
    assert filter_window([too_old], test_start, "30d") == []


def test_each_window_retrains_model():
    # every (fold, window) fit is a fresh RandomForestModel instance
    class SpyRF(RandomForestModel):
        instances: list["SpyRF"] = []

        def __init__(self, assembler):
            super().__init__(assembler, n_estimators=10)
            SpyRF.instances.append(self)
            self.fit_calls = 0

        def fit(self, samples):
            self.fit_calls += 1
            super().fit(samples)

    SpyRF.instances = []
    samples = _samples()
    folds = walk_forward_fold_blocks(samples, n_splits=3)
    assembler = StateFeatureAssembler()
    total_fits = 0
    for fold in folds:
        for window in ("all", "30d"):
            train = filter_window(samples, fold.test_start, window)
            if len(train) < MIN_TRAIN_SAMPLES:
                continue
            model = SpyRF(assembler)
            model.fit(train)
            total_fits += 1
    assert total_fits == sum(1 for m in SpyRF.instances if m.fit_calls == 1)
    assert total_fits >= 4  # multiple windows actually retrained


def test_rf_pipeline_has_no_scaler():
    # spec: keep the existing RF implementation; RF uses imputer only
    model = RandomForestModel(StateFeatureAssembler(), n_estimators=10)
    steps = [name for name, _ in model.estimator.steps]
    assert "scaler" not in steps
    assert "imputer" in steps and "model" in steps


def test_decay_age_is_positive():
    samples = _samples()
    test_start = "2026-06-20T00:00:00+00:00"
    train = filter_window(samples, test_start, "90d")
    weights = decay_weights(train, test_start, lambda_days=0.01)
    assert len(weights) == len(train)
    assert all(w > 0 for w in weights)
    # newer sample -> larger weight
    start = datetime.fromisoformat(test_start)
    ages = [(start - datetime.fromisoformat(s.timestamp)).total_seconds() for s in train]
    assert weights == sorted(weights)  # ascending: oldest sample -> smallest weight
    assert all(ages[i] > 0 for i in range(len(ages)))


def test_decay_weight_is_fold_specific():
    samples = _samples()
    t1 = "2026-06-20T00:00:00+00:00"
    t2 = "2026-07-05T00:00:00+00:00"
    common = filter_window(samples, t1, "90d")
    w1 = decay_weights(common, t1, lambda_days=0.01)
    w2 = decay_weights(common, t2, lambda_days=0.01)
    assert w1 != w2  # same samples, different cutoffs -> different weights


def test_insufficient_samples_skip():
    samples = _samples(n=10)  # tiny set
    folds = walk_forward_fold_blocks(samples, n_splits=3)
    for fold in folds:
        train = filter_window(samples, fold.test_start, "30d")
        assert len(train) < MIN_TRAIN_SAMPLES  # every 30d window must be skipped


def test_reproducibility():
    samples = _samples()
    folds = walk_forward_fold_blocks(samples, n_splits=3)
    fold = folds[1]
    assembler = StateFeatureAssembler()

    def run_once():
        model = DecayRandomForestModel(assembler, n_estimators=50, lambda_days=0.005,
                                       cutoff=fold.test_start)
        model.fit(filter_window(samples, fold.test_start, "90d"))
        probs = [model.predict_proba([assembler.from_sample(s)])[0].long_probability
                 for s in fold.test_block]
        return probs

    first, second = run_once(), run_once()
    # deterministic up to float epsilon (parallel tree aggregation)
    assert max(abs(a - b) for a, b in zip(first, second)) < 1e-9
