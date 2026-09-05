"""P1.5 discipline tests: scaler fit discipline, per-fold independence,
all-NaN handling, dataset immutability, class-weight and GBDT label hygiene."""

from __future__ import annotations

import math
from dataclasses import asdict
from datetime import datetime

from src.chan.chan_state import ChanState
from src.dataset.behavior_dataset import BehaviorSample
from src.eval.walk_forward import expanding_walk_forward
from src.features.state_features import StateFeatureAssembler
from src.models.sklearn_models import (
    LightGBMModel,
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
)

_EMPTY_CHAN_5M = ChanState.empty("S", "5m", datetime(2026, 8, 1)).to_dict()


def _sample(timestamp: str, side: str, market_records=None, kol="aoying_capital") -> BehaviorSample:
    return BehaviorSample(
        kol=kol, symbol="S", timestamp=timestamp, side=side,
        market_state={"5m": market_records or []},
        chan_state={}, geometry_features={}, chan_states={"5m": _EMPTY_CHAN_5M},
    )


def _records(closes=(100.0, 101.0, 102.0)):
    return [
        {
            "open_time": f"2026-08-01T00:{i:02d}:00+00:00",
            "close_time": f"2026-08-01T00:{i:02d}:59+00:00",
            "open": c - 0.5, "high": c + 0.5, "low": c - 0.5, "close": c,
            "volume": 10.0,
        }
        for i, c in enumerate(closes)
    ]


def _features_for(assembler, sample):
    return assembler.from_sample(sample)


def test_standard_scaler_fits_only_on_train():
    assembler = StateFeatureAssembler()
    train = [
        _sample("2026-08-01T01:00:00+00:00", "LONG", _records()),
        _sample("2026-08-01T02:00:00+00:00", "SHORT", _records((99.0, 98.0, 97.0))),
        _sample("2026-08-01T03:00:00+00:00", "LONG", _records((103.0, 104.0, 105.0))),
    ]
    model = LogisticRegressionModel(assembler)
    model.fit(train)

    scaler = model.estimator.named_steps["scaler"]
    train_features = [assembler.from_sample(s) for s in train]
    # imputer first, so expected mean is over finite values only
    imputed_keys = model._feature_keys
    matrix = [[float(f.get(k, float("nan"))) for k in imputed_keys] for f in train_features]
    finite_cols = []
    for col in zip(*matrix):
        vals = [v for v in col if v == v]
        finite_cols.append(sum(vals) / len(vals) if vals else 0.0)
    assert len(scaler.mean_) == len(imputed_keys)
    for actual, expected in zip(scaler.mean_, finite_cols):
        assert abs(actual - expected) < 1e-9  # scaler statistics == train statistics


def test_walk_forward_scaler_refits_per_fold():
    assembler = StateFeatureAssembler()
    samples = sorted(
        [
            _sample(f"2026-08-{i // 24 + 1:02d}T{i % 24:02d}:00:00+00:00",
                    "LONG" if i % 2 else "SHORT", _records((100.0 + i, 101.0 + i, 102.0 + i)))
            for i in range(40)
        ],
        key=lambda s: s.timestamp,
    )
    models = []
    folds = expanding_walk_forward(
        samples, lambda: models.append(LogisticRegressionModel(assembler)) or models[-1],
        assembler, n_splits=3,
    )
    assert len(models) == 3  # fresh model per fold
    scaler_means = [m.estimator.named_steps["scaler"].mean_ for m in models]
    # fold trains differ -> scaler statistics differ per fold
    assert not all(
        all(abs(a - b) < 1e-12 for a, b in zip(m1, m2))
        for m1, m2 in zip(scaler_means, scaler_means[1:])
    )
    # each fold's scaler matches ITS OWN train statistics
    ordered = sorted(samples, key=lambda s: s.timestamp)
    for fold_index, model in enumerate(models):
        train_block = ordered[: (fold_index + 1) * 10]
        expected = _train_col_means(assembler, train_block, model._feature_keys)
        for actual, want in zip(model.estimator.named_steps["scaler"].mean_, expected):
            assert abs(actual - want) < 1e-9


def _train_col_means(assembler, train_block, keys):
    features = [assembler.from_sample(s) for s in train_block]
    matrix = [[float(f.get(k, float("nan"))) for k in keys] for f in features]
    means = []
    for col in zip(*matrix):
        vals = [v for v in col if v == v]
        means.append(sum(vals) / len(vals) if vals else 0.0)
    return means


def test_all_nan_features_are_dropped_from_matrix():
    assembler = StateFeatureAssembler()
    samples = [
        _sample("2026-08-01T01:00:00+00:00", "LONG"),
        _sample("2026-08-01T02:00:00+00:00", "SHORT"),
    ]
    model = LogisticRegressionModel(assembler)
    model.fit(samples)
    # with empty chan snapshots, chan value fields are all-NaN on train
    # (e.g. chan__5m__bi_direction_up) and must be excluded from the matrix
    dropped = set(model._all_nan_keys)
    assert dropped, "expected some all-NaN features to be dropped"
    assert not set(model._feature_keys) & dropped
    predictions = model.predict_proba([assembler.from_sample(s) for s in samples])
    assert len(predictions) == 2


def test_feature_cleanup_does_not_modify_dataset():
    assembler = StateFeatureAssembler()
    samples = [
        _sample("2026-08-01T01:00:00+00:00", "LONG", _records()),
        _sample("2026-08-01T02:00:00+00:00", "SHORT", _records((99.0, 98.0, 97.0))),
    ]
    before = [asdict(s) for s in samples]
    model = LogisticRegressionModel(assembler)
    model.fit(samples)
    model.predict_proba([assembler.from_sample(s) for s in samples])
    after = [asdict(s) for s in samples]
    assert before == after  # raw dataset untouched


def test_xgboost_balanced_uses_train_labels_only():
    assembler = StateFeatureAssembler()
    train = [
        _sample(f"2026-08-01T0{i}:00:00+00:00", "LONG" if i % 3 else "SHORT")
        for i in range(1, 7)
    ]  # 4 LONG, 2 SHORT
    model = XGBoostModel(assembler, n_estimators=10, class_weight="balanced")
    model.fit(train)
    n_long = sum(1 for s in train if s.action == "LONG")
    n_short = sum(1 for s in train if s.action == "SHORT")
    expected = n_long / n_short
    actual = model.estimator.named_steps["model"].get_params()["scale_pos_weight"]
    assert abs(actual - expected) < 1e-9


def test_gbdt_predictions_never_read_actual_action():
    assembler = StateFeatureAssembler()
    samples = [
        _sample("2026-08-01T01:00:00+00:00", "LONG", _records()),
        _sample("2026-08-01T02:00:00+00:00", "SHORT", _records((99.0, 98.0, 97.0))),
        _sample("2026-08-01T03:00:00+00:00", "LONG", _records((103.0, 104.0, 105.0))),
    ]
    for model in (LightGBMModel(assembler, n_estimators=10),
                  XGBoostModel(assembler, n_estimators=10)):
        model.fit(samples)
        features = [assembler.from_sample(s) for s in samples]
        for state in features:
            assert "action" not in state and "side" not in state  # inputs only
        predictions = model.predict_proba(features)
        assert len(predictions) == len(samples)
        for p in predictions:
            assert abs(p.long_probability + p.short_probability - 1.0) < 1e-9
