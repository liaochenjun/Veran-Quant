from __future__ import annotations

from datetime import timedelta

from chan_test_utils import START, make_storage, zigzag_rows
from src.alignment.trade_aligner import KOLTrade, TradeAligner
from src.chan.causal_chan import CausalChanEngine
from src.dataset.behavior_dataset import BehaviorDataset, BehaviorSample
from src.features.state_features import StateFeatureAssembler
from src.market.geometry import GeometryFeatureExtractor
from src.market.point_in_time import PointInTimeMarketState
from src.models.sklearn_models import LogisticRegressionModel, RandomForestModel


def _fake_sample(timestamp: str, side: str, kol: str = "aoying_capital") -> BehaviorSample:
    return BehaviorSample(
        kol=kol, symbol="S", timestamp=timestamp, side=side,
        market_state={}, chan_state={}, geometry_features={}, chan_states={},
    )


def _build_real_samples(tmp_path, n=6):
    storage = make_storage(tmp_path)
    storage.write_klines("BTCUSDT", "5m", zigzag_rows(n=40))
    aligner = TradeAligner(point_in_time=PointInTimeMarketState(storage=storage))
    chan = CausalChanEngine(storage=storage)
    geometry = GeometryFeatureExtractor(lookback=10)
    trades = [
        KOLTrade(kol="aoying_capital", symbol="BTCUSDT", timestamp=START + timedelta(hours=4 + i),
                 side="LONG" if i % 2 == 0 else "SHORT", entry_price=100.0)
        for i in range(n)
    ]
    return BehaviorDataset.from_trades(
        trades, aligner, chan, geometry, geometry_timeframe="5m", chan_timeframes=("5m",)
    ).samples


def test_logistic_regression_fit_and_proba(tmp_path):
    samples = _build_real_samples(tmp_path)
    assembler = StateFeatureAssembler(chan_timeframes=("5m",))
    model = LogisticRegressionModel(assembler)

    model.fit(samples)
    states = [assembler.from_sample(s) for s in samples]
    predictions = model.predict_proba(states)

    assert len(predictions) == len(samples)
    for p in predictions:
        assert abs(p.long_probability + p.short_probability - 1.0) < 1e-9
        assert p.predicted_action in ("LONG", "SHORT")
        assert 0.0 <= p.confidence <= 1.0
    assert model.predict(samples) == [p.predicted_action for p in predictions]


def test_random_forest_fit_and_proba(tmp_path):
    samples = _build_real_samples(tmp_path)
    assembler = StateFeatureAssembler(chan_timeframes=("5m",))
    model = RandomForestModel(assembler, n_estimators=20)

    model.fit(samples)
    predictions = model.predict_proba([assembler.from_sample(s) for s in samples])
    assert len(predictions) == len(samples)
    for p in predictions:
        assert abs(p.long_probability + p.short_probability - 1.0) < 1e-9


def test_sklearn_models_handle_nan_features_and_single_class(tmp_path):
    # empty market_state -> trader one-hot features only, no NaN in matrix;
    # single-class train set -> probabilities renormalized defensively
    samples = [_fake_sample(f"2026-08-0{i}T00:00:00+00:00", "LONG") for i in range(1, 4)]
    assembler = StateFeatureAssembler()
    model = LogisticRegressionModel(assembler)

    model.fit(samples)
    states = [assembler.from_sample(s) for s in samples]
    assert any("trader__aoying_capital" in s for s in states)

    predictions = model.predict_proba(states)
    assert all(p.predicted_action == "LONG" for p in predictions)
    assert all(abs(p.long_probability + p.short_probability - 1.0) < 1e-9 for p in predictions)


def test_feature_matrix_handles_mixed_coverage_states(tmp_path):
    # Regression: partial data coverage means different samples have
    # different feature key sets; the matrix must use the key union and
    # fill missing features with NaN (imputer handles them later).
    from src.models.sklearn_models import _SklearnBehaviorModel

    states = [
        {"a": 1.0, "b": 2.0},
        {"b": 3.0, "c": 4.0},
    ]
    keys, matrix = _SklearnBehaviorModel._feature_matrix(states)
    assert keys == ["a", "b", "c"]
    assert matrix[0][2] != matrix[0][2]  # NaN for missing "c" in state 0
    assert matrix[1][0] != matrix[1][0]  # NaN for missing "a" in state 1

    # round trip: fit and predict on the SAME assembler-produced features
    assembler = StateFeatureAssembler()
    model = LogisticRegressionModel(assembler)
    samples = [
        _fake_sample("2026-08-01T00:00:00+00:00", "LONG"),
        _fake_sample("2026-08-02T00:00:00+00:00", "SHORT"),
    ]
    model.fit(samples)
    predictions = model.predict_proba([assembler.from_sample(s) for s in samples])
    assert len(predictions) == 2


def test_prediction_aligns_to_train_time_feature_schema():
    # Regression: train and val feature key sets can differ when symbols
    # have partial data coverage (120 vs 200 columns crash). Prediction
    # must align to the canonical train-time schema, NaN-filling gaps.
    from src.models.sklearn_models import LogisticRegressionModel

    train_states = [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}, {"a": 1.0, "b": 1.0}]
    val_states = [{"b": 1.0, "c": 0.5}]  # different key set than train

    class _DirectLR(LogisticRegressionModel):
        def __init__(self):
            self.assembler = None
            self.estimator = None
            self.classes = ["LONG", "SHORT"]

        def fit_states(self, states, labels):
            from src.models.sklearn_models import _SklearnBehaviorModel
            self._feature_keys = sorted({k for s in states for k in s})
            _, matrix = _SklearnBehaviorModel._feature_matrix(states, self._feature_keys)
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            self.estimator = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LogisticRegression(max_iter=1000)),
            ])
            self.estimator.fit(matrix, labels)

    model = _DirectLR()
    model.fit_states(train_states, ["LONG", "SHORT", "LONG"])
    predictions = model.predict_proba(val_states)
    assert len(predictions) == 1
    assert abs(predictions[0].long_probability + predictions[0].short_probability - 1.0) < 1e-9


def test_unfitted_sklearn_model_returns_prior():
    assembler = StateFeatureAssembler()
    model = RandomForestModel(assembler, n_estimators=10)
    predictions = model.predict_proba([assembler.from_sample(_fake_sample("2026-08-01T00:00:00+00:00", "SHORT"))])
    assert predictions[0].predicted_action == "LONG"
    assert predictions[0].confidence == 1.0
