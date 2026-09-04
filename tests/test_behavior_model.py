from __future__ import annotations

from src.dataset.behavior_dataset import BehaviorSample
from src.models.behavior_model import BaselineBehaviorModel, ReplayValidator


def _sample(side: str, timestamp: str = "2026-08-20T14:00:00+00:00") -> BehaviorSample:
    return BehaviorSample(
        kol="k", symbol="S", timestamp=timestamp, side=side,
        market_state={}, chan_state={}, geometry_features={},
    )


def test_fit_sets_majority_side():
    model = BaselineBehaviorModel()
    model.fit([_sample("LONG"), _sample("LONG"), _sample("SHORT")])
    assert model.default_side == "LONG"


def test_fit_tie_keeps_current_default():
    # Regression: statistics.mode raises StatisticsError on ties.
    model = BaselineBehaviorModel(default_side="LONG")
    model.fit([_sample("LONG"), _sample("SHORT")])
    assert model.default_side == "LONG"


def test_fit_empty_keeps_current_default():
    model = BaselineBehaviorModel(default_side="LONG")
    model.fit([])
    assert model.default_side == "LONG"


def test_predict_returns_default_for_all_samples():
    model = BaselineBehaviorModel()
    model.fit([_sample("SHORT"), _sample("SHORT"), _sample("LONG")])
    assert model.predict([_sample("LONG"), _sample("SHORT")]) == ["SHORT", "SHORT"]


def test_replay_validator_outputs():
    validator = ReplayValidator(model=BaselineBehaviorModel())
    outputs = validator.run(
        [_sample("LONG", "2026-08-20T14:00:00+00:00"), _sample("SHORT", "2026-08-20T15:00:00+00:00")]
    )

    assert outputs == [
        {
            "timestamp": "2026-08-20T14:00:00+00:00",
            "kol": "k",
            "symbol": "S",
            "predicted_side": "LONG",
            "actual_side": "LONG",
        },
        {
            "timestamp": "2026-08-20T15:00:00+00:00",
            "kol": "k",
            "symbol": "S",
            "predicted_side": "LONG",
            "actual_side": "SHORT",
        },
    ]
