"""P2-1: recency window + exponential time decay for behavior cloning.

Every (fold, window) combination trains a FRESH Random Forest — models are
never shared between windows. Window filtering uses only ``timestamp <
test_start``; time decay weights are computed fold-specifically relative to
``test_start`` (never test_end, never the dataset end).

- ``filter_window``: keep only samples with test_start - window <= ts < test_start
- ``decay_weights``: exp(-lambda * age_days), age = test_start - ts (assert > 0)
- ``DecayRandomForestModel``: RF with optional per-sample exponential weights
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.dataset.behavior_dataset import BehaviorSample
from src.features.state_features import StateFeatureAssembler
from src.models.sklearn_models import RandomForestModel

WINDOWS = ("all", "180d", "90d", "60d", "30d")
MIN_TRAIN_SAMPLES = 20


def _as_utc(timestamp: str) -> datetime:
    dt = datetime.fromisoformat(timestamp)
    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def filter_window(
    samples: list[BehaviorSample], test_start: str, window: str
) -> list[BehaviorSample]:
    """Train set for one (fold, window): strictly before test_start, within window."""
    start = _as_utc(test_start)
    kept = []
    for sample in samples:
        t = _as_utc(sample.timestamp)
        if t >= start:
            continue  # never future data
        if window != "all" and t < start - timedelta(days=int(window[:-1])):
            continue  # older than the window
        kept.append(sample)
    return sorted(kept, key=lambda s: s.timestamp)


def decay_weights(samples: list[BehaviorSample], test_start: str, lambda_days: float) -> list[float]:
    """Exponential weights exp(-lambda * age_days); age relative to test_start."""
    start = _as_utc(test_start)
    weights = []
    for sample in samples:
        t = _as_utc(sample.timestamp)
        age_days = (start - t).total_seconds() / 86400.0
        assert age_days > 0, f"leakage: sample at {sample.timestamp} not before {test_start}"
        weight = math.exp(-lambda_days * age_days)
        assert weight > 0
        weights.append(weight)
    return weights


class DecayRandomForestModel(RandomForestModel):
    """RF with optional exponential recency weights.

    A fresh instance is trained per (fold, window, lambda); nothing is
    shared across experiments. ``cutoff`` is the fold's test_start.
    """

    def __init__(
        self,
        assembler: StateFeatureAssembler,
        n_estimators: int = 300,
        lambda_days: float = 0.0,
        cutoff: Optional[str] = None,
    ):
        super().__init__(assembler, n_estimators=n_estimators)
        self.lambda_days = lambda_days
        self.cutoff = cutoff

    def fit(self, samples: list[BehaviorSample]) -> None:
        if not samples:
            return
        features, labels = self._build_xy(samples, self.assembler)
        self.classes = sorted(set(labels))
        if len(self.classes) < 2:
            self.estimator = None
            return
        _, matrix = self._prepare_features(features)
        if self.lambda_days > 0 and self.cutoff is not None:
            weights = decay_weights(samples, self.cutoff, self.lambda_days)
            self.estimator.fit(matrix, labels, model__sample_weight=weights)
        else:
            self.estimator.fit(matrix, labels)
