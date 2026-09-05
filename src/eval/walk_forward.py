"""Expanding-window walk-forward evaluation with strict time causality.

Rules enforced here:
- samples are sorted by timestamp (never shuffled, never random);
- each fold's train block contains ONLY data strictly before the test block;
- every fold trains a fresh model from scratch (independent fit);
- the model's internal preprocessing (imputer etc.) is fitted inside
  ``model.fit`` on the current fold's train data only;
- the test block is used exclusively for predict-then-answer evaluation.

No fold pads or fabricates data — with few samples the earliest folds
simply have smaller trains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.dataset.behavior_dataset import BehaviorSample
from src.features.state_features import StateFeatureAssembler
from src.models.behavior_model import BehaviorModel
from src.models.training import evaluate_behavior_model


@dataclass(slots=True)
class FoldResult:
    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_val: int
    n_test: int
    metrics: dict


def _log_loss(model: BehaviorModel, assembler: StateFeatureAssembler,
              samples: list[BehaviorSample]) -> float:
    losses = []
    for sample in samples:
        prediction = model.predict_proba([assembler.from_sample(sample)])[0]
        label = sample.action  # answer key, read after predicting
        p = prediction.long_probability if label == "LONG" else prediction.short_probability
        losses.append(-math.log(max(p, 1e-12)))
    return sum(losses) / len(losses) if losses else float("nan")


def _fold_metrics(model: BehaviorModel, assembler: StateFeatureAssembler,
                  test_samples: list[BehaviorSample]) -> dict:
    report = evaluate_behavior_model(model, test_samples, assembler).to_dict()
    report["log_loss"] = _log_loss(model, assembler, test_samples)

    per_trader: dict[str, float] = {}
    traders: dict[str, list[BehaviorSample]] = {}
    for sample in test_samples:
        traders.setdefault(sample.kol, []).append(sample)
    for trader, trader_samples in sorted(traders.items()):
        per_trader[trader] = evaluate_behavior_model(
            model, trader_samples, assembler
        ).macro_f1
    report["per_trader_macro_f1"] = per_trader
    report["worst_trader_macro_f1"] = min(per_trader.values()) if per_trader else float("nan")
    return report


def expanding_walk_forward(
    samples: list[BehaviorSample],
    model_factory: Callable[[], BehaviorModel],
    assembler: StateFeatureAssembler,
    n_splits: int = 4,
) -> list[FoldResult]:
    """Expanding-window walk-forward over n_splits+1 contiguous time blocks.

    Fold i: train = blocks[0..i], test = blocks[i+1]. Test never overlaps
    train in time, and every fold refits from scratch.
    """
    ordered = sorted(samples, key=lambda s: s.timestamp)
    if not ordered:
        return []
    n_blocks = n_splits + 1
    block_size = math.ceil(len(ordered) / n_blocks)
    blocks = [ordered[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
    blocks = [b for b in blocks if b]

    results: list[FoldResult] = []
    for fold_index in range(len(blocks) - 1):
        train_block = [s for block in blocks[:fold_index + 1] for s in block]
        test_block = blocks[fold_index + 1]
        model = model_factory()  # fresh per fold
        model.fit(train_block)  # preprocessing fitted here, train-only
        metrics = _fold_metrics(model, assembler, test_block)
        results.append(
            FoldResult(
                fold_index=fold_index,
                train_start=train_block[0].timestamp,
                train_end=train_block[-1].timestamp,
                test_start=test_block[0].timestamp,
                test_end=test_block[-1].timestamp,
                n_train=len(train_block),
                n_val=0,  # no per-fold validation; selection uses the global val split
                n_test=len(test_block),
                metrics=metrics,
            )
        )
    return results


def aggregate_folds(results: list[FoldResult]) -> dict:
    """Per-metric mean/std/min/max across folds."""
    metric_names = {
        "accuracy", "macro_f1", "log_loss", "worst_trader_macro_f1",
    }
    aggregation: dict[str, dict] = {}
    for name in sorted(metric_names):
        values = [r.metrics[name] for r in results if name in r.metrics]
        values = [v for v in values if v == v]  # drop NaN
        if not values:
            aggregation[name] = {"mean": None, "std": None, "min": None, "max": None}
            continue
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        aggregation[name] = {
            "mean": mean,
            "std": math.sqrt(variance),
            "min": min(values),
            "max": max(values),
        }
    return aggregation
