"""Train / validation / test pipeline for behavior cloning.

Strict phase separation (prompt.txt):

- TRAIN:     STATE_AT_T + KOL_ACTION_AT_T -> fit -> parameter updates.
- VALIDATE:  STATE_AT_T -> predict -> only THEN compare with the label;
             metrics are reported but parameters are never updated.
- TEST:      same predict-then-answer discipline; final exam only, its
             data never enters any training call.

The baseline model has no iterative parameters yet, so early-stopping /
checkpointing is structured in TrainingConfig and documented as no-ops
for the baseline; they become live once a parametric model lands.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from src.dataset.behavior_dataset import BehaviorSample
from src.features.state_features import FEATURE_VERSION, StateFeatureAssembler
from src.models.behavior_model import BaselineBehaviorModel, BehaviorModel
from src.models.prediction import Prediction


@dataclass(slots=True)
class TrainingConfig:
    """Pipeline configuration. Retrain threshold is a config item, never
    hard-coded in logic."""

    max_epochs: int = 1  # baseline has no iterative training
    early_stopping_patience: int = 3
    checkpoint_dir: Optional[str] = None
    random_seed: Optional[int] = None
    retrain_min_new_samples: int = 500  # accumulate feedback before retraining
    high_confidence_threshold: float = 0.7  # for confidence analysis


@dataclass(slots=True)
class EvaluationReport:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]  # side -> {precision, recall, f1, support}
    confusion_matrix: dict[str, dict[str, int]]  # actual -> predicted -> count
    high_confidence_accuracy: Optional[float]  # accuracy of confident predictions
    low_confidence_accuracy: Optional[float]
    high_confidence_threshold: float
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
            "high_confidence_accuracy": self.high_confidence_accuracy,
            "low_confidence_accuracy": self.low_confidence_accuracy,
            "high_confidence_threshold": self.high_confidence_threshold,
            "n_samples": self.n_samples,
        }


@dataclass(slots=True)
class TrainedModel:
    model: BehaviorModel
    model_version: str
    training_data_version: str
    feature_version: str
    created_at: str
    validation_metrics: dict
    test_metrics: Optional[dict] = None


def _training_data_version(samples: list[BehaviorSample]) -> str:
    digest = hashlib.sha256()
    for sample in sorted(samples, key=lambda s: (s.timestamp, s.symbol, s.kol)):
        digest.update(f"{sample.timestamp}|{sample.symbol}|{sample.kol}|{sample.side}".encode())
    return digest.hexdigest()[:12]


def _metrics(y_true: list[str], y_pred: list[str], confidence: list[float],
             threshold: float) -> EvaluationReport:
    n = len(y_true)
    accuracy = sum(1 for a, b in zip(y_true, y_pred) if a == b) / n if n else 0.0
    classes = sorted(set(y_true) | set(y_pred))
    per_class: dict[str, dict[str, float]] = {}
    confusion: dict[str, dict[str, int]] = {actual: {pred: 0 for pred in classes} for actual in classes}
    for actual, pred in zip(y_true, y_pred):
        confusion.setdefault(actual, {}).setdefault(pred, 0)
        confusion[actual][pred] += 1

    macro_p = macro_r = macro_f1 = 0.0
    for cls in classes:
        tp = confusion.get(cls, {}).get(cls, 0)
        fp = sum(confusion[a].get(cls, 0) for a in classes if a != cls)
        fn = sum(confusion[cls].get(p, 0) for p in classes if p != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[cls] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": sum(confusion.get(cls, {}).values()),
        }
        macro_p += precision
        macro_r += recall
        macro_f1 += f1
    k = len(classes) or 1

    high = [(a, b) for a, b, c in zip(y_true, y_pred, confidence) if c >= threshold]
    low = [(a, b) for a, b, c in zip(y_true, y_pred, confidence) if c < threshold]
    high_acc = sum(1 for a, b in high if a == b) / len(high) if high else None
    low_acc = sum(1 for a, b in low if a == b) / len(low) if low else None

    return EvaluationReport(
        accuracy=accuracy,
        macro_precision=macro_p / k,
        macro_recall=macro_r / k,
        macro_f1=macro_f1 / k,
        per_class=per_class,
        confusion_matrix=confusion,
        high_confidence_accuracy=high_acc,
        low_confidence_accuracy=low_acc,
        high_confidence_threshold=threshold,
        n_samples=n,
    )


def evaluate_behavior_model(
    model: BehaviorModel,
    samples: list[BehaviorSample],
    assembler: StateFeatureAssembler,
    confidence_threshold: float = 0.7,
) -> EvaluationReport:
    """Predict first, compare with the label AFTER predicting.

    Never calls model.fit — evaluation can not update parameters.
    """
    y_true: list[str] = []
    y_pred: list[str] = []
    confidence: list[float] = []
    for sample in samples:
        features = assembler.from_sample(sample)  # INPUT only
        prediction: Prediction = model.predict_proba([features])[0]
        y_pred.append(prediction.predicted_action)
        # Only now may the stored KOL action be read (answer key).
        y_true.append(sample.action)
        confidence.append(prediction.confidence)
    return _metrics(y_true, y_pred, confidence, confidence_threshold)


def train_behavior_model(
    train_samples: list[BehaviorSample],
    validation_samples: list[BehaviorSample],
    config: Optional[TrainingConfig] = None,
    model_factory: Optional[Callable[[], BehaviorModel]] = None,
    assembler: Optional[StateFeatureAssembler] = None,
    model_version: Optional[str] = None,
) -> TrainedModel:
    """Train on train (labels allowed), validate without parameter updates."""
    config = config or TrainingConfig()
    assembler = assembler or StateFeatureAssembler()
    model = model_factory() if model_factory else BaselineBehaviorModel()

    # TRAIN: labels are the supervision signal here — the only place they are.
    model.fit(train_samples)

    # VALIDATE: predict-then-answer; no fit, no parameter updates.
    validation_report = evaluate_behavior_model(
        model, validation_samples, assembler, config.high_confidence_threshold
    )

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return TrainedModel(
        model=model,
        model_version=model_version or f"KOL-TWIN-v1-{created[:10].replace('-', '')}",
        training_data_version=_training_data_version(train_samples),
        feature_version=FEATURE_VERSION,
        created_at=created,
        validation_metrics=validation_report.to_dict(),
    )
