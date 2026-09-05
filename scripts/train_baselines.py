"""Baseline model comparison on the full behavior dataset.

Models: majority-class baseline, logistic regression, random forest.
Split: chronological 70/15/15 (never random). Metrics: global +
per-trader + per-action, including log loss and confidence analysis.
The test set is only read AFTER each model has been trained on train
and selected on validation — test never touches fitting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# Make the repo root importable when run as `python scripts/train_baselines.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.behavior_dataset import BehaviorDataset, BehaviorSample  # noqa: E402
from src.features.state_features import StateFeatureAssembler  # noqa: E402
from src.models.behavior_model import BaselineBehaviorModel, BehaviorModel  # noqa: E402
from src.models.prediction import Prediction  # noqa: E402
from src.models.sklearn_models import LogisticRegressionModel, RandomForestModel  # noqa: E402
from src.models.training import evaluate_behavior_model  # noqa: E402


def _log_loss(y_true: list[str], prob_true: list[float]) -> float:
    losses = [-math.log(max(p, 1e-12)) for p in prob_true]
    return sum(losses) / len(losses) if losses else float("nan")


def _collect(
    model: BehaviorModel,
    assembler: StateFeatureAssembler,
    samples: list[BehaviorSample],
) -> tuple[dict, dict]:
    """Predict-then-answer: predictions first, labels read only afterwards."""
    y_true: list[str] = []
    prob_true: list[float] = []
    per_trader: dict[str, list[BehaviorSample]] = defaultdict(list)
    for sample in samples:
        features = assembler.from_sample(sample)
        prediction = model.predict_proba([features])[0]
        label = sample.action  # answer key, read after predicting
        y_true.append(label)
        prob_true.append(
            prediction.long_probability if label == "LONG" else prediction.short_probability
        )
        per_trader[sample.kol].append(sample)

    report = evaluate_behavior_model(model, samples, assembler).to_dict()
    report["log_loss"] = _log_loss(y_true, prob_true)
    trader_metrics: dict[str, dict] = {}
    for trader, trader_samples in sorted(per_trader.items()):
        # per-trader report via the same predict-then-answer evaluator:
        # accuracy + macro F1 (+ per-class precision/recall)
        trader_report = evaluate_behavior_model(model, trader_samples, assembler)
        trader_metrics[trader] = {
            "samples": len(trader_samples),
            "accuracy": trader_report.accuracy,
            "macro_f1": trader_report.macro_f1,
        }
    return report, trader_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/behavior_dataset_full.json")
    parser.add_argument("--report", default="data/processed/baseline_report.json")
    args = parser.parse_args()

    samples = [
        BehaviorSample(**item)
        for item in json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    ]
    dataset = BehaviorDataset(samples=samples)
    train, val, test = dataset.chronological_split()
    print(f"split (chronological 70/15/15): train={len(train)} val={len(val)} test={len(test)}")

    assembler = StateFeatureAssembler()
    models: list[tuple[str, BehaviorModel]] = [
        ("majority", BaselineBehaviorModel()),
        ("logistic_regression", LogisticRegressionModel(assembler)),
        ("random_forest", RandomForestModel(assembler)),
    ]

    results: dict[str, dict] = {}
    for name, model in models:
        model.fit(train)  # labels used here only
        val_report, val_trader = _collect(model, assembler, val)
        test_report, test_trader = _collect(model, assembler, test)
        worst_f1 = min(
            (m["macro_f1"] for m in test_trader.values() if m["samples"]),
            default=float("nan"),
        )
        results[name] = {
            "validation": {"global": val_report, "per_trader": val_trader},
            "test": {"global": test_report, "per_trader": test_trader, "worst_trader_macro_f1": worst_f1},
        }
        print(
            f"\n=== {name} === "
            f"val acc={val_report['accuracy']:.3f} macro_f1={val_report['macro_f1']:.3f} "
            f"log_loss={val_report['log_loss']:.3f} | "
            f"test acc={test_report['accuracy']:.3f} macro_f1={test_report['macro_f1']:.3f} "
            f"log_loss={test_report['log_loss']:.3f} worst_trader_f1={worst_f1:.3f}"
        )
        print("  test per-class:", json.dumps(test_report["per_class"], ensure_ascii=False))
        print("  test confusion:", json.dumps(test_report["confusion_matrix"], ensure_ascii=False))
        for trader, metrics in sorted(test_trader.items()):
            print(f"  trader {trader}: {metrics}")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
