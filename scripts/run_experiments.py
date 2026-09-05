"""Stage P1.5 experiments: strong traditional-ML baselines.

- Walk-forward (same expanding folds for every model): majority, LR,
  LR-balanced, RF, LightGBM, LightGBM-balanced, XGBoost, XGBoost-balanced.
  Class-weight variants are EXPLICIT separate experiments, never silent
  changes to a default model.
- Feature ablation on LR (with StandardScaler), identical global split.
- Feature importance: LR standardized coefficients; RF/LGBM/XGB
  train-fitted importances (analysis only, never used for selection).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Make the repo root importable when run as `python scripts/run_experiments.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.behavior_dataset import BehaviorDataset, BehaviorSample  # noqa: E402
from src.eval.ablation import FEATURE_SETS, filter_features  # noqa: E402
from src.eval.walk_forward import aggregate_folds, expanding_walk_forward  # noqa: E402
from src.features.state_features import StateFeatureAssembler  # noqa: E402
from src.models.behavior_model import BaselineBehaviorModel  # noqa: E402
from src.models.sklearn_models import (  # noqa: E402
    LightGBMModel,
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
)
from src.models.training import evaluate_behavior_model  # noqa: E402


class CachedAssembler:
    def __init__(self, cache: dict[int, dict], feature_set: str = "full"):
        self.cache = cache
        self.feature_set = feature_set

    def from_sample(self, sample: BehaviorSample) -> dict[str, float]:
        return filter_features(self.cache[id(sample)], self.feature_set)


def _log_loss_from_proba(model, assembler, samples) -> float:
    losses = []
    for sample in samples:
        prediction = model.predict_proba([assembler.from_sample(sample)])[0]
        label = sample.action
        p = prediction.long_probability if label == "LONG" else prediction.short_probability
        losses.append(-math.log(max(p, 1e-12)))
    return sum(losses) / len(losses) if losses else float("nan")


def _full_metrics(model, assembler, samples) -> dict:
    report = evaluate_behavior_model(model, samples, assembler).to_dict()
    report["log_loss"] = _log_loss_from_proba(model, assembler, samples)
    per_trader = {}
    traders: dict[str, list] = {}
    for s in samples:
        traders.setdefault(s.kol, []).append(s)
    for trader, subset in sorted(traders.items()):
        per_trader[trader] = evaluate_behavior_model(model, subset, assembler).macro_f1
    report["per_trader_macro_f1"] = per_trader
    report["worst_trader_macro_f1"] = min(per_trader.values()) if per_trader else float("nan")
    return report


def _compact(report: dict) -> dict:
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro_f1"],
        "log_loss": report["log_loss"],
        "long_recall": report["per_class"].get("LONG", {}).get("recall"),
        "short_recall": report["per_class"].get("SHORT", {}).get("recall"),
        "worst_trader_macro_f1": report["worst_trader_macro_f1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/behavior_dataset_full.json")
    parser.add_argument("--report", default="data/processed/experiments_report.json")
    args = parser.parse_args()

    samples = [
        BehaviorSample(**item)
        for item in json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    ]
    dataset = BehaviorDataset(samples=samples)
    train, val, test = dataset.chronological_split()
    print(f"global split: train={len(train)} val={len(val)} test={len(test)}")

    base_assembler = StateFeatureAssembler()
    cache: dict[int, dict] = {}
    for sample in samples:
        cache[id(sample)] = base_assembler.from_sample(sample)
    full = CachedAssembler(cache, "full")
    if samples:
        print(f"precomputed features: {len(cache)} samples, {len(cache[id(samples[0])])} keys")

    output: dict = {}

    # ------------------------------------------------------------------
    # 1. Walk-forward: same folds for every model config
    # ------------------------------------------------------------------
    print("\n=== WALK-FORWARD (expanding window, n_splits=4) ===")
    model_factories = {
        "majority": lambda: BaselineBehaviorModel(),
        "lr": lambda: LogisticRegressionModel(full),
        "lr_balanced": lambda: LogisticRegressionModel(full, class_weight="balanced"),
        "rf": lambda: RandomForestModel(full, n_estimators=300),
        "lightgbm": lambda: LightGBMModel(full, n_estimators=300),
        "lightgbm_balanced": lambda: LightGBMModel(full, n_estimators=300, class_weight="balanced"),
        "xgboost": lambda: XGBoostModel(full, n_estimators=300),
        "xgboost_balanced": lambda: XGBoostModel(full, n_estimators=300, class_weight="balanced"),
    }
    output["walk_forward"] = {}
    for name, factory in model_factories.items():
        folds = expanding_walk_forward(samples, factory, full, n_splits=4)
        for fold in folds:
            c = _compact(fold.metrics)
            print(
                f"[{name}] fold {fold.fold_index}: acc={c['accuracy']:.3f} f1={c['macro_f1']:.3f} "
                f"ll={c['log_loss']:.3f} Lrecall={c['long_recall'] or 0:.3f} Srecall={c['short_recall'] or 0:.3f}"
            )
        aggregation = aggregate_folds(folds)
        output["walk_forward"][name] = {
            "folds": [
                {
                    "fold_index": f.fold_index,
                    "train_range": [f.train_start, f.train_end],
                    "test_range": [f.test_start, f.test_end],
                    "n_train": f.n_train,
                    "n_test": f.n_test,
                    **_compact(f.metrics),
                    "per_trader_macro_f1": f.metrics["per_trader_macro_f1"],
                }
                for f in folds
            ],
            "aggregate": aggregation,
        }
        agg = aggregation
        print(f"[{name}] AGGREGATE: acc {agg['accuracy']['mean']:.3f}±{agg['accuracy']['std']:.3f} | "
              f"f1 {agg['macro_f1']['mean']:.3f}±{agg['macro_f1']['std']:.3f} | "
              f"ll {agg['log_loss']['mean']:.3f} | worstF1 {agg['worst_trader_macro_f1']['mean']:.3f}")

    # ------------------------------------------------------------------
    # 2. Feature ablation (LR with scaler, identical global split)
    # ------------------------------------------------------------------
    print("\n=== FEATURE ABLATION (LR + StandardScaler, global split) ===")
    output["ablation"] = {}
    print(f"{'Feature Set':<26} {'Acc':>6} {'MacroF1':>8} {'LogLoss':>8} {'L Recall':>9} {'S Recall':>9} {'WorstTF1':>9}")
    for feature_set in FEATURE_SETS:
        assembler = CachedAssembler(cache, feature_set)
        model = LogisticRegressionModel(assembler)
        model.fit(train)
        compact = _compact(_full_metrics(model, assembler, test))
        output["ablation"][feature_set] = {
            "n_features": len(assembler.from_sample(test[0])),
            "n_dropped_all_nan": len(model._all_nan_keys),
            **compact,
        }
        print(
            f"{feature_set:<26} {compact['accuracy']:>6.3f} {compact['macro_f1']:>8.3f} "
            f"{compact['log_loss']:>8.3f} {compact['long_recall'] or 0:>9.3f} "
            f"{compact['short_recall'] or 0:>9.3f} {compact['worst_trader_macro_f1']:>9.3f}"
        )

    # ------------------------------------------------------------------
    # 3. Class weight (global split, explicit separate experiments)
    # ------------------------------------------------------------------
    print("\n=== CLASS WEIGHT (global split, explicit experiments) ===")
    output["class_weight"] = {}
    configs = [
        ("lr_default", LogisticRegressionModel(full)),
        ("lr_balanced", LogisticRegressionModel(full, class_weight="balanced")),
        ("lightgbm_default", LightGBMModel(full)),
        ("lightgbm_balanced", LightGBMModel(full, class_weight="balanced")),
        ("xgboost_default", XGBoostModel(full)),
        ("xgboost_balanced", XGBoostModel(full, class_weight="balanced")),
        ("rf_default", RandomForestModel(full)),
        ("rf_balanced", RandomForestModel(full, class_weight="balanced")),
    ]
    for name, model in configs:
        model.fit(train)
        compact = _compact(_full_metrics(model, full, test))
        output["class_weight"][name] = compact
        print(f"{name:<20} acc={compact['accuracy']:.3f} f1={compact['macro_f1']:.3f} "
              f"ll={compact['log_loss']:.3f} Lrecall={compact['long_recall'] or 0:.3f} "
              f"Srecall={compact['short_recall'] or 0:.3f}")

    # ------------------------------------------------------------------
    # 4. Feature importance (train-fitted only; analysis, not selection)
    # ------------------------------------------------------------------
    print("\n=== LR TOP COEFFICIENTS (standardized, train only) ===")
    lr_model = LogisticRegressionModel(full)
    lr_model.fit(train)
    keys = lr_model._feature_keys
    coefs = lr_model.estimator.named_steps["model"].coef_[0]
    ranked = sorted(zip(keys, coefs), key=lambda kv: -abs(kv[1]))[:30]
    output["lr_top_coefficients"] = [
        {"feature": k, "coefficient": c, "direction": "SHORT" if c > 0 else "LONG"}
        for k, c in ranked
    ]
    for entry in output["lr_top_coefficients"][:15]:
        print(f"  {entry['direction']:<5} {entry['coefficient']:>+10.4f}  {entry['feature']}")

    print("\n=== TREE IMPORTANCE (train only) ===")
    output["tree_importance"] = {}
    for name, model in [
        ("rf", RandomForestModel(full)),
        ("lightgbm", LightGBMModel(full)),
        ("xgboost", XGBoostModel(full)),
    ]:
        model.fit(train)
        importance = model.estimator.named_steps["model"].feature_importances_
        keys = model._feature_keys
        top = sorted(zip(keys, importance), key=lambda kv: -kv[1])[:15]
        output["tree_importance"][name] = [
            {"feature": k, "importance": float(v)} for k, v in top
        ]
        print(f"[{name}] top15: " + ", ".join(f"{k.split('__')[-1]}" for k, _ in top[:8]) + " ...")

    # ------------------------------------------------------------------
    # 5. Unified ranking: WF champion + global test champion
    # ------------------------------------------------------------------
    wf_rows = []
    for name, entry in output["walk_forward"].items():
        for fold in entry["folds"]:
            wf_rows.append({"model": name, **fold})
    wf_aggregates = {
        name: {
            "mean_accuracy": e["aggregate"]["accuracy"]["mean"],
            "mean_macro_f1": e["aggregate"]["macro_f1"]["mean"],
            "std_macro_f1": e["aggregate"]["macro_f1"]["std"],
            "mean_log_loss": e["aggregate"]["log_loss"]["mean"],
        }
        for name, e in output["walk_forward"].items()
    }
    # champions: best mean macro_f1, tie-break lower log loss
    wf_champion = min(
        wf_aggregates, key=lambda n: (-wf_aggregates[n]["mean_macro_f1"], wf_aggregates[n]["mean_log_loss"])
    )
    test_champion = min(
        output["class_weight"], key=lambda n: (-output["class_weight"][n]["macro_f1"], output["class_weight"][n]["log_loss"])
    )
    output["champions"] = {
        "walk_forward_champion": wf_champion,
        "walk_forward_champion_metrics": wf_aggregates[wf_champion],
        "global_test_champion": test_champion,
        "global_test_champion_metrics": output["class_weight"][test_champion],
    }
    print(f"\n=== CHAMPIONS ===")
    print(f"walk_forward_champion: {wf_champion} {wf_aggregates[wf_champion]}")
    print(f"global_test_champion:  {test_champion} {output['class_weight'][test_champion]}")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
