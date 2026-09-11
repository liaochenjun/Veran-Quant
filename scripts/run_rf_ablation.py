"""P2-0: Random Forest feature ablation.

Strictly reuses the P1.5 framework (same data, same split, same
walk-forward folds, same RF parameters n_estimators=300 / default
weights). Only the feature subset varies.

Feature sets (per P2-0 spec):
  A market_only            market base features
  B market_geometry        market incl. per-tf geometry
  C market_chan            market base + chan
  D market_chan_geometry   market incl. geometry + chan
  E full_no_trader         == D (explicit alias, no trader identity)
  F full                   E + trader one-hots

Outputs data/processed/p2_rf_feature_ablation.json; the report is
written to docs/p2-rf-feature-ablation-report.md from these numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Make the repo root importable when run as `python scripts/run_rf_ablation.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.dataset.behavior_dataset import BehaviorDataset, BehaviorSample  # noqa: E402
from src.data.leakage_detector import scan_dataset_json  # noqa: E402
from src.eval.ablation import filter_features  # noqa: E402
from src.eval.walk_forward import aggregate_folds, expanding_walk_forward  # noqa: E402
from src.features.state_features import FEATURE_VERSION, StateFeatureAssembler  # noqa: E402
from src.models.sklearn_models import RandomForestModel  # noqa: E402
from src.models.training import evaluate_behavior_model  # noqa: E402

FEATURE_SETS = (
    "market_only",
    "market_geometry",
    "market_chan",
    "market_chan_geometry",
    "full_no_trader",
    "full",
)


class CachedAssembler:
    def __init__(self, cache: dict[int, dict], feature_set: str):
        self.cache = cache
        self.feature_set = feature_set

    def from_sample(self, sample: BehaviorSample) -> dict[str, float]:
        return filter_features(self.cache[id(sample)], self.feature_set)


def _log_loss(model, assembler, samples) -> float:
    losses = []
    for sample in samples:
        prediction = model.predict_proba([assembler.from_sample(sample)])[0]
        label = sample.action
        p = prediction.long_probability if label == "LONG" else prediction.short_probability
        losses.append(-math.log(max(p, 1e-12)))
    return sum(losses) / len(losses) if losses else float("nan")


def _full_metrics(model, assembler, samples) -> dict:
    report = evaluate_behavior_model(model, samples, assembler).to_dict()
    report["log_loss"] = _log_loss(model, assembler, samples)
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


def _delta(other: dict, base: dict) -> dict:
    return {k: other[k] - base[k] for k in base if isinstance(base[k], (int, float))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/behavior_dataset_full.json")
    parser.add_argument("--output", default="data/processed/p2_rf_feature_ablation.json")
    args = parser.parse_args()

    # PIT gate: leakage scan must pass before any experiment runs.
    report = scan_dataset_json(args.dataset)
    assert report.ok, json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    print("leakage detector: PASS")

    samples = [
        BehaviorSample(**item)
        for item in json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    ]
    dataset = BehaviorDataset(samples=samples)
    train, val, test = dataset.chronological_split()
    meta = json.loads(Path("data/processed/behavior_dataset_full.meta.json").read_text(encoding="utf-8"))
    print(f"split: train={len(train)} val={len(val)} test={len(test)} | "
          f"data_version={meta['data_version']} feature_version={FEATURE_VERSION}")

    base_assembler = StateFeatureAssembler()
    cache = {id(s): base_assembler.from_sample(s) for s in samples}

    experiments = {}
    for feature_set in FEATURE_SETS:
        assembler = CachedAssembler(cache, feature_set)
        n_features = len(assembler.from_sample(train[0]))

        # Global 70/15/15
        model = RandomForestModel(assembler, n_estimators=300)  # P1.5 params, unchanged
        model.fit(train)
        val_metrics = _compact(_full_metrics(model, assembler, val))
        test_metrics = _compact(_full_metrics(model, assembler, test))
        dropped = list(model._all_nan_keys)

        # Walk-Forward, same folds as P1.5
        folds = expanding_walk_forward(
            samples,
            lambda: RandomForestModel(assembler, n_estimators=300),
            assembler,
            n_splits=4,
        )
        fold_metrics = [_compact(f.metrics) for f in folds]
        aggregate = aggregate_folds(folds)

        experiments[feature_set] = {
            "n_features": n_features,
            "n_dropped_all_nan": len(dropped),
            "global": {
                "validation": val_metrics,
                "test": test_metrics,
            },
            "walk_forward": {
                "folds": [
                    {
                        "fold_index": f.fold_index,
                        "train_range": [f.train_start, f.train_end],
                        "test_range": [f.test_start, f.test_end],
                        "n_train": f.n_train,
                        "n_test": f.n_test,
                        **_compact(f.metrics),
                    }
                    for f in folds
                ],
                "aggregate": aggregate,
            },
        }
        print(
            f"{feature_set:<22} n={n_features:<3} | global test acc={test_metrics['accuracy']:.3f} "
            f"f1={test_metrics['macro_f1']:.3f} ll={test_metrics['log_loss']:.3f} | "
            f"WF mean f1={aggregate['macro_f1']['mean']:.3f} ll={aggregate['log_loss']['mean']:.3f}"
        )

    # ------------------------------------------------------------------
    # Feature deltas vs market_only (per fold + mean)
    # ------------------------------------------------------------------
    base = experiments["market_only"]
    feature_deltas = {}
    for feature_set in FEATURE_SETS:
        if feature_set == "market_only":
            continue
        per_fold = [
            _delta(fold, base_fold)
            for fold, base_fold in zip(
                experiments[feature_set]["walk_forward"]["folds"],
                base["walk_forward"]["folds"],
            )
        ]
        mean_deltas = {
            metric: sum(d[metric] for d in per_fold) / len(per_fold)
            for metric in ("accuracy", "macro_f1", "log_loss", "short_recall", "worst_trader_macro_f1")
        }
        feature_deltas[feature_set] = {"per_fold": per_fold, "mean": mean_deltas}
        print(f"\nΔ vs market_only [{feature_set}]:")
        for metric in ("accuracy", "macro_f1", "log_loss"):
            signs = [f"{d[metric]:+.3f}" for d in per_fold]
            print(f"  {metric:<8} mean {mean_deltas[metric]:+.3f} | folds {signs}")

    # ------------------------------------------------------------------
    # Feature importance (full RF, train-only) + category counts
    # ------------------------------------------------------------------
    full_assembler = CachedAssembler(cache, "full")
    full_model = RandomForestModel(full_assembler, n_estimators=300)
    full_model.fit(train)
    importances = full_model.estimator.named_steps["model"].feature_importances_
    ranked = sorted(zip(full_model._feature_keys, importances), key=lambda kv: -kv[1])
    top30 = [{"feature": k, "importance": float(v)} for k, v in ranked[:30]]
    category_counts = {"market": 0, "geometry": 0, "chan": 0, "trader": 0}
    for k, _ in ranked:
        if k.startswith("trader__"):
            category_counts["trader"] += 1
        elif k.startswith("chan__"):
            category_counts["chan"] += 1
        elif "__geometry__" in k:
            category_counts["geometry"] += 1
        else:
            category_counts["market"] += 1
    print(f"\nfull RF importance category counts (train-only): {category_counts}")
    print("top10:", ", ".join(k.split('__')[-1] for k, _ in ranked[:10]))

    # ------------------------------------------------------------------
    # Feature quality audit (train-based only; report, never delete)
    # ------------------------------------------------------------------
    train_features = [full_assembler.from_sample(s) for s in train]
    keys = full_model._feature_keys
    matrix = np.array(
        [[float(f.get(k, float("nan"))) for k in keys] for f in train_features]
    )
    finite = np.isfinite(matrix)
    constant_keys = [
        k for i, k in enumerate(keys)
        if finite[:, i].all() and np.allclose(matrix[:, i], matrix[0, i])
    ]
    near_constant_keys = [
        k for i, k in enumerate(keys)
        if finite[:, i].sum() >= 0.95 * len(train)
        and np.nanstd(matrix[:, i]) < 1e-9 and k not in constant_keys
    ]
    audit = {
        "all_nan_keys": list(full_model._all_nan_keys),
        "constant_keys": constant_keys,
        "near_constant_keys": near_constant_keys,
        "high_correlation_pairs": [],
    }
    cols = {}
    for i, k in enumerate(keys):
        col = matrix[:, i]
        mask = np.isfinite(col)
        if mask.sum() >= 2:
            cols[k] = col[mask]
    pairs = []
    for a in cols:
        for b in cols:
            if a >= b:
                continue
            # same-sample correlation requires aligned masks
            mask = np.isfinite(matrix[:, keys.index(a)]) & np.isfinite(matrix[:, keys.index(b)])
            if mask.sum() < 10:
                continue
            r = float(np.corrcoef(matrix[mask, keys.index(a)], matrix[mask, keys.index(b)])[0, 1])
            if abs(r) > 0.95:
                pairs.append((abs(r), a, b))
    audit["high_correlation_pairs"] = [
        {"corr": round(r, 3), "a": a, "b": b} for r, a, b in sorted(pairs, reverse=True)[:20]
    ]
    print(f"\naudit: all_nan={len(audit['all_nan_keys'])} constant={len(constant_keys)} "
          f"near_constant={len(near_constant_keys)} high_corr_pairs={len(pairs)}")
    for r, a, b in sorted(pairs, reverse=True)[:5]:
        print(f"  corr {r:.3f}: {a} <-> {b}")

    # ------------------------------------------------------------------
    # Champion: WF mean macro_f1 first, tie-break mean log loss
    # ------------------------------------------------------------------
    champion = min(
        FEATURE_SETS,
        key=lambda s: (
            -experiments[s]["walk_forward"]["aggregate"]["macro_f1"]["mean"],
            experiments[s]["walk_forward"]["aggregate"]["log_loss"]["mean"],
        ),
    )
    output = {
        "data_version": meta["data_version"],
        "feature_version": FEATURE_VERSION,
        "split": "chronological 70/15/15 (259/55/56)",
        "walk_forward_folds": 4,
        "experiments": experiments,
        "feature_deltas": feature_deltas,
        "champion": {
            "feature_set": champion,
            "walk_forward_mean_macro_f1": experiments[champion]["walk_forward"]["aggregate"]["macro_f1"]["mean"],
            "walk_forward_mean_log_loss": experiments[champion]["walk_forward"]["aggregate"]["log_loss"]["mean"],
        },
        "feature_importance": {"top30": top30, "category_counts": category_counts},
        "feature_quality_audit": audit,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRF feature-ablation champion (WF): {champion}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
