"""P2-1.5 evaluation redesign: multiple test windows, market regimes,
same-state/different-time drift evidence.

Reuses the existing walk-forward blocks, leakage detector and RF baseline.
Every window trains a FRESH RF on samples strictly before the window
start; predictions are recorded before test labels are read. Regime and
state-similarity analyses consume only T-visible features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# Make the repo root importable when run as `python scripts/run_evaluation_redesign.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.behavior_dataset import BehaviorSample  # noqa: E402
from src.data.leakage_detector import scan_dataset_json  # noqa: E402
from src.eval.regime import TREND_VALUES, VOL_VALUES, label_regimes  # noqa: E402
from src.eval.state_shift import state_shift_analysis  # noqa: E402
from src.eval.test_windows import generate_test_windows  # noqa: E402
from src.eval.walk_forward import aggregate_folds, expanding_walk_forward  # noqa: E402
from src.features.state_features import StateFeatureAssembler  # noqa: E402
from src.models.sklearn_models import RandomForestModel  # noqa: E402
from src.models.training import evaluate_behavior_model  # noqa: E402


class CachedAssembler:
    def __init__(self, cache):
        self.cache = cache

    def from_sample(self, sample):
        return self.cache[id(sample)]


def _loss_and_brier(model, assembler, samples):
    losses, briers = [], []
    for sample in samples:
        prediction = model.predict_proba([assembler.from_sample(sample)])[0]
        label = sample.action  # read only after predicting
        p = prediction.long_probability if label == "LONG" else prediction.short_probability
        losses.append(-math.log(max(p, 1e-12)))
        briers.append((1.0 - p) ** 2)
    n = len(samples) or 1
    return sum(losses) / n, sum(briers) / n


def _window_metrics(model, assembler, test_samples) -> dict:
    report = evaluate_behavior_model(model, test_samples, assembler)
    log_loss, brier = _loss_and_brier(model, assembler, test_samples)
    per_trader = {}
    traders: dict[str, list] = defaultdict(list)
    for s in test_samples:
        traders[s.kol].append(s)
    for trader, subset in sorted(traders.items()):
        per_trader[trader] = evaluate_behavior_model(model, subset, assembler).macro_f1
    return {
        "accuracy": report.accuracy,
        "macro_f1": report.macro_f1,
        "logloss": log_loss,
        "brier": brier,
        "long_recall": report.per_class.get("LONG", {}).get("recall"),
        "short_recall": report.per_class.get("SHORT", {}).get("recall"),
        "worst_trader_f1": min(per_trader.values()) if per_trader else float("nan"),
        "per_trader_f1": per_trader,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/behavior_dataset_full.json")
    parser.add_argument("--outdir", default="data/experiments/p2_1_5_evaluation")
    args = parser.parse_args()

    # PIT gate
    report = scan_dataset_json(args.dataset)
    assert report.ok, json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    print("leakage detector: PASS")

    samples = [
        BehaviorSample(**item)
        for item in json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    ]
    assembler = StateFeatureAssembler()
    cache = {id(s): assembler.from_sample(s) for s in samples}
    cached = CachedAssembler(cache)
    print(f"samples={len(samples)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    # ------------------------------------------------------------------
    # 1. Official walk-forward (kept as the temporal generalization benchmark)
    # ------------------------------------------------------------------
    folds = expanding_walk_forward(
        samples, lambda: RandomForestModel(cached, n_estimators=300), cached, n_splits=4
    )
    wf_aggregate = aggregate_folds(folds)
    summary["walk_forward"] = {
        "folds": [
            {
                "fold": f.fold_index,
                "train_range": [f.train_start, f.train_end],
                "test_range": [f.test_start, f.test_end],
                "n_train": f.n_train,
                "n_test": f.n_test,
                "accuracy": f.metrics["accuracy"],
                "macro_f1": f.metrics["macro_f1"],
                "log_loss": f.metrics["log_loss"],
            }
            for f in folds
        ],
        "aggregate": wf_aggregate,
    }
    print("walk-forward aggregate f1:", round(wf_aggregate["macro_f1"]["mean"], 3))

    # ------------------------------------------------------------------
    # 2. Multiple test windows (auto-generated; fresh RF per window)
    # ------------------------------------------------------------------
    windows = generate_test_windows(samples, n_windows=4)
    window_rows = []
    per_test_predictions: dict[int, tuple[str, float]] = {}  # id -> (actual, pred_long)
    for window in windows:
        row = {
            "window_index": window.index,
            "window_start": window.start,
            "window_end": window.end,
            "train_samples": len(window.train_samples),
            "test_samples": len(window.test_samples),
            "trader_count": len({s.kol for s in window.test_samples}),
            "long_count": sum(1 for s in window.test_samples if s.action == "LONG"),
            "short_count": sum(1 for s in window.test_samples if s.action == "SHORT"),
        }
        if window.skipped:
            row.update(status="SKIPPED", reason=window.skip_reason)
            window_rows.append(row)
            continue
        model = RandomForestModel(cached, n_estimators=300)  # fresh per window
        model.fit(window.train_samples)
        metrics = _window_metrics(model, cached, window.test_samples)
        for sample in window.test_samples:
            prediction = model.predict_proba([cached.from_sample(sample)])[0]
            per_test_predictions[id(sample)] = (sample.action, prediction.long_probability)
        row.update(status="OK", **{k: v for k, v in metrics.items() if k != "per_trader_f1"})
        window_rows.append(row)
        print(f"window {window.index}: {window.start[:10]}..{window.end[:10]} "
              f"train={len(window.train_samples)} test={len(window.test_samples)} "
              f"acc={metrics['accuracy']:.3f} f1={metrics['macro_f1']:.3f}")

    summary["test_windows"] = window_rows
    with (outdir / "window_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        keys = sorted({k for r in window_rows for k in r})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in window_rows:
            writer.writerow({k: r.get(k) for k in keys})

    # per-trader / per-action per window
    trader_rows, action_rows = [], []
    for window in windows:
        if window.skipped:
            continue
        model = RandomForestModel(cached, n_estimators=300)
        model.fit(window.train_samples)
        metrics = _window_metrics(model, cached, window.test_samples)
        for trader, f1 in sorted(metrics["per_trader_f1"].items()):
            trader_rows.append({"window": window.index, "trader": trader, "macro_f1": f1})
        report = evaluate_behavior_model(model, window.test_samples, cached)
        for action in ("LONG", "SHORT"):
            action_rows.append({
                "window": window.index, "action": action,
                "recall": report.per_class.get(action, {}).get("recall"),
                "precision": report.per_class.get(action, {}).get("precision"),
                "support": report.per_class.get(action, {}).get("support"),
            })
    for name, rows in (("window_trader_metrics.csv", trader_rows), ("window_action_metrics.csv", action_rows)):
        with (outdir / name).open("w", newline="", encoding="utf-8") as f:
            keys = sorted({k for r in rows for k in r})
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k) for k in keys})

    # ------------------------------------------------------------------
    # 3. Market regime evaluation (labels + actual vs predicted distributions)
    # ------------------------------------------------------------------
    regimes = label_regimes(samples, cache)
    regime_rows = []
    for family, values in (("trend", TREND_VALUES), ("volatility", VOL_VALUES)):
        for value in values:
            group = [s for s in samples if regimes[id(s)][family] == value]
            predicted = [per_test_predictions[id(s)][1] for s in group if id(s) in per_test_predictions]
            actual_long = sum(1 for s in group if s.action == "LONG") / len(group) if group else None
            mean_pred = sum(predicted) / len(predicted) if predicted else None
            regime_rows.append({
                "family": family,
                "regime": value,
                "sample_count": len(group),
                "actual_long_ratio": actual_long,
                "mean_predicted_long": mean_pred,
                "gap": (mean_pred - actual_long) if (mean_pred is not None and actual_long is not None) else None,
                "trader_long_ratios": {
                    trader: (sum(1 for s in group if s.kol == trader and s.action == "LONG")
                             / max(1, sum(1 for s in group if s.kol == trader)))
                    for trader in sorted({s.kol for s in group})
                },
            })
    summary["regimes"] = {
        "rows": [{k: v for k, v in r.items() if k != "trader_long_ratios"} for r in regime_rows],
        "trader_long_ratios_by_regime": [
            {"family": r["family"], "regime": r["regime"], "trader_long_ratios": r["trader_long_ratios"]}
            for r in regime_rows
        ],
    }
    with (outdir / "regime_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        keys = ["family", "regime", "sample_count", "actual_long_ratio", "mean_predicted_long", "gap"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in regime_rows:
            writer.writerow({k: r[k] for k in keys})
    for r in regime_rows:
        print(f"regime {r['family']}/{r['regime']}: n={r['sample_count']} "
              f"actual_long={r['actual_long_ratio']} pred_long={r['mean_predicted_long']}")

    # ------------------------------------------------------------------
    # 4. Same-state / different-time drift evidence
    # ------------------------------------------------------------------
    shift_rows = []
    for trader in (None, "aoying_capital", "liuyuan", "xijiuye"):
        result = state_shift_analysis(samples, cache, trader=trader)
        if result.get("status") != "OK":
            continue
        summary_shift = {k: v for k, v in result.items() if k != "rows"}
        summary.setdefault("state_shift", []).append(summary_shift)
        shift_rows.extend(result["rows"])
        print(f"shift[{result['trader']}]: early_long={result['early_long_ratio']:.3f} "
              f"late_long={result['late_long_ratio']:.3f} "
              f"matched_neighbor={result['state_matched_neighbor_long_ratio']} "
              f"matched_late={result['state_matched_late_long_ratio']}")
    with (outdir / "state_shift_analysis.csv").open("w", newline="", encoding="utf-8") as f:
        keys = ["state_cluster", "time_period", "trader", "sample_count",
                "neighbor_long_ratio", "late_long_ratio", "delta"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in shift_rows:
            writer.writerow({k: r.get(k) for k in keys})

    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outdir}/ (window_metrics, window_trader_metrics, window_action_metrics, regime_metrics, state_shift_analysis, summary)")


if __name__ == "__main__":
    main()
