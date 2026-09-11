"""P2-1 experiments: recency window (Phase A) + time decay (Phase B).

Every (fold, window, lambda) combination trains a FRESH Random Forest;
the walk-forward folds are the SAME blocks used by the existing evaluator
(shared via src/eval/walk_forward.py). All preprocessing (imputer, the
all-NaN feature decision) is fitted per fold-train only. Windows with
fewer than MIN_TRAIN_SAMPLES are SKIPPED, never padded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

# Make the repo root importable when run as `python scripts/run_recency_experiments.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.behavior_dataset import BehaviorSample  # noqa: E402
from src.eval.recency import (  # noqa: E402
    MIN_TRAIN_SAMPLES,
    WINDOWS,
    DecayRandomForestModel,
    filter_window,
)
from src.eval.walk_forward import walk_forward_fold_blocks  # noqa: E402
from src.features.state_features import StateFeatureAssembler  # noqa: E402
from src.models.training import evaluate_behavior_model  # noqa: E402


class CachedAssembler:
    def __init__(self, cache: dict[int, dict]):
        self.cache = cache

    def from_sample(self, sample: BehaviorSample) -> dict[str, float]:
        return self.cache[id(sample)]


def _log_loss_and_brier(model, assembler, samples) -> tuple[float, float]:
    losses, briers = [], []
    for sample in samples:
        prediction = model.predict_proba([assembler.from_sample(sample)])[0]
        label = sample.action  # read only after predicting
        p = prediction.long_probability if label == "LONG" else prediction.short_probability
        losses.append(-math.log(max(p, 1e-12)))
        briers.append((1.0 - p) ** 2)
    n = len(samples) or 1
    return sum(losses) / n, sum(briers) / n


def _fold_metrics(model, assembler, test_samples) -> dict:
    report = evaluate_behavior_model(model, test_samples, assembler).to_dict()
    log_loss, brier = _log_loss_and_brier(model, assembler, test_samples)
    per_trader = {}
    traders: dict[str, list] = {}
    for s in test_samples:
        traders.setdefault(s.kol, []).append(s)
    for trader, subset in sorted(traders.items()):
        per_trader[trader] = evaluate_behavior_model(model, subset, assembler).macro_f1
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro_f1"],
        "log_loss": log_loss,
        "brier": brier,
        "macro_trader_f1": sum(per_trader.values()) / len(per_trader) if per_trader else float("nan"),
        "worst_trader_f1": min(per_trader.values()) if per_trader else float("nan"),
        "long_recall": report["per_class"].get("LONG", {}).get("recall"),
        "short_recall": report["per_class"].get("SHORT", {}).get("recall"),
        "per_trader_f1": per_trader,
    }


def _aggregate(rows: list[dict]) -> dict:
    out = {}
    for metric in ("accuracy", "macro_f1", "log_loss", "brier", "macro_trader_f1",
                   "worst_trader_f1", "long_recall", "short_recall"):
        values = [r[metric] for r in rows if r.get(metric) == r.get(metric)]  # drop NaN
        if not values:
            out[metric] = {"mean": None, "std": None}
            continue
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        out[metric] = {"mean": mean, "std": math.sqrt(var)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/behavior_dataset_full.json")
    parser.add_argument("--outdir", default="data/experiments/p2_1_recency")
    args = parser.parse_args()

    import json as _json
    samples = [
        BehaviorSample(**item)
        for item in _json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    ]
    assembler = StateFeatureAssembler()
    cache = {id(s): assembler.from_sample(s) for s in samples}
    cached = CachedAssembler(cache)
    print(f"samples={len(samples)} | features per sample={len(cache[id(samples[0])])}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    folds = walk_forward_fold_blocks(samples, n_splits=4)
    print(f"folds: {len(folds)}")

    # ------------------------------------------------------------------
    # Phase A: window sweep, fresh RF per (fold, window)
    # ------------------------------------------------------------------
    phase_a_rows: list[dict] = []
    for fold in folds:
        for window in WINDOWS:
            train = filter_window(samples, fold.test_start, window)
            row = {
                "phase": "A",
                "fold": fold.fold_index,
                "window": window,
                "train_samples": len(train),
                "test_samples": len(fold.test_block),
                "trader_count": len({s.kol for s in train}),
                "train_range": [train[0].timestamp, train[-1].timestamp] if train else None,
                "test_range": [fold.test_start, fold.test_end],
            }
            if len(train) < MIN_TRAIN_SAMPLES:
                row.update(status="SKIPPED", reason="insufficient samples")
                phase_a_rows.append(row)
                continue
            model = DecayRandomForestModel(cached, n_estimators=300)  # fresh model
            model.fit(train)
            metrics = _fold_metrics(model, cached, fold.test_block)
            row.update(status="OK", **metrics)
            phase_a_rows.append(row)
            print(f"[A] fold {fold.fold_index} {window:<5} train={len(train):<3} "
                  f"test={len(fold.test_block)} f1={metrics['macro_f1']:.3f} "
                  f"ll={metrics['log_loss']:.3f}")

    window_rows = {
        window: [r for r in phase_a_rows if r["window"] == window and r.get("status") == "OK"]
        for window in WINDOWS
    }
    window_summary = {
        window: {"n_folds": len(rows), "aggregate": _aggregate(rows)}
        for window, rows in window_rows.items()
    }
    # champion window: highest mean macro_f1, tie-break lower mean log loss
    champion_window = min(
        (w for w in WINDOWS if window_rows[w]),
        key=lambda w: (
            -window_summary[w]["aggregate"]["macro_f1"]["mean"],
            window_summary[w]["aggregate"]["log_loss"]["mean"],
        ),
    )
    print(f"\nchampion window (by mean macro_f1, tie log_loss): {champion_window}")

    # per-trader f1 per window (mean across folds)
    trader_table: dict[str, dict] = {}
    for window, rows in window_rows.items():
        for trader in ("aoying_capital", "liuyuan", "xijiuye"):
            values = [r["per_trader_f1"][trader] for r in rows if trader in r["per_trader_f1"]]
            trader_table.setdefault(trader, {})[window] = (
                sum(values) / len(values) if values else None
            )

    # ------------------------------------------------------------------
    # Phase B: time decay on the champion window, fresh RF per (fold, lambda)
    # ------------------------------------------------------------------
    DECAY_LAMBDAS = (0.0, 0.005, 0.01, 0.02)
    phase_b_rows: list[dict] = []
    for fold in folds:
        train = filter_window(samples, fold.test_start, champion_window)
        for lam in DECAY_LAMBDAS:
            row = {
                "phase": "B",
                "fold": fold.fold_index,
                "window": champion_window,
                "lambda": lam,
                "train_samples": len(train),
                "test_samples": len(fold.test_block),
            }
            if len(train) < MIN_TRAIN_SAMPLES:
                row.update(status="SKIPPED", reason="insufficient samples")
                phase_b_rows.append(row)
                continue
            model = DecayRandomForestModel(
                cached, n_estimators=300, lambda_days=lam, cutoff=fold.test_start
            )  # fresh per (fold, lambda)
            model.fit(train)
            metrics = _fold_metrics(model, cached, fold.test_block)
            row.update(status="OK", **metrics)
            phase_b_rows.append(row)
            print(f"[B] fold {fold.fold_index} lam={lam:<5} f1={metrics['macro_f1']:.3f} "
                  f"ll={metrics['log_loss']:.3f}")

    decay_summary = {
        lam: {"n_folds": len([r for r in phase_b_rows if r["lambda"] == lam and r.get("status") == "OK"]),
              "aggregate": _aggregate([r for r in phase_b_rows if r["lambda"] == lam and r.get("status") == "OK"])}
        for lam in DECAY_LAMBDAS
    }

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    def write_csv(name: str, rows: list[dict]) -> None:
        if not rows:
            return
        keys = sorted({k for r in rows for k in r if k not in ("per_trader_f1", "train_range", "test_range")})
        with (outdir / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k) for k in keys})

    write_csv("fold_metrics.csv", phase_a_rows + phase_b_rows)
    write_csv("window_metrics.csv", [
        {"window": w, **window_summary[w]["aggregate"]["macro_f1"], "n_folds": window_summary[w]["n_folds"]}
        for w in WINDOWS if window_rows[w]
    ])
    write_csv("decay_metrics.csv", [
        {"lambda": lam, **decay_summary[lam]["aggregate"]["macro_f1"], "n_folds": decay_summary[lam]["n_folds"]}
        for lam in DECAY_LAMBDAS
    ])
    write_csv("trader_metrics.csv", [
        {"trader": trader, **{w: v for w, v in windows.items()}} for trader, windows in trader_table.items()
    ])

    summary = {
        "dataset_data_version": "0cb9dbfaa775b10b",
        "feature_version": "state-features-v2",
        "walk_forward_folds": 4,
        "min_train_samples": MIN_TRAIN_SAMPLES,
        "phase_a": {"window_summary": window_summary, "champion_window": champion_window},
        "phase_b": {"window": champion_window, "lambdas": DECAY_LAMBDAS, "decay_summary": decay_summary},
        "trader_table": trader_table,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outdir}/ (fold_metrics.csv, window_metrics.csv, decay_metrics.csv, trader_metrics.csv, summary.json)")


if __name__ == "__main__":
    main()
