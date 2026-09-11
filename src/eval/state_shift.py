"""Same-state / different-time behavior comparison (P2-1.5).

Method — fixed in advance, uses NO outcome data and NO future info:

1. Split samples into EARLY (first half by time) and LATE (second half).
2. Standardize the existing feature space (state-features-v2) using EARLY
   samples ONLY.
3. For every LATE sample, find its k nearest EARLY neighbors (Euclidean
   distance on the standardized features).
4. Compare the KOL's long-ratio on LATE samples against the long-ratio of
   their state-matched EARLY neighbors.

A systematic difference under similar states is DIRECTIONAL evidence of
behavior drift; with ~370 samples the default evidence strength is "weak".
"""

from __future__ import annotations

import math
from typing import Optional

from src.dataset.behavior_dataset import BehaviorSample

EARLY_RATIO = 0.5
K_NEIGHBORS = 5


def split_early_late(
    samples: list[BehaviorSample],
) -> tuple[list[BehaviorSample], list[BehaviorSample]]:
    ordered = sorted(samples, key=lambda s: s.timestamp)
    cut = int(len(ordered) * EARLY_RATIO)
    return ordered[:cut], ordered[cut:]


def _standardized(features: list[dict], stats: Optional[dict] = None) -> tuple[list[list[float]], dict]:
    keys = sorted({k for f in features for k in f})
    matrix = [[float(f.get(k, float("nan"))) for k in keys] for f in features]
    if stats is None:
        stats = {}
        for j, key in enumerate(keys):
            values = [row[j] for row in matrix if row[j] == row[j]]
            mean = sum(values) / len(values) if values else 0.0
            var = sum((v - mean) ** 2 for v in values) / len(values) if values else 0.0
            stats[key] = {"mean": mean, "std": math.sqrt(var)}

    def _z(v: float, s: dict) -> float:
        if v != v:  # missing feature -> neutral zero in standardized space
            return 0.0
        return (v - s["mean"]) / s["std"] if s["std"] > 1e-12 else 0.0

    return [[_z(v, stats[k]) for k, v in zip(keys, row)] for row in matrix], stats


def _euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def state_shift_analysis(
    samples: list[BehaviorSample],
    features_cache: dict[int, dict],
    trader: Optional[str] = None,
) -> dict:
    """Directional drift evidence via state-matched early/late comparison."""
    subset = [s for s in samples if trader is None or s.kol == trader]
    early, late = split_early_late(subset)
    if len(early) < K_NEIGHBORS or not late:
        return {"trader": trader or "all", "status": "INSUFFICIENT", "early_samples": len(early),
                "late_samples": len(late)}

    early_features = [features_cache[id(s)] for s in early]
    late_features = [features_cache[id(s)] for s in late]
    early_matrix, stats = _standardized(early_features)
    late_matrix, _ = _standardized(late_features, stats)

    early_long = [1.0 if s.action == "LONG" else 0.0 for s in early]
    late_long = [1.0 if s.action == "LONG" else 0.0 for s in late]

    # state-matched comparison: for each late sample, long-ratio of its
    # k nearest EARLY neighbors vs its own action
    buckets: dict[str, dict] = {
        "neighbor_long_heavy": {"n": 0, "neighbor_long_ratio": [], "late_long_ratio": []},
        "neighbor_mixed": {"n": 0, "neighbor_long_ratio": [], "late_long_ratio": []},
        "neighbor_short_heavy": {"n": 0, "neighbor_long_ratio": [], "late_long_ratio": []},
    }
    for late_vec, actual_long in zip(late_matrix, late_long):
        distances = sorted(
            ((_euclidean(late_vec, ev), ev_long) for ev, ev_long in zip(early_matrix, early_long)),
            key=lambda t: t[0],
        )[:K_NEIGHBORS]
        neighbor_ratio = sum(ev_long for _, ev_long in distances) / K_NEIGHBORS
        if neighbor_ratio > 0.6:
            bucket = "neighbor_long_heavy"
        elif neighbor_ratio < 0.4:
            bucket = "neighbor_short_heavy"
        else:
            bucket = "neighbor_mixed"
        buckets[bucket]["n"] += 1
        buckets[bucket]["neighbor_long_ratio"].append(neighbor_ratio)
        buckets[bucket]["late_long_ratio"].append(actual_long)

    rows = []
    for name, bucket in buckets.items():
        if bucket["n"] == 0:
            continue
        rows.append(
            {
                "state_cluster": name,
                "time_period": "late",
                "trader": trader or "all",
                "sample_count": bucket["n"],
                "neighbor_long_ratio": sum(bucket["neighbor_long_ratio"]) / bucket["n"],
                "late_long_ratio": sum(bucket["late_long_ratio"]) / bucket["n"],
                "delta": sum(bucket["late_long_ratio"]) / bucket["n"]
                - sum(bucket["neighbor_long_ratio"]) / bucket["n"],
            }
        )

    overall_early = sum(early_long) / len(early_long)
    overall_late = sum(late_long) / len(late_long)
    # state-matched aggregate: mean neighbor ratio vs mean late ratio
    matched_ratios = [
        sum(bucket["neighbor_long_ratio"]) / max(bucket["n"], 1) for bucket in buckets.values() if bucket["n"]
    ]
    matched_actuals = [
        sum(bucket["late_long_ratio"]) / max(bucket["n"], 1) for bucket in buckets.values() if bucket["n"]
    ]
    return {
        "trader": trader or "all",
        "status": "OK",
        "early_samples": len(early),
        "late_samples": len(late),
        "early_long_ratio": overall_early,
        "late_long_ratio": overall_late,
        "state_matched_neighbor_long_ratio": sum(matched_ratios) / len(matched_ratios) if matched_ratios else None,
        "state_matched_late_long_ratio": sum(matched_actuals) / len(matched_actuals) if matched_actuals else None,
        "rows": rows,
    }
