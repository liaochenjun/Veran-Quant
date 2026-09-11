"""Regenerate dataset metadata with explicit time-semantics fields.

Time semantics (business-critical, confirmed with the data owner):
- kol_timestamp_semantics: Asia/Shanghai (platform dumps)
- internal_timestamp_semantics: UTC (everything downstream)
- market_timestamp_semantics: UTC (exchange APIs, verified in parquet dtype)
- pit_rule: close_time < as_of (strict)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

# Make the repo root importable when run as `python scripts/write_dataset_meta.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.state_features import FEATURE_VERSION  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/behavior_dataset_full.json")
    parser.add_argument("--meta", default="data/processed/behavior_dataset_full.meta.json")
    args = parser.parse_args()

    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    for s in sorted(data, key=lambda s: (s["timestamp"], s["symbol"], s["kol"])):
        digest.update(f"{s['timestamp']}|{s['symbol']}|{s['kol']}|{s['side']}".encode())

    meta = {
        "dataset": "behavior_dataset_full",
        "sample_count": len(data),
        "trader_count": len({s["kol"] for s in data}),
        "symbol_count": len({s["symbol"] for s in data}),
        "time_start": min(s["timestamp"] for s in data),
        "time_end": max(s["timestamp"] for s in data),
        "action_distribution": dict(Counter(s["side"] for s in data)),
        "trader_distribution": dict(Counter(s["kol"] for s in data)),
        "symbol_distribution": dict(Counter(s["symbol"] for s in data)),
        "monthly_distribution": dict(Counter(s["timestamp"][:7] for s in data)),
        "feature_version": FEATURE_VERSION,
        "data_version": digest.hexdigest()[:16],
        # --- time semantics (explicit, business-critical) ---
        "kol_timestamp_semantics": "Asia/Shanghai",
        "internal_timestamp_semantics": "UTC",
        "market_timestamp_semantics": "UTC",
        "pit_rule": "close_time < as_of",
        "incomplete_samples": [
            {"kol": s["kol"], "symbol": s["symbol"], "timestamp": s["timestamp"],
             "reason": "no kline data on any venue"}
            for s in data if not any(v for v in s["market_state"].values())
        ],
    }
    Path(args.meta).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "symbol_distribution"},
                     ensure_ascii=False, indent=2))
    print("symbol_distribution:", meta["symbol_distribution"])


if __name__ == "__main__":
    main()
