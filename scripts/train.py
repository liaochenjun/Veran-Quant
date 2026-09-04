from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dataset.behavior_dataset import BehaviorDataset, BehaviorSample
from src.models.behavior_model import BaselineBehaviorModel, ReplayValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train behavior cloning baseline")
    parser.add_argument("--dataset", required=True)
    return parser.parse_args()


def load_dataset(path: Path) -> BehaviorDataset:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    samples = [BehaviorSample(**item) for item in raw]
    samples = sorted(samples, key=lambda s: s.timestamp)
    return BehaviorDataset(samples=samples)


def main() -> None:
    args = parse_args()
    dataset = load_dataset(Path(args.dataset))
    train, val, test = dataset.chronological_split()

    model = BaselineBehaviorModel()
    model.fit(train)

    validator = ReplayValidator(model=model)
    _ = validator.run(val)
    _ = validator.run(test)


if __name__ == "__main__":
    main()
