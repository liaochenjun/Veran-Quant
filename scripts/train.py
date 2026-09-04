from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Make the repo root importable when run as `python scripts/train.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.behavior_dataset import BehaviorDataset, BehaviorSample
from src.models.behavior_model import BaselineBehaviorModel, ReplayValidator
from src.utils.logging import configure_logging

LOG_DIR = Path("data/logs")


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


def _accuracy(outputs: list[dict[str, str]]) -> float:
    if not outputs:
        return 0.0
    correct = sum(1 for o in outputs if o["predicted_side"] == o["actual_side"])
    return correct / len(outputs)


def main() -> None:
    args = parse_args()
    logger = configure_logging(LOG_DIR, logger_name="train")
    started = datetime.now()
    try:
        logger.info("Training started at %s", started.isoformat(timespec="seconds"))
        logger.info("Dataset: %s", args.dataset)

        dataset = load_dataset(Path(args.dataset))
        train, val, test = dataset.chronological_split()
        logger.info("Split sizes: train=%d validation=%d test=%d", len(train), len(val), len(test))

        model = BaselineBehaviorModel()
        logger.info("Model: %s (default_side=%s)", type(model).__name__, model.default_side)
        model.fit(train)

        validator = ReplayValidator(model=model)
        val_outputs = validator.run(val)
        test_outputs = validator.run(test)
        val_accuracy = _accuracy(val_outputs)
        test_accuracy = _accuracy(test_outputs)
        logger.info("Validation accuracy: %.3f (%d samples)", val_accuracy, len(val))
        logger.info("Final test accuracy: %.3f (%d samples)", test_accuracy, len(test))

        finished = datetime.now()
        logger.info(
            "Training finished at %s (duration %.1fs)",
            finished.isoformat(timespec="seconds"),
            (finished - started).total_seconds(),
        )
    except Exception:
        logger.exception("Training failed")
        raise


if __name__ == "__main__":
    main()
