"""Real-behavior feedback: prediction log + error memory.

The live prediction does not know the future KOL action. When the real
action arrives, record_actual_kol_action() matches it to the stored
prediction and files a feedback record (correct AND incorrect — both are
kept, per prompt.txt). Feedback NEVER updates model parameters; retraining
is a separate, threshold-triggered decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.models.training import TrainingConfig


def _parse_ts(value: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(slots=True)
class PredictionLog:
    """Append-only store of live predictions (JSON lines)."""

    path: Path
    entries: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.entries = [
                json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]

    def record(self, entry: dict) -> None:
        self.entries.append(entry)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def find(self, timestamp, symbol: str) -> Optional[dict]:
        """Match a stored prediction by instant + symbol."""
        target = _parse_ts(str(timestamp))
        for entry in reversed(self.entries):
            entry_ts = _parse_ts(entry.get("timestamp"))
            if target is not None and entry_ts == target and entry.get("symbol") == symbol:
                return entry
        return None


@dataclass(slots=True)
class ErrorMemory:
    """Append-only feedback store: correct AND incorrect predictions.

    Named per prompt.txt terminology; it keeps both outcomes so retraining
    never over-fits to errors alone.
    """

    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, feedback: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(feedback, ensure_ascii=False) + "\n")

    def new_count(self) -> int:
        if not self.path.exists():
            return 0
        return sum(1 for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip())

    def should_retrain(self, config: TrainingConfig) -> bool:
        """Accumulated-feedback threshold (configurable, not hard-coded)."""
        return self.new_count() >= config.retrain_min_new_samples


def record_actual_kol_action(
    prediction_log: PredictionLog,
    error_memory: ErrorMemory,
    timestamp,
    symbol: str,
    actual_action: str,
) -> Optional[dict]:
    """Match the real KOL action to a stored prediction and file feedback.

    Returns the feedback record, or None when no prediction matches.
    Never touches a model: one wrong prediction can not trigger retraining.
    """
    entry = prediction_log.find(timestamp, symbol)
    if entry is None:
        return None

    feedback = {
        **entry,
        "actual_action": actual_action.upper(),
        "correct": entry["predicted_action"] == actual_action.upper(),
        "feedback_recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    error_memory.add(feedback)
    return feedback
