"""Causal historical replay.

Strict per-step discipline (prompt.txt):

    OBSERVE  ->  PREDICT  ->  RECORD  ->  ADVANCE TIME  ->  EVALUATE OUTCOME

- Before any prediction the replay scans every sample with the leakage
  detector and refuses to run (FAIL, not warning) on any violation.
- Samples must arrive in chronological order; the replay never sorts them
  silently, because silently reordering can mask leakage in the caller.
- The outcome lookup callback is invoked only AFTER the prediction for
  that timestamp has been recorded: the model cannot see future results.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from src.data.leakage_detector import LeakageReport, scan_samples
from src.dataset.behavior_dataset import BehaviorSample
from src.models.behavior_model import BehaviorModel


class ReplayLeakageError(RuntimeError):
    """Raised when replay input fails the leakage scan."""


@dataclass(slots=True)
class ReplayRecord:
    trader_id: str
    symbol: str
    action_timestamp: str
    predicted_side: str
    actual_side: str
    # Attached only after the prediction was recorded (evaluation phase).
    outcome: Optional[dict] = None


@dataclass(slots=True)
class CausalReplay:
    model: BehaviorModel

    def run(
        self,
        samples: list[BehaviorSample],
        outcome_lookup: Optional[Callable[[str, str, str], Optional[dict]]] = None,
    ) -> list[ReplayRecord]:
        """Replay chronologically; outcome_lookup(trader_id, symbol, action_ts)."""
        if not samples:
            return []

        # Chronological discipline: caller must provide sorted samples.
        timestamps = [sample.timestamp for sample in samples]
        if timestamps != sorted(timestamps):
            raise ValueError(
                "Replay samples must be in chronological order; "
                "sorting them silently could mask leakage in the caller"
            )

        # FAIL-fast on any future-information leakage in the input.
        report: LeakageReport = scan_samples([asdict(sample) for sample in samples])
        if not report.ok:
            raise ReplayLeakageError(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))

        records: list[ReplayRecord] = []
        for sample in samples:
            # OBSERVE: the sample IS the frozen state at T (no live reads).
            state_at_t = asdict(sample)

            # PREDICT: model sees only state_at_t.
            prediction = self.model.predict([sample])[0]

            # RECORD: prediction is stored before anything future is touched.
            record = ReplayRecord(
                trader_id=sample.kol,
                symbol=sample.symbol,
                action_timestamp=sample.timestamp,
                predicted_side=prediction,
                actual_side=sample.side,
            )
            records.append(record)

            # ADVANCE TIME -> EVALUATE OUTCOME: only now may future data open.
            if outcome_lookup is not None:
                record.outcome = outcome_lookup(sample.kol, sample.symbol, sample.timestamp)

        return records
