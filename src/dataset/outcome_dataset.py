"""Outcome dataset: after-the-fact trade results, for evaluation ONLY.

Architectural rule (prompt.txt): Behavior Dataset must never import or
auto-read the Outcome Dataset. This separation is enforced by a test in
tests/test_no_future_leakage.py. Outcome data may only be consumed after
a prediction has been recorded (see src/replay/causal_replay.py).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from src.data.kol.normalizer import TradeOutcome


@dataclass(slots=True)
class OutcomeDataset:
    """Container for TradeOutcome records with symbol/time lookup."""

    outcomes: list[TradeOutcome]

    @classmethod
    def from_normalizer(cls, outcomes: Iterable[TradeOutcome]) -> "OutcomeDataset":
        return cls(outcomes=list(outcomes))

    @classmethod
    def from_json(cls, path: str | Path) -> "OutcomeDataset":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("outcomes", [])
        fields = set(TradeOutcome.__dataclass_fields__)
        return cls(
            [TradeOutcome(**{k: v for k, v in row.items() if k in fields}) for row in rows]
        )

    @classmethod
    def from_csv(cls, path: str | Path) -> "OutcomeDataset":
        """Read a normalized CSV that may mix event and outcome columns."""
        from src.data.kol.normalizer import read_events

        _, outcomes = read_events(path)
        return cls(outcomes=outcomes)

    def lookup(self, symbol: str, opened_at_utc: str) -> Optional[TradeOutcome]:
        """Outcome for an open at the given UTC ISO timestamp (exact match)."""
        for outcome in self.outcomes:
            if outcome.symbol == symbol and outcome.opened_at_utc == opened_at_utc:
                return outcome
        return None
