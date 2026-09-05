"""Probabilistic behavior predictions.

A model must output probabilities, not just a class: SHORT 51% and
SHORT 99% mean very different things downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Prediction:
    long_probability: float
    short_probability: float
    predicted_action: str  # "LONG" | "SHORT"
    confidence: float  # probability of the predicted action

    def to_dict(self) -> dict[str, Any]:
        return {
            "long_probability": self.long_probability,
            "short_probability": self.short_probability,
            "predicted_action": self.predicted_action,
            "confidence": self.confidence,
        }
