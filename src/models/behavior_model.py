from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from src.dataset.behavior_dataset import BehaviorSample


class BehaviorModel(ABC):
    @abstractmethod
    def fit(self, samples: list[BehaviorSample]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, samples: list[BehaviorSample]) -> list[str]:
        raise NotImplementedError


@dataclass(slots=True)
class BaselineBehaviorModel(BehaviorModel):
    default_side: str = "LONG"

    def fit(self, samples: list[BehaviorSample]) -> None:
        if not samples:
            return
        counts = Counter(sample.side for sample in samples)
        ranked = counts.most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            # Tie between sides: keep the current default instead of crashing
            # (statistics.mode raises StatisticsError on ties).
            return
        self.default_side = ranked[0][0]

    def predict(self, samples: list[BehaviorSample]) -> list[str]:
        return [self.default_side for _ in samples]


@dataclass(slots=True)
class ReplayValidator:
    model: BehaviorModel

    def run(self, samples: Iterable[BehaviorSample]) -> list[dict[str, str]]:
        sample_list = list(samples)
        predictions = self.model.predict(sample_list)
        outputs: list[dict[str, str]] = []
        for sample, prediction in zip(sample_list, predictions, strict=True):
            outputs.append(
                {
                    "timestamp": sample.timestamp,
                    "kol": sample.kol,
                    "symbol": sample.symbol,
                    "predicted_side": prediction,
                    "actual_side": sample.side,
                }
            )
        return outputs
