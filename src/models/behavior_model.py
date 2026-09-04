from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from statistics import mode
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
        sides = [sample.side for sample in samples]
        self.default_side = mode(sides)

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
