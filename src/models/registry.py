"""Model version registry with Champion / Challenger discipline.

A newly trained model is only a CHALLENGER. It is promoted to CHAMPION
only after beating the incumbent on a fixed, clean test set with no
per-class collapse. Every artifact records its model/training-data/
feature versions and metrics, so any prediction is traceable to the
exact model that made it.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.dataset.behavior_dataset import BehaviorSample
from src.features.state_features import StateFeatureAssembler
from src.models.behavior_model import BehaviorModel
from src.models.training import EvaluationReport, evaluate_behavior_model


@dataclass(slots=True)
class ModelArtifact:
    model: BehaviorModel
    model_version: str
    training_data_version: str
    feature_version: str
    created_at: str
    validation_metrics: dict
    test_metrics: Optional[dict] = None

    def meta(self) -> dict:
        return {
            "model_version": self.model_version,
            "training_data_version": self.training_data_version,
            "feature_version": self.feature_version,
            "created_at": self.created_at,
            "validation_metrics": self.validation_metrics,
            "test_metrics": self.test_metrics,
        }


@dataclass(slots=True)
class ChallengeConfig:
    """Promotion criteria — all configurable, nothing hard-coded."""

    min_accuracy_improvement: float = 0.01
    min_f1_improvement: float = 0.0
    min_class_recall: float = 0.0  # no per-class collapse
    min_test_accuracy: float = 0.0


@dataclass(slots=True)
class ChallengeDecision:
    promoted: bool
    reason: str
    champion_metrics: dict
    challenger_metrics: dict

    def to_dict(self) -> dict:
        return {
            "promoted": self.promoted,
            "reason": self.reason,
            "champion_metrics": self.champion_metrics,
            "challenger_metrics": self.challenger_metrics,
        }


class ModelRegistry:
    """Persist model artifacts; keep track of the current champion."""

    def __init__(self, registry_dir: str | Path) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def _artifact_dir(self, version: str) -> Path:
        return self.registry_dir / version

    def save(self, artifact: ModelArtifact) -> None:
        artifact_dir = self._artifact_dir(artifact.model_version)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with (artifact_dir / "model.pkl").open("wb") as f:
            pickle.dump(artifact.model, f)
        (artifact_dir / "meta.json").write_text(
            json.dumps(artifact.meta(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self, version: str) -> ModelArtifact:
        artifact_dir = self._artifact_dir(version)
        with (artifact_dir / "model.pkl").open("rb") as f:
            model = pickle.load(f)
        meta = json.loads((artifact_dir / "meta.json").read_text(encoding="utf-8"))
        return ModelArtifact(model=model, **{k: v for k, v in meta.items() if k != "model"})

    def list_versions(self) -> list[str]:
        return sorted(p.name for p in self.registry_dir.iterdir() if p.is_dir())

    def current_champion(self) -> Optional[ModelArtifact]:
        champion_file = self.registry_dir / "champion.json"
        if not champion_file.exists():
            return None
        version = json.loads(champion_file.read_text(encoding="utf-8"))["champion"]
        return self.load(version)

    def set_champion(self, version: str) -> None:
        (self.registry_dir / "champion.json").write_text(
            json.dumps({"champion": version, "promoted_at": datetime.now(timezone.utc).isoformat()}),
            encoding="utf-8",
        )


def evaluate_challenger(
    champion: ModelArtifact,
    challenger: ModelArtifact,
    test_samples: list[BehaviorSample],
    assembler: StateFeatureAssembler,
    config: Optional[ChallengeConfig] = None,
) -> ChallengeDecision:
    """Compare challenger vs champion on the fixed, clean test set.

    Promotion requires: clearly better accuracy, no worse F1, no class
    recall collapse, and a minimum absolute test accuracy.
    """
    config = config or ChallengeConfig()
    champion_report: EvaluationReport = evaluate_behavior_model(
        champion.model, test_samples, assembler
    )
    challenger_report: EvaluationReport = evaluate_behavior_model(
        challenger.model, test_samples, assembler
    )
    champion_metrics = champion_report.to_dict()
    challenger_metrics = challenger_report.to_dict()

    reasons: list[str] = []
    ok_accuracy = (
        challenger_report.accuracy >= champion_report.accuracy + config.min_accuracy_improvement
    )
    if not ok_accuracy:
        reasons.append(
            f"accuracy {challenger_report.accuracy:.3f} < champion {champion_report.accuracy:.3f} + {config.min_accuracy_improvement}"
        )
    ok_f1 = challenger_report.macro_f1 >= champion_report.macro_f1 + config.min_f1_improvement
    if not ok_f1:
        reasons.append(f"f1 {challenger_report.macro_f1:.3f} < champion {champion_report.macro_f1:.3f}")
    min_recall = min(
        (m["recall"] for m in challenger_report.per_class.values()), default=0.0
    )
    ok_recall = min_recall >= config.min_class_recall
    if not ok_recall:
        reasons.append(f"min class recall {min_recall:.3f} < {config.min_class_recall}")
    ok_floor = challenger_report.accuracy >= config.min_test_accuracy
    if not ok_floor:
        reasons.append(f"accuracy {challenger_report.accuracy:.3f} < floor {config.min_test_accuracy}")

    promoted = ok_accuracy and ok_f1 and ok_recall and ok_floor
    return ChallengeDecision(
        promoted=promoted,
        reason="; ".join(reasons) if reasons else "challenger beats champion on all criteria",
        champion_metrics=champion_metrics,
        challenger_metrics=challenger_metrics,
    )
