"""Traditional-ML baseline models for behavior cloning.

Models: logistic regression (imputer -> StandardScaler -> LR), random
forest, LightGBM, XGBoost. Discipline guarantees:

- the imputer and the scaler are fitted INSIDE ``fit`` on the current
  train data only (never on validation/test, never on the full dataset);
- walk-forward folds each fit a fresh model, so preprocessing is per-fold;
- features that are all-NaN on the TRAIN set are dropped from the model
  feature matrix (train-only determination; the original dataset is never
  modified);
- class_weight variants are explicit, separate experiments.
"""

from __future__ import annotations

from typing import Optional

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.dataset.behavior_dataset import BehaviorSample
from src.features.state_features import StateFeatureAssembler
from src.models.behavior_model import BehaviorModel
from src.models.prediction import Prediction


class _SklearnBehaviorModel(BehaviorModel):
    """Shared machinery: features -> imputer -> estimator -> Prediction."""

    def __init__(self, assembler: StateFeatureAssembler, estimator) -> None:
        self.assembler = assembler
        self.estimator = estimator
        self.classes: list[str] = ["LONG", "SHORT"]  # fallback until fit
        self._feature_keys: list[str] = []
        self._all_nan_keys: list[str] = []  # dropped at fit (train-only decision)

    @staticmethod
    def _build_xy(samples: list[BehaviorSample], assembler) -> tuple[list[dict], list[str]]:
        return [assembler.from_sample(s) for s in samples], [s.action for s in samples]

    @staticmethod
    def _feature_matrix(
        states: list[dict], keys: list[str] | None = None
    ) -> tuple[list[str], list[list[float]]]:
        if keys is None:
            keys = sorted({k for state in states for k in state})
        return keys, [[float(state.get(k, float("nan"))) for k in keys] for state in states]

    def _prepare_features(self, features: list[dict]) -> tuple[list[str], list[list[float]]]:
        """Canonical train-time key set, minus train-all-NaN features."""
        all_keys = sorted({k for feature in features for k in feature})
        _, full_matrix = self._feature_matrix(features, all_keys)
        self._all_nan_keys = [
            key for key, col in zip(all_keys, zip(*full_matrix))
            if all(v != v for v in col)  # every value is NaN
        ]
        self._feature_keys = [k for k in all_keys if k not in self._all_nan_keys]
        return self._feature_matrix(features, self._feature_keys)

    def fit(self, samples: list[BehaviorSample]) -> None:
        if not samples:
            return
        features, labels = self._build_xy(samples, self.assembler)
        self.classes = sorted(set(labels))
        if len(self.classes) < 2:
            # Single-class split: no discriminative fitting is possible;
            # predict_proba falls back to a deterministic prior.
            self.estimator = None
            return
        _, matrix = self._prepare_features(features)
        self._configure_class_weight(labels)
        self.estimator.fit(matrix, labels)

    def _configure_class_weight(self, labels: list[str]) -> None:
        """Hook for class-weight aware estimators (XGB etc.)."""

    def predict(self, samples: list[BehaviorSample]) -> list[str]:
        features, _ = self._build_xy(samples, self.assembler)
        return [p.predicted_action for p in self.predict_proba(features)]

    def predict_proba(self, states: list[dict]) -> list[Prediction]:
        if self.estimator is None or not hasattr(self.estimator, "classes_"):
            # unfitted or single-class train: deterministic prior
            prior = "LONG" if len(self.classes) != 1 else self.classes[0]
            if prior == "LONG":
                return [Prediction(1.0, 0.0, "LONG", 1.0) for _ in states]
            return [Prediction(0.0, 1.0, "SHORT", 1.0) for _ in states]
        _, matrix = self._feature_matrix(states, self._feature_keys)
        proba = self.estimator.predict_proba(matrix)
        trained_classes = list(self.estimator.classes_)

        predictions = []
        for row in proba:
            prob_by_class = dict(zip(trained_classes, [float(p) for p in row]))
            long_p = prob_by_class.get("LONG", 0.0)
            short_p = prob_by_class.get("SHORT", 0.0)
            total = long_p + short_p
            if total <= 0:
                long_p, short_p = 1.0, 0.0
            else:
                long_p, short_p = long_p / total, short_p / total
            predicted = "LONG" if long_p >= short_p else "SHORT"
            confidence = long_p if predicted == "LONG" else short_p
            predictions.append(Prediction(long_p, short_p, predicted, confidence))
        return predictions


class LogisticRegressionModel(_SklearnBehaviorModel):
    """LR with imputer -> StandardScaler -> logistic regression.

    The scaler fixes the convergence warning from unstandardized inputs
    and makes coefficient magnitudes comparable across features.
    """

    def __init__(
        self,
        assembler: StateFeatureAssembler,
        max_iter: int = 2000,
        random_state: int = 0,
        class_weight: Optional[object] = None,
    ):
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=max_iter,
                        random_state=random_state,
                        class_weight=class_weight,
                    ),
                ),
            ]
        )
        super().__init__(assembler, estimator)


class RandomForestModel(_SklearnBehaviorModel):
    """Random forest (median imputation; trees need no scaling)."""

    def __init__(
        self,
        assembler: StateFeatureAssembler,
        n_estimators: int = 300,
        random_state: int = 0,
        n_jobs: int = -1,
        class_weight: Optional[object] = None,
    ):
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        random_state=random_state,
                        n_jobs=n_jobs,
                        class_weight=class_weight,
                    ),
                ),
            ]
        )
        super().__init__(assembler, estimator)


class LightGBMModel(_SklearnBehaviorModel):
    """LightGBM baseline, conservative defaults."""

    def __init__(
        self,
        assembler: StateFeatureAssembler,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        random_state: int = 0,
        n_jobs: int = -1,
        class_weight: Optional[str] = None,
    ):
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        random_state=random_state,
                        n_jobs=n_jobs,
                        class_weight=class_weight,
                        verbosity=-1,
                    ),
                ),
            ]
        )
        super().__init__(assembler, estimator)


class XGBoostModel(_SklearnBehaviorModel):
    """XGBoost baseline, conservative defaults.

    class_weight="balanced" maps to scale_pos_weight computed from the
    TRAIN labels only (positive class = SHORT, classes sorted LONG<SHORT).
    """

    def __init__(
        self,
        assembler: StateFeatureAssembler,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        random_state: int = 0,
        n_jobs: int = -1,
        class_weight: Optional[str] = None,
    ):
        self._class_weight = class_weight
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        random_state=random_state,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        )
        super().__init__(assembler, estimator)

    def _configure_class_weight(self, labels: list[str]) -> None:
        if self._class_weight == "balanced":
            n_long = sum(1 for label in labels if label == "LONG")
            n_short = sum(1 for label in labels if label == "SHORT")
            # positive class is SHORT (numeric label 1)
            self.estimator.named_steps["model"].set_params(
                scale_pos_weight=(n_long / n_short) if n_short else 1.0
            )

    # xgboost 3.x requires numeric labels when scale_pos_weight is used;
    # map LONG -> 0, SHORT -> 1 and translate predictions back.
    _NUMERIC_LABELS = {"LONG": 0, "SHORT": 1}

    def fit(self, samples: list[BehaviorSample]) -> None:
        if not samples:
            return
        features, labels = self._build_xy(samples, self.assembler)
        self.classes = sorted(set(labels))
        if len(self.classes) < 2:
            self.estimator = None
            return
        _, matrix = self._prepare_features(features)
        numeric_labels = [self._NUMERIC_LABELS[label] for label in labels]
        self._configure_class_weight(labels)
        self.estimator.fit(matrix, numeric_labels)

    def predict_proba(self, states: list[dict]) -> list[Prediction]:
        if self.estimator is None or not hasattr(self.estimator, "classes_"):
            prior = "LONG" if len(self.classes) != 1 else self.classes[0]
            if prior == "LONG":
                return [Prediction(1.0, 0.0, "LONG", 1.0) for _ in states]
            return [Prediction(0.0, 1.0, "SHORT", 1.0) for _ in states]
        _, matrix = self._feature_matrix(states, self._feature_keys)
        proba = self.estimator.predict_proba(matrix)
        predictions = []
        for row in proba:
            long_p = float(row[0]) if proba.shape[1] == 2 else (1.0 if self.classes == ["LONG"] else 0.0)
            short_p = float(row[1]) if proba.shape[1] == 2 else (1.0 if self.classes == ["SHORT"] else 0.0)
            total = long_p + short_p
            if total <= 0:
                long_p, short_p = 1.0, 0.0
            else:
                long_p, short_p = long_p / total, short_p / total
            predicted = "LONG" if long_p >= short_p else "SHORT"
            confidence = long_p if predicted == "LONG" else short_p
            predictions.append(Prediction(long_p, short_p, predicted, confidence))
        return predictions
