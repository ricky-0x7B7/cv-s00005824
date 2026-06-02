"""Baseline classica HOG + SVM: addestramento, valutazione e persistenza."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SVMConfig:
    kernel: str = "rbf"
    C: float = 10.0
    gamma: str | float = "scale"
    class_weight: str | None = "balanced"


def build_svm(config: SVMConfig | None = None):
    """Restituisce una Pipeline sklearn: StandardScaler -> SVC (probability=True)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    cfg = config or SVMConfig()
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel=cfg.kernel,
                    C=cfg.C,
                    gamma=cfg.gamma,
                    class_weight=cfg.class_weight,
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )


def train_svm(X: np.ndarray, y: np.ndarray, config: SVMConfig | None = None):
    """Addestra la pipeline HOG+SVM sulla matrice di feature ``X`` e sulle label ``y``."""
    model = build_svm(config)
    model.fit(X, y)
    return model


def evaluate_classifier(model, X: np.ndarray, y: np.ndarray, labels=(0, 1)) -> dict:
    """Calcola accuracy/precision/recall/F1 e la confusion matrix."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    y_pred = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=list(labels)).tolist(),
        "n_samples": int(len(y)),
    }


def save_model(model, path: str | Path) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    import joblib

    return joblib.load(path)
