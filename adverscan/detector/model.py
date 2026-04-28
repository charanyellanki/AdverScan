"""
Binary adversarial-vs-clean sklearn pipelines with estimator selection driven by macro-F1 validation metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from numpy.typing import NDArray

from sklearn.base import clone

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score, precision_recall_fscore_support

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler


ClassifierBackend = Literal["logistic_regression", "random_forest", "xgboost"]

try:

    import xgboost as xgb

except ImportError:

    xgb = None


logger = logging.getLogger(__name__)


def compute_split_metrics(y_true: NDArray[np.integer], y_pred: NDArray[np.integer]) -> dict[str, float]:
    """

    Produce scalar summaries with emphasis on positives (adversarial class ``1``).

    """

    macro = float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0))

    positives = precision_recall_fscore_support(y_true, y_pred, labels=[1], zero_division=0)

    precision_adv = float(positives[0][0])

    recall_adv = float(positives[1][0])

    f1_adv = float(positives[2][0])

    return {
        "macro_f1": macro,
        "f1_adv_positive": f1_adv,
        "precision_adv": precision_adv,

        "recall_adv": recall_adv,

    }


def logistic_pipeline(seed: int) -> Pipeline:

    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("cls", LogisticRegression(max_iter=6000, class_weight="balanced", random_state=seed)),
        ]

    )


def rf_estimator(seed: int) -> RandomForestClassifier:

    return RandomForestClassifier(
        class_weight="balanced_subsample",

        n_estimators=400,

        max_depth=16,

        min_samples_leaf=2,

        random_state=seed,

        n_jobs=-1,

    )


def xgb_estimator(seed: int) -> Any:

    if xgb is None:

        raise RuntimeError("xgboost not installed")

    return xgb.XGBClassifier(
        n_estimators=500,

        max_depth=8,

        learning_rate=0.05,

        subsample=0.92,

        colsample_bytree=0.92,

        reg_lambda=1.25,

        objective="binary:logistic",

        eval_metric="logloss",

        random_state=seed,

        n_jobs=-1,

        tree_method="hist",

    )


def candidate_pool(seed: int) -> dict[str, Any]:

    cand: dict[str, Any] = {

        "logistic_regression": logistic_pipeline(seed),

        "random_forest": rf_estimator(seed),

    }

    cand["xgboost"] = xgb_estimator(seed) if xgb is not None else None

    return {slug: clf for slug, clf in cand.items() if clf is not None}


@dataclass(slots=True)
class TrainingReport:

    """Summary payload returned beside trained detector snapshots."""

    best_backend: ClassifierBackend
    leaderboard: dict[str, dict[str, float]]
    persisted_path: Path | None


@dataclass(slots=True)

class AdversarialDetector:

    backend: ClassifierBackend
    pipeline: Any
    val_metrics: dict[str, Any]
    train_metrics: dict[str, Any]

    def fit(self, feats: NDArray[np.floating], ys: NDArray[np.integer]) -> None:
        tgt = ys.ravel().astype(np.int64, copy=False)
        self.pipeline.fit(feats, tgt)
        preds = self.pipeline.predict(feats)
        self.train_metrics = compute_split_metrics(tgt, preds)

    def predict_proba(self, feats: NDArray[np.floating]) -> NDArray[np.floating]:

        return self.pipeline.predict_proba(feats)

    def predict_adversarial_score(self, feats: NDArray[np.floating]) -> NDArray[np.floating]:

        probs = self.predict_proba(feats)

        if probs.ndim == 1:

            return probs.astype(np.float32)

        return probs[:, -1]

    def save(self, path: str | Path) -> None:

        target = Path(path)

        blob = dict(
            backend=self.backend,

            pipeline=self.pipeline,

            val_metrics=dict(self.val_metrics),

            train_metrics=dict(self.train_metrics),

        )

        target.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(blob, target)

    @classmethod

    def load(cls, path: str | Path) -> AdversarialDetector:

        blob = joblib.load(Path(path))

        return cls(

            backend=blob["backend"],

            pipeline=blob["pipeline"],

            val_metrics=dict(blob.get("val_metrics", {})),

            train_metrics=dict(blob.get("train_metrics", {})),

        )


def train_and_select_classifier(
    X: NDArray[np.floating],
    y: NDArray[np.integer],
    *,
    val_fraction: float = 0.2,

    save_path: str | Path | None = None,

    random_state: int = 13,

    log: logging.Logger | None = None,
) -> tuple[AdversarialDetector, TrainingReport]:
    lg = log or logger

    ys = np.asarray(y).astype(np.int64, copy=False).ravel()

    feats = np.asarray(X).astype(np.float32, copy=False)

    strata = ys if ys.size > 10 else None

    X_train, X_val, y_train, y_val = train_test_split(feats, ys, stratify=strata, test_size=val_fraction, random_state=random_state)

    pool = candidate_pool(random_state)

    leaderboard_vals: dict[str, dict[str, float]] = {}

    for slug, template in pool.items():

        trial = clone(template)

        trial.fit(X_train, y_train)

        preds_val = trial.predict(X_val)

        metrics = compute_split_metrics(y_val, preds_val)

        leaderboard_vals[slug] = metrics

        lg.info("[%s] val macro=%.6f adversarial macro-F1-pos=%.6f", slug, metrics["macro_f1"], metrics["f1_adv_positive"])

    best_slug = max(leaderboard_vals, key=lambda name: leaderboard_vals[name]["macro_f1"])

    best_template = clone(pool[best_slug])

    final_clf = clone(pool[best_slug])

    best_template.fit(X_train, y_train)

    preds_val_candidate = best_template.predict(X_val)

    final_clf.fit(feats, ys)

    backend: ClassifierBackend = best_slug  # type: ignore[arg-type]

    detector = AdversarialDetector(

        backend=backend,

        pipeline=final_clf,

        train_metrics=dict(compute_split_metrics(y_train, best_template.predict(X_train))),
        val_metrics=dict(

            leaderboard_vals[best_slug]

            | {

                **{"best_estimator_slug": best_slug},

                **{"full_leaderboard_snapshot": leaderboard_vals},

                **{"holdout_predictions_metrics": compute_split_metrics(y_val, preds_val_candidate)},
            },

        ),

    )

    report = TrainingReport(best_backend=detector.backend, leaderboard=leaderboard_vals, persisted_path=None)

    if save_path:

        outp = Path(save_path)

        detector.save(outp)
        report.persisted_path = outp

        lg.info("Saved detector `%s` to `%s`", detector.backend, outp)

    return detector, report


__all__ = [
    "AdversarialDetector",
    "TrainingReport",

    "train_and_select_classifier",

    "compute_split_metrics",

]
