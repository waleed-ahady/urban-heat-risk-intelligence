from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from urban_heat_risk.features.daily import ANOMALY_FEATURES


@dataclass
class FittedAnomalyModel:
    name: str
    pipeline: Pipeline
    training_scores: np.ndarray

    def score(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        matrix = frame[ANOMALY_FEATURES].astype(float)
        estimator = self.pipeline.named_steps["model"]
        transformed = self.pipeline.named_steps["scaler"].transform(matrix)
        raw = -estimator.score_samples(transformed)
        percentile = _empirical_percentile(raw, self.training_scores)
        return raw, percentile


@dataclass
class AnomalyModelBundle:
    isolation_forest: FittedAnomalyModel
    robust_covariance: FittedAnomalyModel
    feature_names: list[str]
    metadata: dict[str, Any]

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        scored = frame.copy()
        iso_raw, iso_pct = self.isolation_forest.score(scored)
        cov_raw, cov_pct = self.robust_covariance.score(scored)
        scored["isolation_forest_raw"] = iso_raw
        scored["isolation_forest_percentile"] = iso_pct
        scored["robust_covariance_raw"] = cov_raw
        scored["robust_covariance_percentile"] = cov_pct
        scored["anomaly_score"] = 100 * (0.70 * iso_pct + 0.30 * cov_pct)
        scored["model_disagreement"] = 100 * np.abs(iso_pct - cov_pct)
        return scored


def fit_model_bundle(
    training_frame: pd.DataFrame,
    contamination: float = 0.03,
    random_state: int = 42,
) -> AnomalyModelBundle:
    matrix = training_frame[ANOMALY_FEATURES].astype(float)
    if matrix.isna().any().any():
        raise ValueError("Anomaly feature matrix contains missing values")

    iso_pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            (
                "model",
                IsolationForest(
                    n_estimators=350,
                    contamination=contamination,
                    max_samples="auto",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    iso_pipeline.fit(matrix)
    iso_scaled = iso_pipeline.named_steps["scaler"].transform(matrix)
    iso_training_scores = -iso_pipeline.named_steps["model"].score_samples(iso_scaled)

    covariance_pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            (
                "model",
                EllipticEnvelope(
                    contamination=contamination,
                    random_state=random_state,
                    support_fraction=0.70,
                ),
            ),
        ]
    )
    covariance_pipeline.fit(matrix)
    covariance_scaled = covariance_pipeline.named_steps["scaler"].transform(matrix)
    covariance_training_scores = -covariance_pipeline.named_steps["model"].score_samples(
        covariance_scaled
    )

    return AnomalyModelBundle(
        isolation_forest=FittedAnomalyModel(
            "Isolation Forest", iso_pipeline, np.sort(iso_training_scores)
        ),
        robust_covariance=FittedAnomalyModel(
            "Robust Covariance", covariance_pipeline, np.sort(covariance_training_scores)
        ),
        feature_names=list(ANOMALY_FEATURES),
        metadata={
            "contamination": contamination,
            "random_state": random_state,
            "training_rows": int(len(matrix)),
            "ensemble_weights": {
                "isolation_forest": 0.70,
                "robust_covariance": 0.30,
            },
        },
    )


def evaluate_bundle(bundle: AnomalyModelBundle, evaluation_frame: pd.DataFrame) -> dict[str, Any]:
    if "proxy_extreme" not in evaluation_frame:
        raise ValueError("Evaluation frame requires proxy_extreme labels")
    scored = bundle.score(evaluation_frame)
    y_true = scored["proxy_extreme"].astype(int).to_numpy()
    metrics: dict[str, Any] = {
        "rows": int(len(scored)),
        "positive_rate": float(y_true.mean()),
    }
    for model_name, score_column in {
        "isolation_forest": "isolation_forest_percentile",
        "robust_covariance": "robust_covariance_percentile",
        "ensemble": "anomaly_score",
    }.items():
        scores = scored[score_column].to_numpy()
        metrics[model_name] = _binary_metrics(y_true, scores)
    metrics["model_score_correlation"] = float(
        scored["isolation_forest_percentile"].corr(scored["robust_covariance_percentile"])
    )
    return metrics


def _binary_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": None, "average_precision": None, "precision_at_top_3pct": None}
    count = max(1, int(np.ceil(len(scores) * 0.03)))
    top_indices = np.argsort(scores)[-count:]
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "precision_at_top_3pct": float(y_true[top_indices].mean()),
    }


def _empirical_percentile(values: np.ndarray, sorted_reference: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_reference, values, side="right")
    return np.clip(positions / max(len(sorted_reference), 1), 0.0, 1.0)
