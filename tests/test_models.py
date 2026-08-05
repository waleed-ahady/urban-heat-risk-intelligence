from __future__ import annotations

import numpy as np
import pandas as pd

from urban_heat_risk.features.daily import ANOMALY_FEATURES
from urban_heat_risk.models.anomaly import evaluate_bundle, fit_model_bundle


def _feature_frame(rows: int = 600, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        rng.normal(0, 1, size=(rows, len(ANOMALY_FEATURES))), columns=ANOMALY_FEATURES
    )
    frame["proxy_extreme"] = 0
    extreme_indices = frame.index[-20:]
    frame.loc[extreme_indices, ANOMALY_FEATURES[:3]] += 7
    frame.loc[extreme_indices, "proxy_extreme"] = 1
    return frame


def test_two_models_and_ensemble_return_bounded_scores() -> None:
    frame = _feature_frame()
    bundle = fit_model_bundle(frame.iloc[:500], contamination=0.04, random_state=42)
    scored = bundle.score(frame.iloc[500:])
    expected = {
        "isolation_forest_percentile",
        "robust_covariance_percentile",
        "anomaly_score",
        "model_disagreement",
    }
    assert expected.issubset(scored.columns)
    assert scored["anomaly_score"].between(0, 100).all()


def test_evaluation_metrics_are_present() -> None:
    frame = _feature_frame()
    bundle = fit_model_bundle(frame.iloc[:500], contamination=0.04, random_state=42)
    metrics = evaluate_bundle(bundle, frame.iloc[500:])
    assert metrics["isolation_forest"]["roc_auc"] is not None
    assert 0 <= metrics["ensemble"]["precision_at_top_3pct"] <= 1
